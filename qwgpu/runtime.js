// Custom pure-WebGPU Qwen2.5 decode runtime. int8 weights (per-channel scale),
// f32 norms/biases, GPU-resident KV cache, runtime-swappable LoRA (A/B f32
// buffers consumed by the GEMV kernel). No tf.js → no per-op dispatch overhead.
//
// Correctness is validated against the tf.js forward (which == HuggingFace).
import { GEMV, GEMV4, LORA_A, RMSNORM, ROPE, ATTN_PARTIAL, ATTN_COMBINE, ADD, SILUMUL, EMBED, EMBED_BUF, ARGMAX } from './kernels.js';
import { quantizeInt8RowMajor, quantizeInt4Group } from './quantize.js';
import { loadModelWeights, urlReader } from '../weights.js';

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

export class QwenWGPU {
  constructor(device, cfg) { this.dev = device; this.cfg = cfg; this.lora = null; this.bufs = {}; }

  _buf(size, usage = STORAGE) { return this.dev.createBuffer({ size, usage }); }
  _f32(arr, usage = STORAGE) { const b = this._buf(arr.byteLength, usage); this.dev.queue.writeBuffer(b, 0, arr); return b; }
  _u32(arr) { const b = this._buf(arr.byteLength, STORAGE); this.dev.queue.writeBuffer(b, 0, arr); return b; }
  _uni(arr) {
    if (!this._uniPool) { this._uniPool = []; this._uniIdx = 0; }
    let b = this._uniPool[this._uniIdx];
    if (!b) { b = this._buf(32, UNIFORM); this._uniPool[this._uniIdx] = b; }
    this._uniIdx++;
    this.dev.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return b;
  }
  _resetUni() { this._uniIdx = 0; }

  _pipe(code) {
    const m = this.dev.createShaderModule({ code });
    return this.dev.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
  }

  async build(baseUrl, onProgress = () => {}) {
    const dev = this.dev, c = this.cfg;
    this.CHUNK = 128; this.MAXBATCH = 16;
    this.pipes = { gemv: this._pipe(GEMV), loraA: this._pipe(LORA_A), rms: this._pipe(RMSNORM), rope: this._pipe(ROPE), attnP: this._pipe(ATTN_PARTIAL), attnC: this._pipe(ATTN_COMBINE), add: this._pipe(ADD), silu: this._pipe(SILUMUL), embed: this._pipe(EMBED), embedBuf: this._pipe(EMBED_BUF), argmax: this._pipe(ARGMAX), gemv4: this._pipe(GEMV4) };
    onProgress('loading f32 weights', 0);
    // Load f32 weights via tf.js-free path: reuse weights.js but on CPU arrays.
    const W = await this._loadRaw(baseUrl, onProgress);
    onProgress('quantizing to int8 + uploading', 0.5);
    this.q = {}; this.q4 = {};
    const quant4 = (name) => { const t = W[name]; const { packed, scale, groupsPerRow } = quantizeInt4Group(t.data, t.shape[0], t.shape[1], 128); this.q4[name] = { w: this._u32(packed), scale: this._f32(scale), N: t.shape[0], K: t.shape[1], gpr: groupsPerRow }; };
    const quant = (name) => { const t = W[name]; const { packed, scale } = quantizeInt8RowMajor(t.data, t.shape[0], t.shape[1]); this.q[name] = { w: this._u32(packed), scale: this._f32(scale), N: t.shape[0], K: t.shape[1] }; };
    const f32buf = (name) => { this.bufs[name] = this._f32(W[name].data); };
    quant('model.embed_tokens.weight'); // [vocab,hidden] -> int8 (embed lookup + lm_head)
    f32buf('model.norm.weight');
    const proj = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'];
    for (let i = 0; i < c.numLayers; i++) {
      const p = `model.layers.${i}`;
      f32buf(`${p}.input_layernorm.weight`); f32buf(`${p}.post_attention_layernorm.weight`);
      for (const s of proj) quant4(`${p}.${s}.weight`);
      for (const b of ['q', 'k', 'v']) f32buf(`${p}.self_attn.${b}_proj.bias`);
      if (i % 6 === 0) await new Promise(r => setTimeout(r, 0));
    }
    // Context window: thinking model emits ~4k tokens, so size generously.
    this.maxCtx = 8192;
    this._buildRope(this.maxCtx);
    // KV cache (f32) per layer
    this.kc = [], this.vc = [];
    const kvSize = c.numKVHeads * this.maxCtx * c.headDim * 4;
    for (let i = 0; i < c.numLayers; i++) { this.kc.push(this._buf(kvSize)); this.vc.push(this._buf(kvSize)); }
    // scratch buffers (reused each token)
    const H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize;
    const NSPLITMAX = Math.ceil(this.maxCtx / this.CHUNK);
    this.s = {
      hidden: this._buf(H * 4), normed: this._buf(H * 4), q: this._buf(qd * 4), k: this._buf(kvd * 4), v: this._buf(kvd * 4),
      attn: this._buf(qd * 4), tmp: this._buf(Math.max(qd, I) * 4), tmp2: this._buf(I * 4), logits: this._buf(c.vocabSize * 4),
      dummy: this._buf(64), loraD: this._buf(256 * 4), amax: this._buf(4),
      pm: this._buf(c.numHeads * NSPLITMAX * 4), pz: this._buf(c.numHeads * NSPLITMAX * 4), po: this._buf(c.numHeads * NSPLITMAX * c.headDim * 4),
      idsBuf: this._buf(this.MAXBATCH * 4),
    };
    this.idsRead = this._buf(this.MAXBATCH * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    onProgress('ready', 1);
    this._uniCache = {};
    return this;
  }

  async _loadRaw(baseUrl, onProgress) {
    // Reuse weights.js but capture raw Float32Array (no tf). Monkey: weights.js
    // returns tf tensors; instead we re-parse here for plain arrays.
    const reader = urlReader(baseUrl);
    const out = {};
    const idx = JSON.parse(await reader.text('model.safetensors.index.json'));
    const shards = [...new Set(Object.values(idx.weight_map))];
    const dec = (u8, n) => { const u16 = new Uint16Array(u8.buffer, u8.byteOffset, n); const o = new Float32Array(n); const o32 = new Uint32Array(o.buffer); for (let i = 0; i < n; i++) o32[i] = u16[i] << 16; return o; };
    for (const shard of shards) {
      const lenBuf = await reader.range(shard, 0, 8); const hl = Number(new DataView(lenBuf).getBigUint64(0, true));
      const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(await reader.range(shard, 8, 8 + hl)))); const dataStart = 8 + hl;
      for (const name of Object.keys(hdr)) {
        if (name === '__metadata__') continue;
        const t = hdr[name]; const numel = t.shape.reduce((a, b) => a * b, 1); const [s, e] = t.data_offsets;
        const buf = await reader.range(shard, dataStart + s, dataStart + e);
        out[name] = { data: dec(new Uint8Array(buf), numel), shape: t.shape };
        onProgress(name, 0.3);
      }
    }
    return out;
  }

  _buildRope(maxSeq) {
    const { headDim, ropeTheta } = this.cfg; const half = headDim / 2;
    const cos = new Float32Array(maxSeq * headDim), sin = new Float32Array(maxSeq * headDim);
    for (let p = 0; p < maxSeq; p++) for (let i = 0; i < half; i++) {
      const a = p / Math.pow(ropeTheta, (2 * i) / headDim); const cc = Math.cos(a), ss = Math.sin(a);
      cos[p * headDim + i] = cc; cos[p * headDim + half + i] = cc; sin[p * headDim + i] = ss; sin[p * headDim + half + i] = ss;
    }
    this.ropeCos = this._f32(cos); this.ropeSin = this._f32(sin); this._ropeRow = headDim * 4;
  }

  setLora(adapter) { this.lora = adapter; }   // {modules: {key:{A,B,rank,scale}}}  A:[K][rank], B:[rank][N] f32 GPUBuffers
  clearLora() { this.lora = null; }

  _bg(pipe, buffers) {
    return this.dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })) });
  }
  _dispatch(enc, pipe, bg, gx, gy=1, cat) {
    let ts;
    if (this.prof && this.prof.idx < this.prof.cap) { const i = this.prof.idx++; this.prof.cats.push(cat || 'misc'); ts = { querySet: this.prof.qs, beginningOfPassWriteIndex: 2*i, endOfPassWriteIndex: 2*i+1 }; }
    const p = enc.beginComputePass(ts ? { timestampWrites: ts } : undefined); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(gx, gy); p.end();
  }
  enableProf(cap = 700) { this.prof = { qs: this.dev.createQuerySet({ type: 'timestamp', count: cap * 2 }), cap, idx: 0, cats: [], resolve: this._buf(cap * 16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC), read: this._buf(cap * 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ) }; }
  async profToken(id, pos) {
    this._resetUni(); this.prof.idx = 0; this.prof.cats = [];
    const enc = this.dev.createCommandEncoder(); this.embedRow(enc, id); this.step(enc, id, pos);
    const n = this.prof.idx; enc.resolveQuerySet(this.prof.qs, 0, n * 2, this.prof.resolve, 0);
    enc.copyBufferToBuffer(this.prof.resolve, 0, this.prof.read, 0, n * 16);
    this.dev.queue.submit([enc.finish()]); await this.prof.read.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(this.prof.read.getMappedRange()); const sums = {};
    for (let i = 0; i < n; i++) { const us = Number(t[2*i+1] - t[2*i]) / 1000; const c = this.prof.cats[i]; sums[c] = (sums[c] || 0) + us; }
    this.prof.read.unmap(); return sums;
  }

  // y = int8-GEMV(x, q) [+bias] [+lora]. q={w,scale,N,K}. moduleKey for LoRA lookup.
  gemv(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) { // d = x@A  (rank outputs)
      const bgA = this._bg(this.pipes.loraA, [xBuf, mod.A, this.s.loraD, this._uni(new Uint32Array([q.K, mod.rank]))]);
      this._dispatch(enc, this.pipes.loraA, bgA, mod.rank, 1, 'loraA');
    }
    const meta = new ArrayBuffer(32); const dv = new DataView(meta);
    dv.setUint32(0, q.K, true); dv.setUint32(4, q.N, true); dv.setUint32(8, mod ? mod.rank : 0, true);
    dv.setUint32(12, biasBuf ? 1 : 0, true); dv.setUint32(16, mod ? 1 : 0, true);
    const gx = Math.min(q.N, 65535), gy = Math.ceil(q.N / gx); dv.setUint32(20, gx, true);
    dv.setFloat32(24, mod ? mod.scale : 0, true);
    const bg = this._bg(this.pipes.gemv, [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf, this._uni(new Uint8Array(meta))]);
    this._dispatch(enc, this.pipes.gemv, bg, gx, gy, `gemv:${q.N}x${q.K}`);
  }

  gemv4(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) { this._dispatch(enc, this.pipes.loraA, this._bg(this.pipes.loraA, [xBuf, mod.A, this.s.loraD, this._uni(new Uint32Array([q.K, mod.rank]))]), mod.rank, 1, 'loraA'); }
    const gx = Math.min(q.N, 65535), gy = Math.ceil(q.N / gx);
    const meta = new ArrayBuffer(32); const dv = new DataView(meta);
    dv.setUint32(0, q.K, true); dv.setUint32(4, q.N, true); dv.setUint32(8, mod ? mod.rank : 0, true);
    dv.setUint32(12, biasBuf ? 1 : 0, true); dv.setUint32(16, mod ? 1 : 0, true); dv.setUint32(20, gx, true);
    dv.setFloat32(24, mod ? mod.scale : 0, true); dv.setUint32(28, q.gpr, true);
    const bg = this._bg(this.pipes.gemv4, [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf, this._uni(new Uint8Array(meta))]);
    this._dispatch(enc, this.pipes.gemv4, bg, gx, gy, `g4:${q.N}x${q.K}`);
  }
  rms(enc, xBuf, gBuf, yBuf, K) {
    const u = new ArrayBuffer(8); const dv = new DataView(u); dv.setFloat32(0, K, true); dv.setFloat32(4, this.cfg.rmsNormEps, true);
    this._dispatch(enc, this.pipes.rms, this._bg(this.pipes.rms, [xBuf, gBuf, yBuf, this._uni(new Uint8Array(u))]), 1, 1, 'rms');
  }
  rope(enc, xBuf, pos, nHeads) {
    this._dispatch(enc, this.pipes.rope, this._bg(this.pipes.rope, [xBuf, this.ropeCos, this.ropeSin, this._uni(new Uint32Array([nHeads, this.cfg.headDim, pos]))]), Math.ceil(nHeads*(this.cfg.headDim/2)/256), 1, 'rope');
  }
  attn(enc, qBuf, kc, vc, oBuf, ctx) {
    const c = this.cfg, S = this.s; const nsplit = Math.ceil(ctx / this.CHUNK);
    // pass 1: per (head, ctx-chunk) partial softmax → pm/pz/po (nHeads*nsplit workgroups)
    const bgP = this._bg(this.pipes.attnP, [qBuf, kc, vc, S.pm, S.pz, S.po,
      this._uni(new Uint32Array([c.numHeads, c.numKVHeads, ctx, c.headDim])), this._uni(new Uint32Array([nsplit, this.CHUNK]))]);
    this._dispatch(enc, this.pipes.attnP, bgP, c.numHeads, nsplit, 'attnP');
    // pass 2: combine splits per head → o
    const bgC = this._bg(this.pipes.attnC, [S.pm, S.pz, S.po, oBuf, this._uni(new Uint32Array([c.numHeads, c.headDim, nsplit, 0]))]);
    this._dispatch(enc, this.pipes.attnC, bgC, c.numHeads, 1, 'attnC');
  }

  // Decode one token at absolute position `pos`. Writes logits to s.logits. Returns nothing.
  step(enc, tokenId, pos) {
    const c = this.cfg, S = this.s, hd = c.headDim, kvd = c.numKVHeads * hd;
    // embed: dequant row tokenId of embed_tokens int8 -> hidden (use gemv? no; copy+scale). Use a tiny loraA-style? Simplest: a gemv with a one-hot is overkill.
    // We do embed lookup on CPU-uploaded row: handled by caller via this.embedRow(tokenId) into S.hidden.
    for (let i = 0; i < c.numLayers; i++) {
      const p = `model.layers.${i}`;
      this.rms(enc, S.hidden, this.bufs[`${p}.input_layernorm.weight`], S.normed, c.hiddenSize);
      this.gemv4(enc, S.normed, this.q4[`${p}.self_attn.q_proj.weight`], S.q, this.bufs[`${p}.self_attn.q_proj.bias`], `layers.${i}.self_attn.q_proj`);
      this.gemv4(enc, S.normed, this.q4[`${p}.self_attn.k_proj.weight`], S.k, this.bufs[`${p}.self_attn.k_proj.bias`], `layers.${i}.self_attn.k_proj`);
      this.gemv4(enc, S.normed, this.q4[`${p}.self_attn.v_proj.weight`], S.v, this.bufs[`${p}.self_attn.v_proj.bias`], `layers.${i}.self_attn.v_proj`);
      this.rope(enc, S.q, pos, c.numHeads); this.rope(enc, S.k, pos, c.numKVHeads);
      // append k,v to cache at position pos
      enc.copyBufferToBuffer(S.k, 0, this.kc[i], pos * kvd * 4, kvd * 4);
      enc.copyBufferToBuffer(S.v, 0, this.vc[i], pos * kvd * 4, kvd * 4);
      this.attn(enc, S.q, this.kc[i], this.vc[i], S.attn, pos + 1);
      this.gemv4(enc, S.attn, this.q4[`${p}.self_attn.o_proj.weight`], S.tmp, null, `layers.${i}.self_attn.o_proj`);
      this._addInto(enc, S.hidden, S.tmp, c.hiddenSize);             // residual
      this.rms(enc, S.hidden, this.bufs[`${p}.post_attention_layernorm.weight`], S.normed, c.hiddenSize);
      this.gemv4(enc, S.normed, this.q4[`${p}.mlp.gate_proj.weight`], S.tmp, null, `layers.${i}.mlp.gate_proj`);
      this.gemv4(enc, S.normed, this.q4[`${p}.mlp.up_proj.weight`], S.tmp2, null, `layers.${i}.mlp.up_proj`);
      this._siluMul(enc, S.tmp, S.tmp2, c.intermediateSize);          // tmp = silu(gate)*up
      this.gemv4(enc, S.tmp, this.q4[`${p}.mlp.down_proj.weight`], S.normed, null, `layers.${i}.mlp.down_proj`);
      this._addInto(enc, S.hidden, S.normed, c.hiddenSize);
    }
    this.rms(enc, S.hidden, this.bufs['model.norm.weight'], S.normed, c.hiddenSize);
    this.gemv(enc, S.normed, this.q['model.embed_tokens.weight'], S.logits, null, null); // lm_head (tied)
  }

  _addInto(enc, yBuf, aBuf, n) { this._dispatch(enc, this.pipes.add, this._bg(this.pipes.add, [aBuf, yBuf, this._uni(new Uint32Array([n]))]), Math.ceil(n/256), 1, 'add'); }
  _siluMul(enc, gateBuf, upBuf, n) { this._dispatch(enc, this.pipes.silu, this._bg(this.pipes.silu, [gateBuf, upBuf, this._uni(new Uint32Array([n]))]), Math.ceil(n/256), 1, 'silu'); }
  embedRow(enc, id) { const e = this.q['model.embed_tokens.weight']; this._dispatch(enc, this.pipes.embed, this._bg(this.pipes.embed, [e.w, e.scale, this.s.hidden, this._uni(new Uint32Array([id, this.cfg.hiddenSize]))]), Math.ceil(this.cfg.hiddenSize/256), 1, 'embed'); }
  async argmaxLogits() {
    const enc = this.dev.createCommandEncoder();
    this._dispatch(enc, this.pipes.argmax, this._bg(this.pipes.argmax, [this.s.logits, this.s.amax, this._uni(new Uint32Array([this.cfg.vocabSize]))]), 1);
    const rb = this._buf(4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    enc.copyBufferToBuffer(this.s.amax, 0, rb, 0, 4); this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const id = new Uint32Array(rb.getMappedRange())[0]; rb.unmap(); rb.destroy(); return id;
  }
  // Run one token end-to-end (embed + step) and submit.
  token(id, pos) { this._resetUni(); const enc = this.dev.createCommandEncoder(); this.embedRow(enc, id); this.step(enc, id, pos); this.dev.queue.submit([enc.finish()]); }

  // embed the token id held in s.amax (GPU-resident, from a prior argmax)
  embedFromBuf(enc) { const e = this.q['model.embed_tokens.weight']; this._dispatch(enc, this.pipes.embedBuf, this._bg(this.pipes.embedBuf, [e.w, e.scale, this.s.hidden, this.s.amax, this._uni(new Uint32Array([this.cfg.hiddenSize]))]), Math.ceil(this.cfg.hiddenSize/256), 1, 'embed'); }
  // argmax(logits) -> s.amax, within the given encoder (no submit/readback)
  argmaxInto(enc) { this._dispatch(enc, this.pipes.argmax, this._bg(this.pipes.argmax, [this.s.logits, this.s.amax, this._uni(new Uint32Array([this.cfg.vocabSize]))]), 1, 1, 'argmax'); }

  // GPU-resident batched greedy decode: chains embed->step->argmax->embed for K
  // tokens in ONE submit (no per-token CPU sync), reads back K ids once. Assumes
  // s.amax holds the current token to embed. Returns the K generated ids.
  async decodeBatch(startPos, K) {
    this._resetUni();
    const enc = this.dev.createCommandEncoder();
    for (let k = 0; k < K; k++) {
      this.embedFromBuf(enc);
      this.step(enc, 0, startPos + k);
      this.argmaxInto(enc);
      enc.copyBufferToBuffer(this.s.amax, 0, this.s.idsBuf, k * 4, 4);
    }
    enc.copyBufferToBuffer(this.s.idsBuf, 0, this.idsRead, 0, K * 4);
    this.dev.queue.submit([enc.finish()]);
    await this.idsRead.mapAsync(GPUMapMode.READ);
    const ids = Array.from(new Uint32Array(this.idsRead.getMappedRange(), 0, K)); this.idsRead.unmap();
    return ids;
  }

}
