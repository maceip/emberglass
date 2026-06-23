// Prove runtime LoRA hot-swap in the custom WebGPU runtime: load base ONCE,
// swap adapter A/B buffers live, confirm output changes per-adapter and reverts
// to base when cleared — no base reload, no requant. Also measure speed w/ LoRA.
import { QwenWGPU } from './qwgpu/runtime.js';
import { QWEN25_3B } from './qwen25.js';

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function randn(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

function buildAdapter(dev, cfg, { targets, rank, scale, seed, std }) {
  const dims = {
    'self_attn.q_proj': [cfg.hiddenSize, cfg.numHeads * cfg.headDim],
    'self_attn.k_proj': [cfg.hiddenSize, cfg.numKVHeads * cfg.headDim],
    'self_attn.v_proj': [cfg.hiddenSize, cfg.numKVHeads * cfg.headDim],
    'self_attn.o_proj': [cfg.numHeads * cfg.headDim, cfg.hiddenSize],
    'mlp.gate_proj': [cfg.hiddenSize, cfg.intermediateSize],
    'mlp.up_proj': [cfg.hiddenSize, cfg.intermediateSize],
    'mlp.down_proj': [cfg.intermediateSize, cfg.hiddenSize],
  };
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const rng = mulberry32(seed);
  const mk = (arr) => { const b = dev.createBuffer({ size: arr.byteLength, usage: S }); dev.queue.writeBuffer(b, 0, arr); return b; };
  const modules = {};
  for (let i = 0; i < cfg.numLayers; i++) for (const t of targets) {
    const [din, dout] = dims[t];
    const A = new Float32Array(rank * din); for (let j = 0; j < A.length; j++) A[j] = randn(rng) * std; // [rank][din] transposed
    const B = new Float32Array(rank * dout); for (let j = 0; j < B.length; j++) B[j] = randn(rng) * std;
    modules[`layers.${i}.${t}`] = { A: mk(A), B: mk(B), rank, scale };
  }
  return { modules };
}

window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice({ requiredFeatures: ['subgroups'], requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  dev.addEventListener?.('uncapturederror', e => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));
  const ref = await (await fetch('./ref.json')).json(); const ids = ref.ids; const cfg = QWEN25_3B;
  const rt = new QwenWGPU(dev, cfg); await rt.build('/model'); console.log('VWG built (base loaded ONCE)');

  // read the current logits vector out of the runtime (s.logits) after a prompt.
  const rbuf = dev.createBuffer({ size: cfg.vocabSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const logitsAfterPrompt = async () => {
    for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
    const enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(rt.s.logits, 0, rbuf, 0, cfg.vocabSize * 4); dev.queue.submit([enc.finish()]);
    await rbuf.mapAsync(GPUMapMode.READ); const a = new Float32Array(rbuf.getMappedRange()).slice(); rbuf.unmap(); return a;
  };
  const maxAbsDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
  const gen = async (n) => {
    for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
    let nxt = await rt.argmaxLogits(); const out = [nxt]; let pos = ids.length;
    for (let s = 0; s < n - 1; s++) { rt.token(nxt, pos); pos++; nxt = await rt.argmaxLogits(); out.push(nxt); }
    return out;
  };
  const T = ['self_attn.q_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'];
  const X = buildAdapter(dev, cfg, { targets: T, rank: 16, scale: 2.5, seed: 1, std: 0.02 });
  const Y = buildAdapter(dev, cfg, { targets: T, rank: 16, scale: 2.5, seed: 999, std: 0.02 });

  // determinism probe: 4 consecutive BASE logit reads, zero adapter activity.
  rt.clearLora();
  const P = []; for (let i = 0; i < 4; i++) P.push(await logitsAfterPrompt());
  console.log('VWG base-determinism diffs: ' + [1, 2, 3].map(i => maxAbsDiff(P[i], P[0]).toFixed(6)).join(', '));

  // logit-level proof: LoRA must measurably change logits; revert must restore them EXACTLY.
  rt.clearLora(); const Lbase = await logitsAfterPrompt();
  rt.setLora(X);   const LX1   = await logitsAfterPrompt();
  rt.clearLora(); const Lbase2 = await logitsAfterPrompt();
  rt.setLora(Y);   const LY    = await logitsAfterPrompt();
  rt.setLora(X);   const LX2   = await logitsAfterPrompt();
  const dXbase = maxAbsDiff(LX1, Lbase), dRevert = maxAbsDiff(Lbase2, Lbase);
  const dYX = maxAbsDiff(LY, LX1), dX1X2 = maxAbsDiff(LX2, LX1);
  console.log('VWG logit Δ(X,base)=' + dXbase.toFixed(4) + '  Δ(revert,base)=' + dRevert.toFixed(6) + '  Δ(Y,X)=' + dYX.toFixed(4) + '  Δ(X2,X1)=' + dX1X2.toFixed(6));

  const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  rt.clearLora(); const base1 = await gen(12);
  rt.setLora(X);   const genX  = await gen(12);
  rt.clearLora(); const base2 = await gen(12);
  rt.setLora(Y);   const genY  = await gen(12);
  rt.setLora(X);   const genX2 = await gen(12);
  console.log('VWG base   ' + JSON.stringify(base1));
  console.log('VWG ref    ' + JSON.stringify(ref.gen_ids.slice(0, 12)));
  console.log('VWG loraX  ' + JSON.stringify(genX));
  console.log('VWG loraY  ' + JSON.stringify(genY));
  console.log('VWG loraX2 ' + JSON.stringify(genX2));
  const checks = [
    ['base == HF reference (sanity)', eq(base1, ref.gen_ids.slice(0, 12))],
    ['LoRA X changes logits measurably', dXbase > 0.5],
    ['clearLora restores logits EXACTLY (no reload)', dRevert === 0],
    ['LoRA Y produces different logits than X', dYX > 0.5],
    ['re-applying X is bit-exact deterministic', dX1X2 === 0],
    ['re-applying X reproduces gen sequence', eq(genX2, genX)],
  ];
  let pass = 0; for (const [name, ok] of checks) { console.log('VWG ' + (ok ? 'PASS' : 'FAIL') + '  ' + name); if (ok) pass++; }
  console.log('VWG HOTSWAP ' + (pass === checks.length ? 'ALL PASS (' + pass + '/' + checks.length + ')' : 'FAILED ' + pass + '/' + checks.length));

  // speed with LoRA active
  rt.setLora(X); await dev.queue.onSubmittedWorkDone(); let pos = ids.length, nxt = genX[genX.length - 1];
  for (let p = 0; p < ids.length; p++) rt.token(ids[p], p); nxt = await rt.argmaxLogits();
  const t0 = performance.now(); for (let s = 0; s < 30; s++) { rt.token(nxt, pos); pos++; nxt = await rt.argmaxLogits(); }
  console.log('VWG SPEED(LoRA active) ' + (30 / ((performance.now() - t0) / 1000)).toFixed(1) + ' tok/s');
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))));
