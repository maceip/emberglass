// Parse a PEFT/MLX LoRA adapter (safetensors + adapter_config.json) into the
// LoraAdapter format used by qwen25.js: per module key {A:[in,r], B:[r,out], scale}.
// Orientation is inferred from shapes (rank = the small shared dim), so it works
// across PEFT ([r,in]/[out,r]) and MLX naming variants.
import * as tf from '@tensorflow/tfjs-core';
import { LoraAdapter } from './qwen25.js';

function parseSt(buf) {
  const dv = new DataView(buf);
  const hl = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
  return { header, dataStart: 8 + hl, u8: new Uint8Array(buf) };
}
function bf16f32(u8, off, n) { const u16 = new Uint16Array(u8.buffer, u8.byteOffset + off, n); const o = new Float32Array(n); const o32 = new Uint32Array(o.buffer); for (let i = 0; i < n; i++) o32[i] = u16[i] << 16; return o; }
function f32(u8, off, n) { return new Float32Array(u8.buffer.slice(u8.byteOffset + off, u8.byteOffset + off + n * 4)); }
function readTensor(st, name) {
  const t = st.header[name]; const n = t.shape.reduce((a, b) => a * b, 1);
  const dt = t.dtype.toUpperCase();
  const arr = dt === 'BF16' ? bf16f32(st.u8, st.dataStart + t.data_offsets[0], n) : f32(st.u8, st.dataStart + t.data_offsets[0], n);
  return { arr, shape: t.shape };
}

// normalize any "...layers.5.self_attn.q_proj..." into "layers.5.self_attn.q_proj"
function moduleKey(name) {
  const m = name.match(/layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+?)(_proj)?\.(lora_[ABab]|lora_[ab]\b)/i);
  if (!m) return null;
  const sub = m[2], proj = m[3].replace(/_proj$/, '');
  return `layers.${m[1]}.${sub}.${proj}_proj`;
}
function isA(name) { return /lora_a/i.test(name); }

export async function loadLoraAdapter(files, cfg) {
  const byName = {}; for (const f of files) byName[f.name] = f;
  const stFile = files.find(f => f.name.endsWith('.safetensors'));
  if (!stFile) throw new Error('no .safetensors in adapter files');
  const cfgFile = files.find(f => /adapter_config\.json|config\.json/.test(f.name));
  let rank = 16, alpha = null;
  if (cfgFile) { const c = JSON.parse(await cfgFile.text()); rank = c.r ?? c.rank ?? c.lora_rank ?? rank; alpha = c.lora_alpha ?? c.alpha ?? null; }

  const st = parseSt(await stFile.arrayBuffer());
  const names = Object.keys(st.header).filter(k => k !== '__metadata__' && /lora_[abAB]/.test(k));
  const groups = {}; // key -> {A:{arr,shape}, B:{arr,shape}}
  for (const nm of names) {
    const key = moduleKey(nm); if (!key) continue;
    (groups[key] ||= {})[isA(nm) ? 'A' : 'B'] = readTensor(st, nm);
  }

  const modules = {};
  for (const key of Object.keys(groups)) {
    const g = groups[key]; if (!g.A || !g.B) continue;
    const r = Math.min(...g.A.shape, ...g.B.shape); // rank = smallest dim
    // want A_mine [in,r], B_mine [r,out]
    const aIn = g.A.shape[0] === r ? g.A.shape[1] : g.A.shape[0];
    let A = tf.tensor(g.A.arr, g.A.shape, 'float32');
    let B = tf.tensor(g.B.arr, g.B.shape, 'float32');
    if (A.shape[0] === r) A = tf.transpose(A);      // [r,in]->[in,r]
    if (B.shape[1] === r) B = tf.transpose(B);      // [out,r]->[r,out]
    const scale = alpha ? alpha / r : 2.0;
    modules[key] = { A: tf.keep(A), B: tf.keep(B), scale };
  }
  const name = stFile.name.replace(/\.safetensors$/, '');
  return { name, adapter: new LoraAdapter(name, modules) };
}

/** Synthetic adapter (random A/B at given rank) — for proving hot-swap end-to-end
 *  in the browser without a real adapter file. */
export function syntheticAdapter(name, cfg, rank = 8, scale = 2.0, targets = ['self_attn.q_proj', 'mlp.gate_proj']) {
  const modules = {};
  const dims = {
    'self_attn.q_proj': [cfg.hiddenSize, cfg.numHeads * cfg.headDim],
    'self_attn.k_proj': [cfg.hiddenSize, cfg.numKVHeads * cfg.headDim],
    'self_attn.v_proj': [cfg.hiddenSize, cfg.numKVHeads * cfg.headDim],
    'self_attn.o_proj': [cfg.numHeads * cfg.headDim, cfg.hiddenSize],
    'mlp.gate_proj': [cfg.hiddenSize, cfg.intermediateSize],
    'mlp.up_proj': [cfg.hiddenSize, cfg.intermediateSize],
    'mlp.down_proj': [cfg.intermediateSize, cfg.hiddenSize],
  };
  for (let i = 0; i < cfg.numLayers; i++) for (const t of targets) {
    const [din, dout] = dims[t];
    modules[`layers.${i}.${t}`] = { A: tf.keep(tf.randomNormal([din, rank], 0, 0.02)), B: tf.keep(tf.randomNormal([rank, dout], 0, 0.02)), scale };
  }
  return new LoraAdapter(name, modules);
}
