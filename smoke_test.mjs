// Numerical proof that the Qwen2.5 forward pass runs and that LoRA hot-swap
// actually changes the output — no model download, CPU backend, tiny config.
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { QwenModel, LoraAdapter } from './qwen25.js';

await tf.setBackend('cpu');
await tf.ready();

const cfg = {
  hiddenSize: 64, numLayers: 2, numHeads: 4, numKVHeads: 2, headDim: 16,
  intermediateSize: 128, vocabSize: 100, rmsNormEps: 1e-6, ropeTheta: 1e6,
};

// build deterministic-ish random weights for every key the model reads
function randn(shape) { return tf.randomNormal(shape, 0, 0.02); }
const w = {};
w['model.embed_tokens.weight'] = randn([cfg.vocabSize, cfg.hiddenSize]);
w['model.norm.weight'] = tf.ones([cfg.hiddenSize]);
const qDim = cfg.numHeads * cfg.headDim, kvDim = cfg.numKVHeads * cfg.headDim;
for (let i = 0; i < cfg.numLayers; i++) {
  const p = `model.layers.${i}`;
  w[`${p}.input_layernorm.weight`] = tf.ones([cfg.hiddenSize]);
  w[`${p}.post_attention_layernorm.weight`] = tf.ones([cfg.hiddenSize]);
  w[`${p}.self_attn.q_proj.weight`] = randn([qDim, cfg.hiddenSize]);
  w[`${p}.self_attn.k_proj.weight`] = randn([kvDim, cfg.hiddenSize]);
  w[`${p}.self_attn.v_proj.weight`] = randn([kvDim, cfg.hiddenSize]);
  w[`${p}.self_attn.o_proj.weight`] = randn([cfg.hiddenSize, qDim]);
  w[`${p}.self_attn.q_proj.bias`] = randn([qDim]);
  w[`${p}.self_attn.k_proj.bias`] = randn([kvDim]);
  w[`${p}.self_attn.v_proj.bias`] = randn([kvDim]);
  w[`${p}.mlp.gate_proj.weight`] = randn([cfg.intermediateSize, cfg.hiddenSize]);
  w[`${p}.mlp.up_proj.weight`] = randn([cfg.intermediateSize, cfg.hiddenSize]);
  w[`${p}.mlp.down_proj.weight`] = randn([cfg.hiddenSize, cfg.intermediateSize]);
}

const model = new QwenModel(cfg, w);
const ids = tf.tensor2d([[1, 2, 3, 4, 5]], [1, 5], 'int32');

function lastLogits() {
  const emb = model.embed(ids);
  const { logits, kvCaches } = model.forward(emb, 0, null);
  const data = logits.dataSync();
  emb.dispose(); logits.dispose(); model.disposeKV(kvCaches);
  return Float32Array.from(data);
}
const argmax = (a) => { let m = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[m]) m = i; return m; };
const maxAbsDiff = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i])); return d; };

// --- base (no LoRA) ---
const base = lastLogits();
console.log(`[base]      logits len=${base.length} argmax=${argmax(base)} finite=${base.every(Number.isFinite)}`);

// --- adapter A: large delta on q_proj of layer 0 ---
const loraA = new LoraAdapter('A', {
  'layers.0.self_attn.q_proj': { A: tf.randomNormal([cfg.hiddenSize, 8], 0, 0.5), B: tf.randomNormal([8, qDim], 0, 0.5), scale: 4.0 },
  'layers.1.mlp.gate_proj':    { A: tf.randomNormal([cfg.hiddenSize, 8], 0, 0.5), B: tf.randomNormal([8, cfg.intermediateSize], 0, 0.5), scale: 4.0 },
});
model.setLora(loraA);
const withA = lastLogits();
console.log(`[lora A]    argmax=${argmax(withA)}  maxΔ vs base=${maxAbsDiff(base, withA).toFixed(4)}`);

// --- adapter B: different deltas ---
const loraB = new LoraAdapter('B', {
  'layers.0.self_attn.v_proj': { A: tf.randomNormal([cfg.hiddenSize, 8], 0, 0.5), B: tf.randomNormal([8, kvDim], 0, 0.5), scale: 4.0 },
});
model.setLora(loraB);                 // <-- hot-swap, no base reload
const withB = lastLogits();
console.log(`[lora B]    argmax=${argmax(withB)}  maxΔ vs base=${maxAbsDiff(base, withB).toFixed(4)}  maxΔ vs A=${maxAbsDiff(withA, withB).toFixed(4)}`);

// --- clear back to base ---
model.clearLora();
const back = lastLogits();
console.log(`[cleared]   argmax=${argmax(back)}  maxΔ vs base=${maxAbsDiff(base, back).toFixed(8)} (should be ~0)`);

// --- verdict ---
const ok =
  base.every(Number.isFinite) &&
  maxAbsDiff(base, withA) > 1e-3 &&
  maxAbsDiff(withA, withB) > 1e-3 &&
  maxAbsDiff(base, back) < 1e-6;
console.log(`\nRESULT: ${ok ? 'PASS' : 'FAIL'} — forward runs; LoRA swap changes output; clear restores base. tensors=${tf.memory().numTensors}`);
process.exit(ok ? 0 : 1);
