import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgpu';
import { QwenModel, QWEN25_3B } from './qwen25.js';
import { loadModelWeights, urlReader } from './weights.js';
window.run = async () => {
  await tf.setBackend('webgpu'); await tf.ready();
  const ref = await (await fetch('./ref.json')).json(); const ids = ref.ids;
  const weights = await loadModelWeights(urlReader('/model'), () => {});
  const model = new QwenModel(QWEN25_3B, weights);
  const topArg = async l => { const am = tf.argMax(tf.reshape(l, [-1])); const d = await am.data(); am.dispose(); return d[0]; };
  // prefill
  const idsT = tf.tensor2d([ids], [1, ids.length], 'int32');
  const emb = model.embed(idsT); let pf = model.forward(emb, 0, null); emb.dispose(); idsT.dispose();
  let kv = pf.kvCaches, pos = ids.length, nxt = await topArg(pf.logits); pf.logits.dispose();
  // warm-up 3 + timed 30
  const N = 30, WARM = 3;
  for (let s = 0; s < WARM + N; s++) {
    if (s === WARM) { await tf.nextFrame(); var t0 = performance.now(); }
    const tt = tf.tensor2d([[nxt]], [1, 1], 'int32'); const ee = model.embed(tt);
    const r = model.forward(ee, pos, kv); ee.dispose(); tt.dispose(); kv = r.kvCaches; pos++;
    nxt = await topArg(r.logits); r.logits.dispose();
  }
  const dt = (performance.now() - t0) / 1000;
  console.log(`BENCH ${N} decode tokens in ${dt.toFixed(2)}s = ${(N/dt).toFixed(2)} tok/s  (ctx~${pos})`);
  model.disposeKV(kv);
  console.log('BENCH DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('BENCH ERROR ' + e.message)));
