import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgpu';
import { QwenModel, QWEN25_3B } from './qwen25.js';
import { loadModelWeights, urlReader } from './weights.js';
import { syntheticAdapter } from './lora.js';
window.run = async () => {
  await tf.setBackend('webgpu'); await tf.ready();
  const ref = await (await fetch('./ref.json')).json(); const ids = ref.ids;
  const weights = await loadModelWeights(urlReader('/model'), () => {});
  const model = new QwenModel(QWEN25_3B, weights);
  console.log('LORA model loaded once (' + (tf.memory().numBytes/1e9).toFixed(1) + 'GB on GPU)');
  const topArg = async l => { const d = await l.data(); let m = 0; for (let i = 1; i < d.length; i++) if (d[i] > d[m]) m = i; return m; };
  async function gen(n) {
    const idsT = tf.tensor2d([ids], [1, ids.length], 'int32');
    const emb = model.embed(idsT); let pf = model.forward(emb, 0, null); emb.dispose(); idsT.dispose();
    let kv = pf.kvCaches, pos = ids.length, nxt = await topArg(pf.logits); pf.logits.dispose(); const got = [];
    for (let s = 0; s < n; s++) { got.push(nxt); const tt = tf.tensor2d([[nxt]], [1, 1], 'int32'); const ee = model.embed(tt); const r = model.forward(ee, pos, kv); ee.dispose(); tt.dispose(); kv = r.kvCaches; pos++; nxt = await topArg(r.logits); r.logits.dispose(); }
    model.disposeKV(kv); return got;
  }
  model.clearLora(); const base = await gen(12); console.log('LORA base(none) ' + JSON.stringify(base));
  model.setLora(syntheticAdapter('demoA', model.cfg, 8, 3.0)); const A = await gen(12); console.log('LORA demoA      ' + JSON.stringify(A));
  model.setLora(syntheticAdapter('demoB', model.cfg, 8, 3.0, ['self_attn.v_proj', 'mlp.up_proj'])); const B = await gen(12); console.log('LORA demoB      ' + JSON.stringify(B));
  model.clearLora(); const back = await gen(12); console.log('LORA cleared    ' + JSON.stringify(back));
  console.log('LORA DONE  base!=A:' + (JSON.stringify(base) !== JSON.stringify(A)) + '  A!=B:' + (JSON.stringify(A) !== JSON.stringify(B)) + '  base==cleared:' + (JSON.stringify(base) === JSON.stringify(back)));
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('LORA ERROR ' + e.message)));
