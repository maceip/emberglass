// Browser/WebGPU bisect: run the same checkpoints as bisect.mjs but on the
// webgpu backend, log them so Playwright can compare against ref.json (CPU/HF).
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgpu';
import { QwenModel, QWEN25_3B } from './qwen25.js';
import { loadModelWeights, urlReader } from './weights.js';

const r5 = a => Array.from(a).map(x => +x.toFixed(5));
async function topArg(l) { const d = await l.data(); let m = 0; for (let i = 1; i < d.length; i++) if (d[i] > d[m]) m = i; return m; }

window.runDebug = async () => {
  await tf.setBackend('webgpu'); await tf.ready();
  const ref = await (await fetch('./ref.json')).json();
  const ids = ref.ids, T = ids.length, H = QWEN25_3B.hiddenSize;
  console.log('DBG loading weights…');
  const weights = await loadModelWeights(urlReader('/model'), () => {});
  const model = new QwenModel(QWEN25_3B, weights);
  console.log('DBG loaded; backend=' + tf.getBackend());
  for (const k of ['model.layers.0.input_layernorm.weight','model.layers.0.self_attn.q_proj.weight','model.layers.0.self_attn.q_proj.bias','model.layers.0.mlp.down_proj.weight']) { const d = await model.w[k].data(); console.log('DBG W ' + k + ' ' + JSON.stringify(r5(d.slice(0,8)))); console.log('DBG W ' + k + ' REF ' + JSON.stringify(ref['W_'+k])); }

  const idsT = tf.tensor2d([ids], [1, ids.length], 'int32');
  const emb = model.embed(idsT);
  const ed = await emb.data();
  console.log('DBG embed_last_8 WGPU ' + JSON.stringify(r5(ed.slice((T - 1) * H, (T - 1) * H + 8))));
  console.log('DBG embed_last_8 REF  ' + JSON.stringify(ref.embed_last_8));

  const l0 = model.layer(0, emb, 0, null);
  const ld = await l0.data();
  console.log('DBG layer0_last_8 WGPU ' + JSON.stringify(r5(ld.slice((T - 1) * H, (T - 1) * H + 8))));
  console.log('DBG layer0_last_8 REF  ' + JSON.stringify(ref.layer0_last_8));
  if (model._newKV) { model._newKV.k.dispose(); model._newKV.v.dispose(); }
  l0.dispose();

  const embP = model.embed(idsT);
  let pf = model.forward(embP, 0, null);
  embP.dispose();
  const am = await topArg(pf.logits);
  console.log('DBG argmax WGPU ' + am + ' REF ' + ref.argmax);
  let kv = pf.kvCaches, pos = ids.length, nxt = am; pf.logits.dispose();
  const got = [];
  for (let s = 0; s < 16; s++) {
    got.push(nxt);
    const tt = tf.tensor2d([[nxt]], [1, 1], 'int32');
    const ee = model.embed(tt);
    const r = model.forward(ee, pos, kv);
    ee.dispose(); tt.dispose(); kv = r.kvCaches; pos++;
    nxt = await topArg(r.logits); r.logits.dispose();
  }
  model.disposeKV(kv);
  console.log('DBG gen_ids WGPU ' + JSON.stringify(got));
  console.log('DBG gen_ids REF  ' + JSON.stringify(ref.gen_ids));
  console.log('DBG DONE match=' + (JSON.stringify(got) === JSON.stringify(ref.gen_ids)));
};
window.addEventListener('DOMContentLoaded', () => { window.runDebug().catch(e => console.log('DBG ERROR ' + e.message + ' ' + e.stack)); });
