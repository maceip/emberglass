import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgpu';
import { QwenModel, QWEN25_3B } from './qwen25.js';
import { loadModelWeights, urlReader } from './weights.js';
window.run = async () => {
  await tf.setBackend('webgpu'); await tf.ready();
  const ref = await (await fetch('./ref.json')).json(); const ids = ref.ids, T = ids.length;
  const weights = await loadModelWeights(urlReader('/model'), () => {});
  const model = new QwenModel(QWEN25_3B, weights);
  const emb = model.embed(tf.tensor2d([ids], [1, ids.length], 'int32'));
  const d = model.debugLayer0(emb, 0);
  for (const k of ['ln1','qproj','kproj','qr','kr','kRep','kT','scores','attnProj','h','ln2','mlpO','out']) {
    const t = d[k]; const C = t.shape[t.shape.length-1]; const fl = await t.data(); const off = (T-1)*C;
    console.log('SUB ' + k + ' ' + JSON.stringify(Array.from(fl.slice(off, off+6)).map(x => +x.toFixed(5))));
  }
  console.log('SUB DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('SUB ERROR ' + e.message)));
