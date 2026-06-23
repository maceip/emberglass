import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-cpu';
const D = n => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 0.5);
async function cmp(name, build) {
  await tf.setBackend('cpu'); await tf.ready(); const c = build(); const cd = await c.data(); c.dispose();
  await tf.setBackend('webgpu'); await tf.ready(); const w = build(); const wd = await w.data(); w.dispose();
  let m = 0, im = 0; for (let i = 0; i < cd.length; i++) { const d = Math.abs(cd[i] - wd[i]); if (d > m) { m = d; im = i; } }
  console.log(`OP ${name} maxDiff=${m.toExponential(2)} (cpu=${cd[im]?.toFixed(4)} wgpu=${wd[im]?.toFixed(4)}) ${m > 1e-2 ? '<<< DIVERGES' : 'ok'}`);
}
const H = 2048, T = 18;
window.run = async () => {
  await cmp('k_proj 256x2048 tB', () => tf.matMul(tf.tensor(D(T*H), [1, T, H]), tf.tensor(D(256*H), [256, H]), false, true));
  await cmp('k_proj+bias', () => tf.add(tf.matMul(tf.tensor(D(T*H), [1, T, H]), tf.tensor(D(256*H), [256, H]), false, true), tf.tensor(D(256), [256])));
  await cmp('v_proj 256x2048 tB', () => tf.matMul(tf.tensor(D(T*H), [1, T, H]), tf.tensor(D(256*H), [256, H]), false, true));
  await cmp('512x2048 tB', () => tf.matMul(tf.tensor(D(T*H), [1, T, H]), tf.tensor(D(512*H), [512, H]), false, true));
  await cmp('128x2048 tB', () => tf.matMul(tf.tensor(D(T*H), [1, T, H]), tf.tensor(D(128*H), [128, H]), false, true));
  console.log('OP DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('OP ERROR ' + e.message)));
