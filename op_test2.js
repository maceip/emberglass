// Op divergence at REAL Qwen2.5 shapes (the small test missed size-dependent bugs).
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-cpu';

const D = n => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 0.5);
async function cmp(name, build) {
  await tf.setBackend('cpu'); await tf.ready();
  const c = build(); const cd = await c.data(); c.dispose();
  await tf.setBackend('webgpu'); await tf.ready();
  const w = build(); const wd = await w.data(); w.dispose();
  let m = 0, im = 0; for (let i = 0; i < cd.length; i++) { const d = Math.abs(cd[i] - wd[i]); if (d > m) { m = d; im = i; } }
  console.log(`OP ${name} maxDiff=${m.toExponential(2)} @${im} (cpu=${cd[im]?.toFixed(4)} wgpu=${wd[im]?.toFixed(4)}) ${m > 1e-2 ? '<<< DIVERGES' : 'ok'}`);
}

window.run = async () => {
  const H = 2048, I = 11008, hd = 128, T = 18;
  await cmp('qproj 2048x2048 tB', () => tf.matMul(tf.tensor(D(T * H), [1, T, H]), tf.tensor(D(H * H), [H, H]), false, true));
  await cmp('gate 11008x2048 tB', () => tf.matMul(tf.tensor(D(T * H), [1, T, H]), tf.tensor(D(I * H), [I, H]), false, true));
  await cmp('down 2048x11008 tB', () => tf.matMul(tf.tensor(D(T * I), [1, T, I]), tf.tensor(D(H * I), [H, I]), false, true));
  await cmp('lmhead 151936 tB', () => tf.matMul(tf.tensor(D(H), [1, 1, H]), tf.tensor(D(151936 * H), [151936, H]), false, true));
  await cmp('bias_bcast', () => tf.add(tf.tensor(D(T * H), [1, T, H]), tf.tensor(D(H), [H])));
  await cmp('rmsnorm2048', () => { const x = tf.tensor(D(T * H), [1, T, H]); return tf.mul(x, tf.rsqrt(tf.add(tf.mean(tf.square(x), -1, true), 1e-6))); });
  await cmp('attn_qk 16x18x128 tB', () => tf.matMul(tf.tensor(D(16 * T * hd), [1, 16, T, hd]), tf.tensor(D(16 * T * hd), [1, 16, T, hd]), false, true));
  await cmp('attn_av', () => tf.matMul(tf.tensor(D(16 * T * T), [1, 16, T, T]), tf.tensor(D(16 * T * hd), [1, 16, T, hd]), false, false));
  await cmp('tileGQA 2->16', () => { const k = tf.tensor(D(2 * T * hd), [1, 2, T, hd]); return tf.reshape(tf.tile(tf.expandDims(k, 2), [1, 1, 8, 1, 1]), [1, 16, T, hd]); });
  await cmp('rope_slice128', () => { const x = tf.tensor(D(16 * T * hd), [1, 16, T, hd]); const a = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, 64]); const b = tf.slice(x, [0, 0, 0, 64], [-1, -1, -1, 64]); return tf.concat([tf.neg(b), a], 3); });
  console.log('OP2 DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('OP ERROR ' + e.message + ' ' + e.stack)));
