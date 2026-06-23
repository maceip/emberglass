// Find which tf.js op diverges on webgpu vs cpu (same deterministic inputs).
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-cpu';

const D = n => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 0.5);

async function cmp(name, build) {
  await tf.setBackend('cpu'); await tf.ready();
  const c = build(); const cd = await c.data(); c.dispose();
  await tf.setBackend('webgpu'); await tf.ready();
  const w = build(); const wd = await w.data(); w.dispose();
  let m = 0; for (let i = 0; i < cd.length; i++) m = Math.max(m, Math.abs(cd[i] - wd[i]));
  console.log(`OP ${name} maxDiff=${m.toExponential(2)} ${m > 1e-3 ? '<<< DIVERGES' : 'ok'}`);
}

window.run = async () => {
  await cmp('matmul2d_transposeB', () => tf.matMul(tf.tensor(D(32), [4, 8]), tf.tensor(D(48), [6, 8]), false, true));
  await cmp('matmul3d_transposeB', () => tf.matMul(tf.tensor(D(64), [2, 4, 8]), tf.tensor(D(48), [6, 8]), false, true));
  await cmp('matmul3d_2d_false', () => tf.matMul(tf.tensor(D(64), [2, 4, 8]), tf.tensor(D(48), [8, 6]), false, false));
  await cmp('bmm4d_transposeB', () => tf.matMul(tf.tensor(D(64), [1, 2, 4, 8]), tf.tensor(D(96), [1, 2, 6, 8]), false, true));
  await cmp('bmm4d_explicitT', () => { const b = tf.tensor(D(96), [1, 2, 6, 8]); return tf.matMul(tf.tensor(D(64), [1, 2, 4, 8]), tf.transpose(b, [0, 1, 3, 2]), false, false); });
  await cmp('mean_axis-1', () => tf.mean(tf.square(tf.tensor(D(64), [2, 4, 8])), -1, true));
  await cmp('rsqrt', () => tf.rsqrt(tf.add(tf.tensor(D(8).map(Math.abs), [8]), 1e-6)));
  await cmp('softmax-1', () => tf.softmax(tf.tensor(D(30), [2, 3, 5]), -1));
  await cmp('gather', () => tf.gather(tf.tensor(D(40), [10, 4]), tf.tensor1d([1, 3, 5, 0], 'int32')));
  await cmp('tile_gqa', () => { const k = tf.tensor(D(64), [1, 2, 4, 8]); return tf.reshape(tf.tile(tf.expandDims(k, 2), [1, 1, 4, 1, 1]), [1, 8, 4, 8]); });
  await cmp('concat4d_ax2', () => tf.concat([tf.tensor(D(64), [1, 2, 4, 8]), tf.tensor(D(32), [1, 2, 2, 8])], 2));
  await cmp('sigmoid', () => tf.sigmoid(tf.tensor(D(16), [16])));
  await cmp('transpose4d', () => tf.transpose(tf.tensor(D(64), [1, 2, 4, 8]), [0, 2, 1, 3]));
  await cmp('slice4d', () => tf.slice(tf.tensor(D(64), [1, 2, 4, 8]), [0, 0, 0, 0], [1, 2, 4, 4]));
  console.log('OP DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('OP ERROR ' + e.message)));
