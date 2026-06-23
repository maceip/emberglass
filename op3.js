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
const T = 18, hd = 128;
window.run = async () => {
  await cmp('bcast_mul_heads [1,16,T,128]*[1,1,T,128]', () => tf.mul(tf.tensor(D(16*T*hd),[1,16,T,hd]), tf.tensor(D(T*hd),[1,1,T,hd])));
  await cmp('full_applyRope', () => {
    const x = tf.tensor(D(16*T*hd),[1,16,T,hd]);
    const cos = tf.reshape(tf.tensor(D(T*hd),[T,hd]),[1,1,T,hd]);
    const sin = tf.reshape(tf.tensor(D(T*hd).map(v=>v*0.3),[T,hd]),[1,1,T,hd]);
    const x1 = tf.slice(x,[0,0,0,0],[-1,-1,-1,hd/2]); const x2 = tf.slice(x,[0,0,0,hd/2],[-1,-1,-1,hd/2]);
    const rot = tf.concat([tf.neg(x2),x1],3);
    return tf.add(tf.mul(x,cos), tf.mul(rot,sin));
  });
  await cmp('slice_rope_table [8192,128]->[18,128]', () => tf.slice(tf.tensor(D(8192*hd),[8192,hd]),[0,0],[T,hd]));
  console.log('OP3 DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('OP ERROR '+e.message)));
