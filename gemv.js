import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgpu';
const D = n => Float32Array.from({ length: n }, (_, i) => Math.sin(i*0.1)*0.1);
window.run = async () => {
  await tf.setBackend('webgpu'); await tf.ready();
  const H=2048,I=11008;
  // weights like a layer (pre-transposed [in,out]): q/o 2048x2048, k/v 2048x256, gate/up 2048x11008, down 11008x2048
  const Wq=tf.tensor(D(H*H),[H,H]), Wkv=tf.tensor(D(H*256),[H,256]), Wg=tf.tensor(D(H*I),[H,I]), Wd=tf.tensor(D(I*H),[I,H]);
  const x=tf.tensor(D(H),[1,1,H]), xi=tf.tensor(D(I),[1,1,I]);
  const oneLayer = () => { // 7 matmuls = one layer's projections
    const a=tf.matMul(x,Wq,false,false); const b=tf.matMul(x,Wkv,false,false); const c=tf.matMul(x,Wkv,false,false);
    const d=tf.matMul(x,Wq,false,false); const e=tf.matMul(x,Wg,false,false); const f=tf.matMul(x,Wg,false,false);
    const g=tf.matMul(xi,Wd,false,false);
    return [a,b,c,d,e,f,g];
  };
  // warm
  for(let i=0;i<3;i++){ const t=oneLayer(); await t[6].data(); t.forEach(x=>x.dispose()); }
  // time 36 layers' worth of matmuls, pipelined, 1 readback
  await tf.nextFrame(); const t0=performance.now();
  let last;
  for(let L=0;L<36;L++){ const t=oneLayer(); last=t[6]; t.slice(0,6).forEach(x=>x.dispose()); if(L<35) last.dispose(); }
  await last.data(); // single sync
  const dt=performance.now()-t0;
  console.log(`GEMV 36 layers x7 matmuls = ${dt.toFixed(1)}ms  => matmul-only ceiling ${(1000/dt).toFixed(1)} tok/s`);
  console.log('GEMV DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('GEMV ERROR '+e.message)));
