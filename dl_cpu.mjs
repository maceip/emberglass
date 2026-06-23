import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import fs from 'fs';
import { QwenModel, QWEN25_3B } from './qwen25.js';
import { loadModelWeights } from './weights.js';
await tf.setBackend('cpu'); await tf.ready();
const MD='model'; const fds={};
const reader={ async range(p,s,e){const l=e-s;const b=Buffer.allocUnsafe(l);fs.readSync(fds[p]??=fs.openSync(`${MD}/${p}`,'r'),b,0,l,s);return b.buffer.slice(b.byteOffset,b.byteOffset+l);}, async text(p){return fs.readFileSync(`${MD}/${p}`,'utf8');} };
const ref=JSON.parse(fs.readFileSync('ref.json')); const ids=ref.ids, T=ids.length;
const weights=await loadModelWeights(reader); const model=new QwenModel(QWEN25_3B,weights);
const emb=model.embed(tf.tensor2d([ids],[1,ids.length],'int32'));
const d=model.debugLayer0(emb,0);
for(const k of ['ln1','qproj','kproj','qr','kr','kRep','kT','scores','attnProj','h','ln2','mlpO','out']){const t=d[k];const C=t.shape[t.shape.length-1];const fl=await t.data();const off=(T-1)*C;console.log('SUB '+k+' '+JSON.stringify(Array.from(fl.slice(off,off+6)).map(x=>+x.toFixed(5))));}
