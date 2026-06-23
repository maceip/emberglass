// Bisect the forward-pass bug: load REAL weights (CPU), compare embedding +
// layer-0 output + final argmax against the HF reference (ref.json).
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import fs from 'fs';
import { QwenModel, QWEN25_3B } from './qwen25.js';
import { loadModelWeights } from './weights.js';

await tf.setBackend('cpu'); await tf.ready();
const MD = 'model';
function nodeReader() {
  const fds = {};
  const fd = p => (fds[p] ??= fs.openSync(`${MD}/${p}`, 'r'));
  return {
    async range(path, start, end) {
      const len = end - start;
      const b = Buffer.allocUnsafe(len);
      fs.readSync(fd(path), b, 0, len, start);
      return b.buffer.slice(b.byteOffset, b.byteOffset + len);
    },
    async text(path) { return fs.readFileSync(`${MD}/${path}`, 'utf8'); },
  };
}
const ref = JSON.parse(fs.readFileSync('ref.json'));
const ids = ref.ids;
const T = ids.length, H = QWEN25_3B.hiddenSize;
const r5 = a => Array.from(a).map(x => +x.toFixed(5));

console.log('loading real weights on CPU…');
const t0 = Date.now();
const weights = await loadModelWeights(nodeReader());
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
const model = new QwenModel(QWEN25_3B, weights);

const idsT = tf.tensor2d([ids], [1, ids.length], 'int32');
const emb = model.embed(idsT);
const embData = await emb.data();
console.log('JS  embed_last_8:', r5(embData.slice((T - 1) * H, (T - 1) * H + 8)));
console.log('REF embed_last_8:', ref.embed_last_8);

const l0 = model.layer(0, emb, 0, null);
const l0Data = await l0.data();
console.log('JS  layer0_last_8:', r5(l0Data.slice((T - 1) * H, (T - 1) * H + 8)));
console.log('REF layer0_last_8:', ref.layer0_last_8);
if (model._newKV) { model._newKV.k.dispose(); model._newKV.v.dispose(); }
l0.dispose();

// full forward (fresh embed; forward leaves the input embeds alone)
const emb2 = model.embed(idsT);
const { logits, kvCaches } = model.forward(emb2, 0, null);
const lg = await logits.data();
let am = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[am]) am = i;
// top5
const idx = Array.from(lg.keys()).sort((a, b) => lg[b] - lg[a]).slice(0, 5);
console.log('JS  argmax:', am, ' top5:', idx, ' vals:', idx.map(i => +lg[i].toFixed(3)));
console.log('REF argmax:', ref.argmax, ' top5:', ref.top5_ids, ' vals:', ref.top5_vals);
model.disposeKV(kvCaches);

// ---- decode-loop comparison ----
console.log('\nREF gen_ids:', ref.gen_ids);
async function topArg(l) { const d = await l.data(); let m = 0; for (let i = 1; i < d.length; i++) if (d[i] > d[m]) m = i; return m; }
const embP = model.embed(idsT);
let pf = model.forward(embP, 0, null);
embP.dispose();
let kv = pf.kvCaches, pos = ids.length;
let nxt = await topArg(pf.logits); pf.logits.dispose();
const got = [];
for (let s = 0; s < 16; s++) {
  got.push(nxt);
  const tt = tf.tensor2d([[nxt]], [1, 1], 'int32');
  const ee = model.embed(tt);
  const r = model.forward(ee, pos, kv);
  ee.dispose(); tt.dispose();
  kv = r.kvCaches; pos++;
  nxt = await topArg(r.logits); r.logits.dispose();
}
model.disposeKV(kv);
console.log('JS  gen_ids:', got);
console.log('MATCH:', JSON.stringify(got) === JSON.stringify(ref.gen_ids) ? 'EXACT' : 'DIFFERS at index ' + got.findIndex((v, i) => v !== ref.gen_ids[i]));
