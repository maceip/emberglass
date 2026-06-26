/*
 *   ,;
 *  \@@#\:          :/.        .:;;:
 * _@@@@@@#+\|/!;;!-@@@--;    ,@@@@@;
 * .!_*@@@@@@@@@@@@@@@@@@@;   |@@@@@\
 *     .:!|+@@@@@##@@@@@@@#!  -@@@@@#,
 *         .\@@@*;,\@@@@@@@@+,*@@@@@@+.
 *     :*#@@@@@@@@@@@@@@-+@@@@@@@\@@@@-.
 *     .#@@@@@#@@@@#*@@@+ /@@@@@@;\@@@@+.
 *      ;\/:,  -@@@@;|@@@\ ,+@@@@!.+@@@@*:
 *             ,@@@@#*@@@@@#+__!.  ,*@@@@@/
 *              \##+_@@@@@@@@,      ,+@@@_:
 *                   ;;,,..,:         !;.
 */
var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// src/config.js
var QWEN25_3B = {
  hiddenSize: 2048,
  numLayers: 36,
  numHeads: 16,
  numKVHeads: 2,
  headDim: 128,
  intermediateSize: 11008,
  vocabSize: 151936,
  rmsNormEps: 1e-6,
  ropeTheta: 1e6,
  /*
   * TECHNIQUE: Tie word embeddings
   *   input embedding == output head.
   *   Simplifies loading (one tensor), schema, and final projection math.
   *   Required by the current model_uploader + schema.
   */
  tieWordEmbeddings: true,
  // QKV projections carry a bias in Qwen2.5; o_proj and the MLP do not.
  attentionBias: true
};

// src/readers.js
function urlReader(baseUrl, headers = {}) {
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  return {
    async range(path, start, end) {
      const r = await fetch(base + path, {
        headers: { ...headers, Range: `bytes=${start}-${end - 1}` }
      });
      if (!r.ok && r.status !== 206) {
        throw new Error(`range ${path} ${start}-${end}: ${r.status}`);
      }
      return await r.arrayBuffer();
    },
    async text(path) {
      const r = await fetch(base + path, { headers });
      if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
      return await r.text();
    }
  };
}
__name(urlReader, "urlReader");
function hfReader(repo, token = "", rev = "main") {
  return urlReader(
    `https://huggingface.co/${repo}/resolve/${rev}`,
    token ? { Authorization: `Bearer ${token}` } : {}
  );
}
__name(hfReader, "hfReader");
function fileReader(fileMap) {
  const pick = /* @__PURE__ */ __name((path) => fileMap[path] || fileMap[path.split("/").pop()], "pick");
  return {
    async range(path, start, end) {
      const f = pick(path);
      if (!f) throw new Error(`file not provided: ${path}`);
      return await f.slice(start, end).arrayBuffer();
    },
    async text(path) {
      const f = pick(path);
      if (!f) throw new Error(`file not provided: ${path}`);
      return await f.text();
    }
  };
}
__name(fileReader, "fileReader");

// src/services/adapter_registry.js
var AdapterRegistry = class {
  static {
    __name(this, "AdapterRegistry");
  }
  constructor() {
    this.adapters = { none: null };
  }
  add(name, modules) {
    this.adapters[name] = { modules };
    return this.adapters[name];
  }
  get(name) {
    return this.adapters[name] || null;
  }
  /*
   * TECHNIQUE: Runtime adapter swapping via setLora
   *   Registry holds pre-uploaded A/B buffers. applyToRuntime calls
   *   rt.setLora which just swaps references — no weight reload.
   */
  applyToRuntime(name, rt) {
    const adapter = this.get(name);
    if (adapter) rt.setLora(adapter);
    else rt.clearLora();
    return adapter;
  }
};

// src/qwgpu/kernels.js
var GEMV = `
enable subgroups;
requires immediate_address_space;
requires subgroup_id;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;       // [N][K/4] int8
@group(0) @binding(2) var<storage,read> scale: array<f32>;   // [N]
@group(0) @binding(3) var<storage,read> bias: array<f32>;    // [N] or dummy
@group(0) @binding(4) var<storage,read> loraD: array<f32>;   // [rank] precomputed x@A (or dummy)
@group(0) @binding(5) var<storage,read> loraB: array<f32>;   // [rank][N] (or dummy)
@group(0) @binding(6) var<storage,read_write> y: array<f32>; // [N]
var<immediate> m: Meta;
var<workgroup> part: array<f32,64>;       // one slot per subgroup
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32,
        @builtin(subgroup_id) sgroup: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }               // workgroup-uniform: whole group exits together
  let K4 = m.K/4u; let rb = n*K4;
  var acc = 0.0;
  for (var k = tid; k < K4; k = k + 64u) {
    let p = w[rb+k];
    let v = unpack4xI8(p);                 // vec4<i32>
    let kk = k*4u;
    acc = acc + x[kk]*f32(v.x) + x[kk+1u]*f32(v.y) + x[kk+2u]*f32(v.z) + x[kk+3u]*f32(v.w);
  }
  let ssum = subgroupAdd(acc);            // reduce within subgroup (no barrier)
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var red = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { red = red + part[i]; }
    var o = red * scale[n];
    if (m.hasBias == 1u) { o = o + bias[n]; }
    if (m.hasLora == 1u) { var dl = 0.0; for (var r = 0u; r < m.rank; r = r + 1u) { dl = dl + loraD[r] * loraB[r*m.N + n]; } o = o + m.scaleLo * dl; }
    y[n] = o;
  }
}`;
var LORA_A = `
enable subgroups;
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> x: array<f32>;     // [K]
@group(0) @binding(1) var<storage,read> A: array<f32>;     // [rank][K] (transposed)
@group(0) @binding(2) var<storage,read_write> d: array<f32>; // [rank]
var<immediate> m: vec2<u32>;           // K, rank
var<workgroup> part: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let r = wid.x; let K = m.x; if (r >= m.y) { return; }
  let rb = r*K; var acc = 0.0;
  for (var k = lid.x; k < K; k = k + 64u) { acc = acc + x[k]*A[rb + k]; }
  let s = subgroupAdd(acc);
  if (sgid == 0u) { part[lid.x / sgsz] = s; }
  workgroupBarrier();
  if (lid.x == 0u) { let nsg=(64u+sgsz-1u)/sgsz; var o=0.0; for(var i=0u;i<nsg;i=i+1u){o=o+part[i];} d[r]=o; }
}`;
var LORA_A_BATCH = `
enable subgroups;
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> x: array<f32>;       // [T][K]
@group(0) @binding(1) var<storage,read> A: array<f32>;       // [rank][K]
@group(0) @binding(2) var<storage,read_write> d: array<f32>; // [T][rank]
var<immediate> m: vec4<u32>;             // K, rank, T, _
var<workgroup> part: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let r = wid.x; let t = wid.y; let K = m.x; let rank = m.y; if (r >= rank || t >= m.z) { return; }
  let xb = t*K; let ab = r*K; var acc = 0.0;
  for (var k = lid.x; k < K; k = k + 64u) { acc = acc + x[xb + k]*A[ab + k]; }
  let s = subgroupAdd(acc);
  if (sgid == 0u) { part[lid.x / sgsz] = s; }
  workgroupBarrier();
  if (lid.x == 0u) { let nsg=(64u+sgsz-1u)/sgsz; var o=0.0; for(var i=0u;i<nsg;i=i+1u){o=o+part[i];} d[t*rank + r]=o; }
}`;
var LORA_B_ADD_T = `
requires immediate_address_space;
struct Meta { T:u32, N:u32, rank:u32, gx:u32, scale:f32, p1:f32, p2:f32, p3:f32 };
@group(0) @binding(0) var<storage,read> d: array<f32>;        // [T][rank]
@group(0) @binding(1) var<storage,read> B: array<f32>;        // [rank][N]
@group(0) @binding(2) var<storage,read_write> Y: array<f32>;  // [T][N]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * (m.gx * 256u) + gid.x;
  if (i >= m.T * m.N) { return; }
  let t = i / m.N; let n = i % m.N; var acc = 0.0;
  for (var r = 0u; r < m.rank; r = r + 1u) { acc = acc + d[t*m.rank + r] * B[r*m.N + n]; }
  Y[i] = Y[i] + m.scale * acc;
}`;
var LORA_B_ADD = `
requires immediate_address_space;
struct Meta { N:u32, rank:u32, p0:u32, p1:u32, scale:f32, f0:f32, f1:f32, f2:f32 };
@group(0) @binding(0) var<storage,read> d: array<f32>;       // [rank]
@group(0) @binding(1) var<storage,read> B: array<f32>;       // [rank][N]
@group(0) @binding(2) var<storage,read_write> y: array<f32>; // [N]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= m.N) { return; }
  var acc = 0.0;
  for (var r = 0u; r < m.rank; r = r + 1u) { acc = acc + d[r] * B[r*m.N + n]; }
  y[n] = y[n] + m.scale * acc;
}`;
var RMSNORM = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> g: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
var<immediate> m: vec2<f32>;   // K, eps
var<workgroup> part: array<f32,256>;
@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x);
  var s = 0.0; for (var k = tid; k < K; k = k + WG) { let v = x[k]; s = s + v*v; }
  part[tid] = s; workgroupBarrier();
  for (var t = WG / 2u; t > 0u; t = t/2u) { if (tid < t) { part[tid] = part[tid] + part[tid+t]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0]/m.x + m.y);
  for (var k = tid; k < K; k = k + WG) { y[k] = x[k]*inv*g[k]; }
}`;
var RMSNORM_F16 = `
requires immediate_address_space;
enable f16;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> g: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
var<immediate> m: vec2<f32>;   // K, eps
// Reduction accumulates in f32 even though the normalize is f16: summing v*v over
// thousands of dims overflows f16 (>65504) at high-magnitude tokens (the attention
// sink), which collapses inv to 0. Keeping the sum in f32 is the overflow-safe path.
var<workgroup> part: array<f32,256>;
@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x);
  var s = 0.0;
  for (var k = tid; k < K; k = k + WG) { let v = f32(x[k]); s = s + v*v; }
  part[tid] = s; workgroupBarrier();
  for (var t = WG / 2u; t > 0u; t = t/2u) { if (tid < t) { part[tid] = part[tid] + part[tid+t]; } workgroupBarrier(); }
  let inv = f16(inverseSqrt(part[0]/m.x + m.y));
  for (var k = tid; k < K; k = k + WG) { y[k] = f32( f16(x[k]) * inv * f16(g[k]) ); }
}`;
var ROPE = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
var<immediate> m: vec3<u32>;             // nHeads, headDim, pos
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let pos = m.z; let half = D/2u;
  if (g >= H*half) { return; }
  let h = g / half; let j = g % half;
  let lo = h*D + j; let hi = lo + half; let off = pos*D + j;
  let c = cosT[off]; let s = sinT[off];
  let xl = x[lo]; let xh = x[hi];
  // EXACT rotate-half: separately-rounded products (fma(a,b,0)) prevent the
  // compiler from contracting x*c - x*s into a single fma, matching the PyTorch
  // reference rounding exactly.
  x[lo] = fma(xl, c, 0.0) + fma(-xh, s, 0.0);
  x[hi] = fma(xh, c, 0.0) + fma(xl, s, 0.0);
}`;
var ROPE_F16 = `
requires immediate_address_space;
enable f16;
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
var<immediate> m: vec3<u32>;             // nHeads, headDim, pos
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let pos = m.z; let half = D/2u;
  if (g >= H*half) { return; }
  let h = g / half; let j = g % half;
  let lo = h*D + j; let hi = lo + half; let off = pos*D + j;
  let c = f16(cosT[off]); let s = f16(sinT[off]);
  let xl = f16(x[lo]); let xh = f16(x[hi]);
  x[lo] = f32( fma(xl, c, 0.0h) + fma(-xh, s, 0.0h) );
  x[hi] = f32( fma(xh, c, 0.0h) + fma(xl, s, 0.0h) );
}`;
var ROPE_QK = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read_write> q: array<f32>;
@group(0) @binding(1) var<storage,read_write> k: array<f32>;
@group(0) @binding(2) var<storage,read> cosT: array<f32>;
@group(0) @binding(3) var<storage,read> sinT: array<f32>;
var<immediate> m: vec4<u32>;             // qHeads, kvHeads, headDim, pos
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let qH = m.x; let kH = m.y; let D = m.z; let pos = m.w; let half = D/2u;
  let qPairs = qH * half; let kPairs = kH * half; let total = qPairs + kPairs;
  if (g >= total) { return; }
  let isK = g >= qPairs;
  var r = g;
  if (isK) { r = g - qPairs; }
  let h = r / half; let j = r % half;
  let lo = h*D + j; let hi = lo + half; let off = pos*D + j;
  let c = cosT[off]; let s = sinT[off];
  if (isK) {
    let xl = k[lo]; let xh = k[hi];
    k[lo] = fma(xl, c, 0.0) + fma(-xh, s, 0.0); k[hi] = fma(xh, c, 0.0) + fma(xl, s, 0.0);
  } else {
    let xl = q[lo]; let xh = q[hi];
    q[lo] = fma(xl, c, 0.0) + fma(-xh, s, 0.0); q[hi] = fma(xh, c, 0.0) + fma(xl, s, 0.0);
  }
}`;
var ROPE_QK_F16 = `
requires immediate_address_space;
enable f16;
@group(0) @binding(0) var<storage,read_write> q: array<f32>;
@group(0) @binding(1) var<storage,read_write> k: array<f32>;
@group(0) @binding(2) var<storage,read> cosT: array<f32>;
@group(0) @binding(3) var<storage,read> sinT: array<f32>;
var<immediate> m: vec4<u32>;             // qHeads, kvHeads, headDim, pos
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let qH = m.x; let kH = m.y; let D = m.z; let pos = m.w; let half = D/2u;
  let qPairs = qH * half; let kPairs = kH * half; let total = qPairs + kPairs;
  if (g >= total) { return; }
  let isK = g >= qPairs;
  var r = g;
  if (isK) { r = g - qPairs; }
  let h = r / half; let j = r % half;
  let lo = h*D + j; let hi = lo + half; let off = pos*D + j;
  let c = f16(cosT[off]); let s = f16(sinT[off]);
  if (isK) {
    let xl = f16(k[lo]); let xh = f16(k[hi]);
    k[lo] = f32( fma(xl, c, 0.0h) + fma(-xh, s, 0.0h) ); k[hi] = f32( fma(xh, c, 0.0h) + fma(xl, s, 0.0h) );
  } else {
    let xl = f16(q[lo]); let xh = f16(q[hi]);
    q[lo] = f32( fma(xl, c, 0.0h) + fma(-xh, s, 0.0h) ); q[hi] = f32( fma(xh, c, 0.0h) + fma(xl, s, 0.0h) );
  }
}`;
var ATTN_PARTIAL = `
requires immediate_address_space;
enable subgroups;
override WG: u32 = 128u;
struct AttnP { nHeads: u32, nKV: u32, ctx: u32, hd: u32, nsplit: u32, chunk: u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> pm: array<f32>;  // [nHeads*nsplit] per-split max
@group(0) @binding(4) var<storage,read_write> pz: array<f32>;  // [nHeads*nsplit] per-split sum
@group(0) @binding(5) var<storage,read_write> po: array<f32>;  // [nHeads*nsplit*hd] unnorm weighted V
var<immediate> m: AttnP;
var<workgroup> sc: array<f32,128>;
var<workgroup> red: array<f32,32>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let s = wid.y; let tid = lid.x;
  let nHeads = m.nHeads; let nKV = m.nKV; let ctx = m.ctx; let hd = m.hd; let nsplit = m.nsplit; let chunk = m.chunk;
  let kvh = h / (nHeads / nKV);
  let qbase = h*hd; let stride = nKV*hd; let hoff = kvh*hd; let scale = 1.0/sqrt(f32(hd));
  let nsg = (128u + sgsz - 1u) / sgsz;
  let t0 = s*chunk; var t1 = t0 + chunk; if (t1 > ctx) { t1 = ctx; }
  let t = t0 + tid; var sv = -1e30;
  if (t < t1) { var dot = 0.0; let kb = t*stride + hoff; for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qbase+d]*kc[kb+d]; } sv = dot*scale; }
  let sgm = subgroupMax(sv); if (sgid == 0u) { red[tid/sgsz] = sgm; }
  workgroupBarrier();
  var M = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { M = max(M, red[i]); }
  workgroupBarrier();
  var ev = 0.0; if (t < t1) { ev = exp(sv - M); } sc[tid] = ev;
  let sgs = subgroupAdd(ev); if (sgid == 0u) { red[tid/sgsz] = sgs; }
  workgroupBarrier();
  var Z = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { Z = Z + red[i]; }
  workgroupBarrier();
  let len = t1 - t0; let pbase = (h*nsplit + s)*hd;
  for (var d = tid; d < hd; d = d + 128u) {
    var acc = 0.0; for (var tt = 0u; tt < len; tt = tt + 1u) { acc = acc + sc[tt]*vc[(t0+tt)*stride + hoff + d]; }
    po[pbase + d] = acc;
  }
  if (tid == 0u) { pm[h*nsplit + s] = M; pz[h*nsplit + s] = Z; }
}`;
var ATTN_PARTIAL_F16 = `
requires immediate_address_space;
enable subgroups;
enable f16;
override WG: u32 = 128u;
struct AttnP { nHeads: u32, nKV: u32, ctx: u32, hd: u32, nsplit: u32, chunk: u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> pm: array<f32>;  // [nHeads*nsplit] per-split max
@group(0) @binding(4) var<storage,read_write> pz: array<f32>;  // [nHeads*nsplit] per-split sum
@group(0) @binding(5) var<storage,read_write> po: array<f32>;  // [nHeads*nsplit*hd] unnorm weighted V
var<immediate> m: AttnP;
// f16 "staging" mode: Q/K/V values are read through f16 (so they carry f16 rounding,
// modelling an f16 KV cache), but every REDUCTION \u2014 the QK dot, the softmax max/sum,
// and the weighted-V accumulation \u2014 runs in f32. Accumulating scores in f16 overflows
// at long context / high-magnitude tokens; f32 accumulation is the overflow-safe path
// (matches the Gemma-4 "scores/PV accumulate in f32, only K/V carry f16 rounding").
var<workgroup> sc: array<f32,128>;
var<workgroup> red: array<f32,32>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let s = wid.y; let tid = lid.x;
  let nHeads = m.nHeads; let nKV = m.nKV; let ctx = m.ctx; let hd = m.hd; let nsplit = m.nsplit; let chunk = m.chunk;
  let kvh = h / (nHeads / nKV);
  let qbase = h*hd; let stride = nKV*hd; let hoff = kvh*hd; let scale = 1.0 / sqrt(f32(hd));
  let nsg = (WG + sgsz - 1u) / sgsz;
  let t0 = s*chunk; var t1 = t0 + chunk; if (t1 > ctx) { t1 = ctx; }
  let t = t0 + tid; var sv = -1e30;
  if (t < t1) { var dot = 0.0; let kb = t*stride + hoff; for (var d = 0u; d < hd; d = d + 1u) { dot = dot + f32(f16(q[qbase+d])) * f32(f16(kc[kb+d])); } sv = dot*scale; }
  let sgm = subgroupMax(sv); if (sgid == 0u) { red[tid/sgsz] = sgm; }
  workgroupBarrier();
  var M = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { M = max(M, red[i]); }
  workgroupBarrier();
  var ev = 0.0; if (t < t1) { ev = exp(sv - M); } sc[tid] = ev;
  let sgs = subgroupAdd(ev); if (sgid == 0u) { red[tid/sgsz] = sgs; }
  workgroupBarrier();
  var Z = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { Z = Z + red[i]; }
  workgroupBarrier();
  let len = t1 - t0; let pbase = (h*nsplit + s)*hd;
  for (var d = tid; d < hd; d = d + WG) {
    var acc = 0.0; for (var tt = 0u; tt < len; tt = tt + 1u) { acc = acc + sc[tt] * f32(f16(vc[(t0+tt)*stride + hoff + d])); }
    po[pbase + d] = acc;
  }
  if (tid == 0u) { pm[h*nsplit + s] = M; pz[h*nsplit + s] = Z; }
}`;
var ATTN_COMBINE = `
requires immediate_address_space;
override WG: u32 = 128u;
@group(0) @binding(0) var<storage,read> pm: array<f32>;
@group(0) @binding(1) var<storage,read> pz: array<f32>;
@group(0) @binding(2) var<storage,read> po: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
var<immediate> m: vec4<u32>;   // nHeads, hd, nsplit, _
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x; let tid = lid.x; let hd = m.y; let nsplit = m.z; let base = h*nsplit;
  var M = -1e30; for (var s = 0u; s < nsplit; s = s + 1u) { M = max(M, pm[base+s]); }
  var Z = 0.0; for (var s = 0u; s < nsplit; s = s + 1u) { Z = Z + pz[base+s]*exp(pm[base+s]-M); }
  let invZ = 1.0 / Z;
  for (var d = tid; d < hd; d = d + WG) {
    var acc = 0.0;
    for (var s = 0u; s < nsplit; s = s + 1u) { acc = acc + exp(pm[base+s]-M)*po[(base+s)*hd + d]; }
    o[h*hd + d] = acc * invZ;
  }
}`;
var ATTN_COMBINE_F16 = `
requires immediate_address_space;
enable f16;
override WG: u32 = 128u;
@group(0) @binding(0) var<storage,read> pm: array<f32>;
@group(0) @binding(1) var<storage,read> pz: array<f32>;
@group(0) @binding(2) var<storage,read> po: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
var<immediate> m: vec4<u32>;   // nHeads, hd, nsplit, _
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x; let tid = lid.x; let hd = m.y; let nsplit = m.z; let base = h*nsplit;
  // Cross-split softmax merge accumulates max/sum in f32 (overflow-safe); only the
  // final per-element weighting carries f16 rounding.
  var M = -1e30; for (var s = 0u; s < nsplit; s = s + 1u) { M = max(M, pm[base+s]); }
  var Z = 0.0; for (var s = 0u; s < nsplit; s = s + 1u) { Z = Z + pz[base+s] * exp(pm[base+s] - M); }
  let invZ = 1.0 / Z;
  for (var d = tid; d < hd; d = d + WG) {
    var acc = 0.0;
    for (var s = 0u; s < nsplit; s = s + 1u) { acc = acc + exp(pm[base+s] - M) * f32(f16(po[(base+s)*hd + d])); }
    o[h*hd + d] = acc * invZ;
  }
}`;
var GEMM4 = `
requires immediate_address_space;
struct Meta { K:u32, N:u32, T:u32, gpr:u32, hasBias:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> A: array<f32>;       // [T][K]
@group(0) @binding(1) var<storage,read> W: array<u32>;       // [N][K/8] int4
@group(0) @binding(2) var<storage,read> scale: array<f32>;   // [N][gpr]
@group(0) @binding(3) var<storage,read> bias: array<f32>;    // [N] or dummy
@group(0) @binding(4) var<storage,read_write> Y: array<f32>; // [T][N]
var<immediate> m: Meta;
const BM = 16u; const BN = 64u;
var<workgroup> As: array<f32, 128>;   // BM*8 \u2014 A staged for one 8-wide K chunk
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tTile = wid.y * BM; let col = wid.x * BN + lid.x; let valid = col < m.N;
  let K8 = m.K/8u; let rb = col*K8;
  var acc: array<f32, 16>;
  for (var i = 0u; i < BM; i = i + 1u) { acc[i] = 0.0; }
  for (var c = 0u; c < K8; c = c + 1u) {
    for (var l = lid.x; l < BM*8u; l = l + 64u) {
      let tt = l / 8u; let trow = tTile + tt;
      As[l] = select(0.0, A[trow*m.K + c*8u + (l % 8u)], trow < m.T);
    }
    workgroupBarrier();
    if (valid) {
      let word = W[rb + c]; let sc = scale[col*m.gpr + ((c*8u) >> 7u)];
      let w0=f32(i32(word<<28u)>>28u)*sc; let w1=f32(i32(word<<24u)>>28u)*sc;
      let w2=f32(i32(word<<20u)>>28u)*sc; let w3=f32(i32(word<<16u)>>28u)*sc;
      let w4=f32(i32(word<<12u)>>28u)*sc; let w5=f32(i32(word<<8u)>>28u)*sc;
      let w6=f32(i32(word<<4u)>>28u)*sc;  let w7=f32(i32(word)>>28u)*sc;
      for (var t = 0u; t < BM; t = t + 1u) {
        let b = t*8u;
        acc[t] = acc[t] + As[b]*w0+As[b+1u]*w1+As[b+2u]*w2+As[b+3u]*w3+As[b+4u]*w4+As[b+5u]*w5+As[b+6u]*w6+As[b+7u]*w7;
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    let bv = select(0.0, bias[col], m.hasBias == 1u);
    for (var t = 0u; t < BM; t = t + 1u) { let trow = tTile + t; if (trow < m.T) { Y[trow*m.N + col] = acc[t] + bv; } }
  }
}`;
var GEMM4_ADD_T = `
requires immediate_address_space;
struct Meta { K:u32, N:u32, T:u32, gpr:u32, hasBias:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> A: array<f32>;
@group(0) @binding(1) var<storage,read> W: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read_write> Y: array<f32>;
var<immediate> m: Meta;
const BM = 16u; const BN = 64u;
var<workgroup> As: array<f32, 128>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tTile = wid.y * BM; let col = wid.x * BN + lid.x; let valid = col < m.N;
  let K8 = m.K/8u; let rb = col*K8;
  var acc: array<f32, 16>;
  for (var i = 0u; i < BM; i = i + 1u) { acc[i] = 0.0; }
  for (var c = 0u; c < K8; c = c + 1u) {
    for (var l = lid.x; l < BM*8u; l = l + 64u) {
      let tt = l / 8u; let trow = tTile + tt;
      As[l] = select(0.0, A[trow*m.K + c*8u + (l % 8u)], trow < m.T);
    }
    workgroupBarrier();
    if (valid) {
      let word = W[rb + c]; let sc = scale[col*m.gpr + ((c*8u) >> 7u)];
      let w0=f32(i32(word<<28u)>>28u)*sc; let w1=f32(i32(word<<24u)>>28u)*sc;
      let w2=f32(i32(word<<20u)>>28u)*sc; let w3=f32(i32(word<<16u)>>28u)*sc;
      let w4=f32(i32(word<<12u)>>28u)*sc; let w5=f32(i32(word<<8u)>>28u)*sc;
      let w6=f32(i32(word<<4u)>>28u)*sc;  let w7=f32(i32(word)>>28u)*sc;
      for (var t = 0u; t < BM; t = t + 1u) {
        let b = t*8u;
        acc[t] = acc[t] + As[b]*w0+As[b+1u]*w1+As[b+2u]*w2+As[b+3u]*w3+As[b+4u]*w4+As[b+5u]*w5+As[b+6u]*w6+As[b+7u]*w7;
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    let bv = select(0.0, bias[col], m.hasBias == 1u);
    for (var t = 0u; t < BM; t = t + 1u) {
      let trow = tTile + t;
      if (trow < m.T) { Y[trow*m.N + col] = Y[trow*m.N + col] + acc[t] + bv; }
    }
  }
}`;
var ADD = `
requires immediate_address_space;
requires linear_indexing;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read_write> y: array<f32>;
var<immediate> n: u32;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_index) gid: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * WG;
  for (var i = gid; i < n; i = i + stride) { y[i] = y[i] + a[i]; }
}`;
var ADD_F16 = `
requires immediate_address_space;
requires linear_indexing;
enable f16;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read_write> y: array<f32>;
var<immediate> n: u32;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_index) gid: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * WG;
  for (var i = gid; i < n; i = i + stride) { y[i] = f32(f16(y[i]) + f16(a[i])); }
}`;
var SILUMUL_F16 = `
requires immediate_address_space;
enable f16;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read_write> gate: array<f32>;
@group(0) @binding(1) var<storage,read> up: array<f32>;
var<immediate> n: u32;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * WG;
  // Activation (silu) in f32 to avoid the f16 exp(-v) -> Inf intermediate for very
  // negative v; only the bandwidth-bound elementwise multiply carries f16 rounding.
  for (var i = g.x; i < n; i = i + stride) { let v = gate[i]; let sg = v / (1.0 + exp(-v)); gate[i] = f32( f16(sg) * f16(up[i]) ); }
}`;
var SILUMUL = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read_write> gate: array<f32>;
@group(0) @binding(1) var<storage,read> up: array<f32>;
var<immediate> n: u32;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * WG;
  for (var i = g.x; i < n; i = i + stride) { let v = gate[i]; gate[i] = (v/(1.0+exp(-v)))*up[i]; }
}`;
var EMBED = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> w: array<u32>;
@group(0) @binding(1) var<storage,read> scale: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
var<immediate> m: vec2<u32>;   // id, hidden
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let k = g.x; let id = m.x; let H = m.y; if (k >= H) { return; }
  let v = unpack4xI8(w[id*(H/4u) + (k>>2u)]); let lane = k & 3u;
  var b: i32; if (lane==0u){b=v.x;} else if (lane==1u){b=v.y;} else if (lane==2u){b=v.z;} else {b=v.w;}
  out[k] = f32(b) * scale[id];
}`;
var EMBED_BUF = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> w: array<u32>;
@group(0) @binding(1) var<storage,read> scale: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<storage,read> idbuf: array<u32>;   // idbuf[0] = token id
var<immediate> H: u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let k = g.x; let id = idbuf[0]; if (k >= H) { return; }
  let v = unpack4xI8(w[id*(H/4u) + (k>>2u)]); let lane = k & 3u;
  var b: i32; if (lane==0u){b=v.x;} else if (lane==1u){b=v.y;} else if (lane==2u){b=v.z;} else {b=v.w;}
  out[k] = f32(b) * scale[id];
}`;
var RMSNORM_T = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> g: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
var<immediate> m: vec2<f32>;   // K, eps
var<workgroup> part: array<f32,256>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x); let base = wid.x * K;
  var s = 0.0; for (var k = tid; k < K; k = k + WG) { let v = x[base+k]; s = s + v*v; }
  part[tid] = s; workgroupBarrier();
  for (var t = WG / 2u; t > 0u; t = t/2u) { if (tid < t) { part[tid] = part[tid] + part[tid+t]; } workgroupBarrier(); }
  let inv = inverseSqrt(part[0]/m.x + m.y);
  for (var k = tid; k < K; k = k + WG) { y[base+k] = x[base+k]*inv*g[k]; }
}`;
var RMSNORM_T_F16 = `
requires immediate_address_space;
enable f16;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> g: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
var<immediate> m: vec2<f32>;   // K, eps
// f32 reduction (see RMSNORM_F16): overflow-safe sum-of-squares, f16 normalize.
var<workgroup> part: array<f32,256>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x); let base = wid.x * K;
  var s = 0.0;
  for (var k = tid; k < K; k = k + WG) { let v = f32(x[base+k]); s = s + v*v; }
  part[tid] = s; workgroupBarrier();
  for (var t = WG / 2u; t > 0u; t = t/2u) { if (tid < t) { part[tid] = part[tid] + part[tid+t]; } workgroupBarrier(); }
  let inv = f16(inverseSqrt(part[0]/m.x + m.y));
  for (var k = tid; k < K; k = k + WG) { y[base+k] = f32( f16(x[base+k]) * inv * f16(g[k]) ); }
}`;
var ROPE_T = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
var<immediate> m: vec4<u32>;   // nHeads, headDim, T, pos0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let T = m.z; let pos0 = m.w; let half = D/2u;
  let perRow = H*half; if (g >= T*perRow) { return; }
  let row = g / perRow; let r = g % perRow; let h = r / half; let j = r % half;
  let rb = row*H*D; let lo = rb + h*D + j; let hi = lo + half; let off = (pos0+row)*D + j;
  let c = cosT[off]; let s = sinT[off]; let xl = x[lo]; let xh = x[hi];
  x[lo] = fma(xl, c, 0.0) + fma(-xh, s, 0.0); x[hi] = fma(xh, c, 0.0) + fma(xl, s, 0.0);
}`;
var ROPE_T_F16 = `
requires immediate_address_space;
enable f16;
@group(0) @binding(0) var<storage,read_write> x: array<f32>;
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
var<immediate> m: vec4<u32>;   // nHeads, headDim, T, pos0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let T = m.z; let pos0 = m.w; let half = D/2u;
  let perRow = H*half; if (g >= T*perRow) { return; }
  let row = g / perRow; let r = g % perRow; let h = r / half; let j = r % half;
  let rb = row*H*D; let lo = rb + h*D + j; let hi = lo + half; let off = (pos0+row)*D + j;
  let c = f16(cosT[off]); let s = f16(sinT[off]); let xl = f16(x[lo]); let xh = f16(x[hi]);
  x[lo] = f32( fma(xl, c, 0.0h) + fma(-xh, s, 0.0h) ); x[hi] = f32( fma(xh, c, 0.0h) + fma(xl, s, 0.0h) );
}`;
var EMBED_T = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> w: array<u32>;
@group(0) @binding(1) var<storage,read> scale: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<storage,read> ids: array<u32>;
var<immediate> m: vec4<u32>;   // T, H, idOffset, _
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let T = m.x; let H = m.y; let N = T*H; let stride = nwg.x * 256u;
  for (var i = gid.x; i < N; i = i + stride) {
    let t = i / H; let k = i % H; let id = ids[m.z + t];
    let v = unpack4xI8(w[id*(H/4u) + (k>>2u)]); let lane = k & 3u;
    var b: i32; if (lane==0u){b=v.x;} else if (lane==1u){b=v.y;} else if (lane==2u){b=v.z;} else {b=v.w;}
    out[i] = f32(b) * scale[id];
  }
}`;
var ATTN_PREFILL = `
enable subgroups;
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> q: array<f32>;       // [T][nHeads*hd]
@group(0) @binding(1) var<storage,read> kc: array<f32>;      // [ctx][nKV*hd]
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>; // [T][nHeads*hd]
var<immediate> m: vec4<u32>;             // nHeads, nKV, hd, T
var<workgroup> ps: array<f32,256>;   // exp-scores for the current key block
var<workgroup> acc: array<f32,128>;  // running weighted-V accumulator (hd<=128)
var<workgroup> red: array<f32,64>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let t = wid.y; let tid = lid.x; let nHeads = m.x; let nKV = m.y; let hd = m.z;
  let ctx = t + 1u; let kvh = h / (nHeads / nKV);
  let qbase = t*nHeads*hd + h*hd; let stride = nKV*hd; let hoff = kvh*hd; let scl = 1.0/sqrt(f32(hd));
  let nsg = (256u + sgsz - 1u) / sgsz;
  for (var d = tid; d < hd; d = d + 256u) { acc[d] = 0.0; }
  var mrun = -1e30; var lrun = 0.0;
  let nblk = (ctx + 255u) / 256u;
  for (var blk = 0u; blk < nblk; blk = blk + 1u) {
    let kbase = blk*256u; let kk = kbase + tid;
    var s = -1e30;
    if (kk < ctx) { var dot = 0.0; let kb = kk*stride + hoff; for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qbase+d]*kc[kb+d]; } s = dot*scl; }
    let sgm = subgroupMax(s); if (sgid == 0u) { red[tid/sgsz] = sgm; }
    workgroupBarrier();                                   // A: block-max partials visible
    var bm = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { bm = max(bm, red[i]); }
    let mnew = max(mrun, bm); let corr = exp(mrun - mnew);
    var p = 0.0; if (kk < ctx) { p = exp(s - mnew); }
    ps[tid] = p;
    workgroupBarrier();                                   // B: bm reads done + ps visible
    let sgs = subgroupAdd(p); if (sgid == 0u) { red[tid/sgsz] = sgs; }
    workgroupBarrier();                                   // C: block-sum partials visible
    var bs = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { bs = bs + red[i]; }
    lrun = lrun*corr + bs;
    let bcount = min(256u, ctx - kbase);
    for (var d = tid; d < hd; d = d + 256u) {
      var aa = acc[d]*corr;
      for (var j = 0u; j < bcount; j = j + 1u) { aa = aa + ps[j]*vc[(kbase+j)*stride + hoff + d]; }
      acc[d] = aa;
    }
    mrun = mnew;
    workgroupBarrier();                                   // D: acc's ps reads done before next block
  }
  let invL = 1.0/lrun;
  for (var d = tid; d < hd; d = d + 256u) { o[qbase + d] = acc[d]*invL; }
}`;
var ATTN_PREFILL_BLOCK = `
enable subgroups;
requires immediate_address_space;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32, qStart:u32, ctx:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
var<immediate> m: Meta;
const BQ = 4u; const BK = 128u;
var<workgroup> ps: array<f32, 512>;    // BQ*BK
var<workgroup> acc: array<f32, 512>;   // BQ*hd (hd<=128)
var<workgroup> red: array<f32, 128>;   // BQ*subgroup-count
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let qBlock = wid.y; let tid = lid.x; let hd = m.hd;
  let kvh = h / (m.nHeads / m.nKV); let stride = m.nKV * hd; let hoff = kvh * hd;
  let nsg = (128u + sgsz - 1u) / sgsz; let scl = 1.0 / sqrt(f32(hd));
  var mrun: array<f32, 4>; var lrun: array<f32, 4>;
  for (var r = 0u; r < BQ; r = r + 1u) { mrun[r] = -1e30; lrun[r] = 0.0; }
  for (var i = tid; i < BQ*hd; i = i + 128u) { acc[i] = 0.0; }
  workgroupBarrier();
  let nblk = (m.ctx + BK - 1u) / BK;
  for (var blk = 0u; blk < nblk; blk = blk + 1u) {
    let kbase = blk * BK; let kk = kbase + tid;
    var score: array<f32, 4>;
    var validQ: array<bool, 4>;
    var dot: array<f32, 4>;
    var corrRun: array<f32, 4>;
    for (var r = 0u; r < BQ; r = r + 1u) {
      let qt = qBlock * BQ + r; let absQ = m.qStart + qt;
      validQ[r] = qt < m.T && kk < m.ctx && kk <= absQ;
      dot[r] = 0.0; score[r] = -1e30;
    }
    if (kk < m.ctx) {
      let kb = kk*stride + hoff;
      for (var d = 0u; d < hd; d = d + 1u) {
        let kval = kc[kb+d];
        for (var r = 0u; r < BQ; r = r + 1u) {
          let qt = qBlock * BQ + r;
          if (validQ[r]) { dot[r] = dot[r] + q[qt*m.nHeads*hd + h*hd + d] * kval; }
        }
      }
      for (var r = 0u; r < BQ; r = r + 1u) {
        if (validQ[r]) { score[r] = dot[r] * scl; }
      }
    }
    for (var r = 0u; r < BQ; r = r + 1u) {
      let s = score[r];
      let sgm = subgroupMax(s);
      if (sgid == 0u) { red[r*32u + tid/sgsz] = sgm; }
      workgroupBarrier();
      var bm = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { bm = max(bm, red[r*32u+i]); }
      let mnew = max(mrun[r], bm); let corr = exp(mrun[r] - mnew);
      corrRun[r] = corr;
      var p = 0.0; if (validQ[r]) { p = exp(s - mnew); }
      ps[r*BK + tid] = p;
      workgroupBarrier();
      let sgs = subgroupAdd(p);
      if (sgid == 0u) { red[r*32u + tid/sgsz] = sgs; }
      workgroupBarrier();
      var bs = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { bs = bs + red[r*32u+i]; }
      lrun[r] = lrun[r] * corr + bs;
      mrun[r] = mnew;
      workgroupBarrier();
    }
    let bcount = min(BK, m.ctx - kbase);
    for (var d = tid; d < hd; d = d + 128u) {
      var aa: array<f32, 4>;
      for (var r = 0u; r < BQ; r = r + 1u) { aa[r] = acc[r*hd+d] * corrRun[r]; }
      for (var j = 0u; j < bcount; j = j + 1u) {
        let vv = vc[(kbase+j)*stride + hoff + d];
        for (var r = 0u; r < BQ; r = r + 1u) { aa[r] = aa[r] + ps[r*BK+j] * vv; }
      }
      for (var r = 0u; r < BQ; r = r + 1u) { acc[r*hd+d] = aa[r]; }
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < BQ; r = r + 1u) {
    let qt = qBlock * BQ + r;
    if (qt < m.T) {
      let invL = 1.0 / lrun[r]; let ob = qt*m.nHeads*hd + h*hd;
      for (var d = tid; d < hd; d = d + 128u) { o[ob+d] = acc[r*hd+d] * invL; }
    }
  }
}`;
var ARGMAX = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read_write> out: array<u32>;
var<immediate> n: u32;
var<workgroup> bv: array<f32,256>; var<workgroup> bi: array<u32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; var v = -1e30; var idx = 0xffffffffu;
  for (var i = tid; i < n; i = i + 256u) { let x = logits[i]; if (x > v || (x == v && i < idx)) { v = x; idx = i; } }
  bv[tid] = v; bi[tid] = idx; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s/2u) { if (tid < s) { let ov = bv[tid+s]; let oi = bi[tid+s]; if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) { bv[tid] = ov; bi[tid] = oi; } } workgroupBarrier(); }
  if (tid == 0u) { out[0] = bi[0]; }
}`;
var TOPK_SELECT = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read_write> ids: array<u32>;
@group(0) @binding(2) var<storage,read_write> vals: array<f32>;
var<immediate> m: vec2<u32>; // vocabSize, selectedCount
var<workgroup> bv: array<f32,256>; var<workgroup> bi: array<u32,256>;
fn alreadySelected(id: u32, n: u32) -> bool {
  for (var j = 0u; j < n; j = j + 1u) { if (ids[j] == id) { return true; } }
  return false;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let n = m.x; let selected = m.y;
  var v = -1e30; var idx = 0xffffffffu;
  for (var i = tid; i < n; i = i + 256u) {
    let x = logits[i];
    if (!alreadySelected(i, selected) && (x > v || (x == v && i < idx))) { v = x; idx = i; }
  }
  bv[tid] = v; bi[tid] = idx; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s/2u) {
    if (tid < s) {
      let ov = bv[tid+s]; let oi = bi[tid+s];
      if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) { bv[tid] = ov; bi[tid] = oi; }
    }
    workgroupBarrier();
  }
  if (tid == 0u) { ids[selected] = bi[0]; vals[selected] = bv[0]; }
}`;
var SAMPLE_TOPK = `
requires immediate_address_space;
struct Meta { k:u32, pad:u32, temp:f32, r:f32 };
@group(0) @binding(0) var<storage,read> ids: array<u32>;
@group(0) @binding(1) var<storage,read> vals: array<f32>;
@group(0) @binding(2) var<storage,read_write> outId: array<u32>;  // [1] the chosen token
var<immediate> m: Meta;
var<workgroup> s: array<f32, 64>;    // working softmax probs / prefix sums (small k)
var<workgroup> red: array<f32, 64>;  // reduction scratch for the softmax denominator
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let k = m.k;
  let temp = m.temp;
  let r = m.r;
  let t = select(temp, 1.0, temp <= 0.0);

  // Load + temperature scale into shared (one thread per slot)
  var v = -1e30;
  if (tid < k) {
    let lv = vals[tid];
    v = lv;
    if (t != 1.0) { v = lv / t; }
  }
  let ev = select(0.0, exp(v), tid < k);
  s[tid] = ev;
  red[tid] = ev;
  workgroupBarrier();

  // sum
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (tid < stride && (tid + stride) < 64u) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
  }
  let sum = red[0];
  let invSum = select(0.0, 1.0 / sum, sum > 0.0);

  // normalize + prefix sum for nucleus / categorical pick
  if (tid < k) {
    s[tid] = s[tid] * invSum;
  } else {
    s[tid] = 0.0;
  }
  workgroupBarrier();

  // prefix sum (small k, simple scan)
  for (var stride = 1u; stride < 64u; stride = stride * 2u) {
    var add = 0.0;
    if (tid >= stride && tid < 64u) {
      add = s[tid - stride];
    }
    workgroupBarrier();
    if (tid >= stride && tid < 64u) {
      s[tid] = s[tid] + add;
    }
    workgroupBarrier();
  }

  // find the smallest j such that prefix[j] >= r  (or last if r>=1)
  if (tid == 0u) {
    var chosen = select(0u, k - 1u, k > 0u);
    if (sum > 0.0) {
      for (var j = 0u; j < k; j = j + 1u) {
        let pj = s[j];
        if (r <= pj) { chosen = j; break; }
      }
    }
    outId[0] = select(0u, ids[chosen], k > 0u);
  }
}`;
var GEMV4 = `
enable subgroups;
requires immediate_address_space;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read> loraD: array<f32>;
@group(0) @binding(5) var<storage,read> loraB: array<f32>;
@group(0) @binding(6) var<storage,read_write> y: array<f32>;
var<immediate> m: Meta;
var<workgroup> part: array<f32,64>;       // one slot per subgroup
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }               // workgroup-uniform: whole group exits together
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + 64u) {
    let word = w[rb+c]; let bk = c*8u; let sc = scale[sbase + (bk >> 7u)];
    var p = 0.0;
    p = p + x[bk]    * f32(i32(word << 28u) >> 28u);
    p = p + x[bk+1u] * f32(i32(word << 24u) >> 28u);
    p = p + x[bk+2u] * f32(i32(word << 20u) >> 28u);
    p = p + x[bk+3u] * f32(i32(word << 16u) >> 28u);
    p = p + x[bk+4u] * f32(i32(word << 12u) >> 28u);
    p = p + x[bk+5u] * f32(i32(word << 8u)  >> 28u);
    p = p + x[bk+6u] * f32(i32(word << 4u)  >> 28u);
    p = p + x[bk+7u] * f32(i32(word)        >> 28u);
    acc = acc + p * sc;
  }
  let ssum = subgroupAdd(acc);            // reduce within subgroup (no barrier)
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    if (m.hasBias == 1u) { o = o + bias[n]; }
    if (m.hasLora == 1u) { var dl = 0.0; for (var r = 0u; r < m.rank; r = r + 1u) { dl = dl + loraD[r] * loraB[r*m.N + n]; } o = o + m.scaleLo * dl; }
    y[n] = o;
  }
}`;
var GEMV4_ADD = `
enable subgroups;
requires immediate_address_space;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read> loraD: array<f32>;
@group(0) @binding(5) var<storage,read> loraB: array<f32>;
@group(0) @binding(6) var<storage,read_write> y: array<f32>;
var<immediate> m: Meta;
var<workgroup> part: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + 64u) {
    let word = w[rb+c]; let bk = c*8u; let sc = scale[sbase + (bk >> 7u)];
    var p = 0.0;
    p = p + x[bk]    * f32(i32(word << 28u) >> 28u);
    p = p + x[bk+1u] * f32(i32(word << 24u) >> 28u);
    p = p + x[bk+2u] * f32(i32(word << 20u) >> 28u);
    p = p + x[bk+3u] * f32(i32(word << 16u) >> 28u);
    p = p + x[bk+4u] * f32(i32(word << 12u) >> 28u);
    p = p + x[bk+5u] * f32(i32(word << 8u)  >> 28u);
    p = p + x[bk+6u] * f32(i32(word << 4u)  >> 28u);
    p = p + x[bk+7u] * f32(i32(word)        >> 28u);
    acc = acc + p * sc;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    if (m.hasBias == 1u) { o = o + bias[n]; }
    if (m.hasLora == 1u) { var dl = 0.0; for (var r = 0u; r < m.rank; r = r + 1u) { dl = dl + loraD[r] * loraB[r*m.N + n]; } o = o + m.scaleLo * dl; }
    y[n] = y[n] + o;
  }
}`;
var QKV_GEMV4 = `
enable subgroups;
requires immediate_address_space;
struct Meta { K:u32, totalN:u32, qN:u32, kN:u32, vN:u32, gpr:u32, gridX:u32, p0:u32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read> bias: array<f32>;
@group(0) @binding(4) var<storage,read_write> qOut: array<f32>;
@group(0) @binding(5) var<storage,read_write> kOut: array<f32>;
@group(0) @binding(6) var<storage,read_write> vOut: array<f32>;
var<immediate> m: Meta;
var<workgroup> part: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.totalN) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + 64u) {
    let word = w[rb+c]; let bk = c*8u; let sc = scale[sbase + (bk >> 7u)];
    var p = 0.0;
    p = p + x[bk]    * f32(i32(word << 28u) >> 28u);
    p = p + x[bk+1u] * f32(i32(word << 24u) >> 28u);
    p = p + x[bk+2u] * f32(i32(word << 20u) >> 28u);
    p = p + x[bk+3u] * f32(i32(word << 16u) >> 28u);
    p = p + x[bk+4u] * f32(i32(word << 12u) >> 28u);
    p = p + x[bk+5u] * f32(i32(word << 8u)  >> 28u);
    p = p + x[bk+6u] * f32(i32(word << 4u)  >> 28u);
    p = p + x[bk+7u] * f32(i32(word)        >> 28u);
    acc = acc + p * sc;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    o = o + bias[n];
    if (n < m.qN) {
      qOut[n] = o;
    } else if (n < m.qN + m.kN) {
      kOut[n - m.qN] = o;
    } else {
      vOut[n - m.qN - m.kN] = o;
    }
  }
}`;
var GATE_UP_SILU_GEMV4 = `
enable subgroups;
requires immediate_address_space;
struct Meta { K:u32, N:u32, gpr:u32, gridX:u32, gateRank:u32, upRank:u32, hasGateLora:u32, hasUpLora:u32, gateScaleLo:f32, upScaleLo:f32, p0:f32, p1:f32 };
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> scale: array<f32>;
@group(0) @binding(3) var<storage,read_write> y: array<f32>;
@group(0) @binding(4) var<storage,read> gateD: array<f32>;
@group(0) @binding(5) var<storage,read> gateB: array<f32>;
@group(0) @binding(6) var<storage,read> upD: array<f32>;
@group(0) @binding(7) var<storage,read> upB: array<f32>;
var<immediate> m: Meta;
var<workgroup> partG: array<f32,64>;
var<workgroup> partU: array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rbG = n*K8; let rbU = (m.N + n)*K8;
  let sbG = n*m.gpr; let sbU = (m.N + n)*m.gpr;
  var accG = 0.0; var accU = 0.0;
  for (var c = tid; c < K8; c = c + 64u) {
    let bk = c*8u; let wg = w[rbG+c]; let wu = w[rbU+c];
    let scG = scale[sbG + (bk >> 7u)]; let scU = scale[sbU + (bk >> 7u)];
    let x0=x[bk]; let x1=x[bk+1u]; let x2=x[bk+2u]; let x3=x[bk+3u];
    let x4=x[bk+4u]; let x5=x[bk+5u]; let x6=x[bk+6u]; let x7=x[bk+7u];
    var pg = 0.0; var pu = 0.0;
    pg = pg + x0*f32(i32(wg<<28u)>>28u) + x1*f32(i32(wg<<24u)>>28u) + x2*f32(i32(wg<<20u)>>28u) + x3*f32(i32(wg<<16u)>>28u);
    pg = pg + x4*f32(i32(wg<<12u)>>28u) + x5*f32(i32(wg<<8u)>>28u)  + x6*f32(i32(wg<<4u)>>28u)  + x7*f32(i32(wg)>>28u);
    pu = pu + x0*f32(i32(wu<<28u)>>28u) + x1*f32(i32(wu<<24u)>>28u) + x2*f32(i32(wu<<20u)>>28u) + x3*f32(i32(wu<<16u)>>28u);
    pu = pu + x4*f32(i32(wu<<12u)>>28u) + x5*f32(i32(wu<<8u)>>28u)  + x6*f32(i32(wu<<4u)>>28u)  + x7*f32(i32(wu)>>28u);
    accG = accG + pg * scG; accU = accU + pu * scU;
  }
  let sg = subgroupAdd(accG); let su = subgroupAdd(accU);
  if (sgid == 0u) { partG[tid / sgsz] = sg; partU[tid / sgsz] = su; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var gate = 0.0; var up = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { gate = gate + partG[i]; up = up + partU[i]; }
    if (m.hasGateLora == 1u) {
      var dl = 0.0; for (var r = 0u; r < m.gateRank; r = r + 1u) { dl = dl + gateD[r] * gateB[r*m.N + n]; }
      gate = gate + m.gateScaleLo * dl;
    }
    if (m.hasUpLora == 1u) {
      var dl = 0.0; for (var r = 0u; r < m.upRank; r = r + 1u) { dl = dl + upD[r] * upB[r*m.N + n]; }
      up = up + m.upScaleLo * dl;
    }
    y[n] = (gate / (1.0 + exp(-gate))) * up;
  }
}`;
var DYN_QUANT_X = `
requires immediate_address_space;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> x_q: array<u32>;
@group(0) @binding(2) var<storage, read_write> scale_x: array<f32>;
var<immediate> K: u32;
var<workgroup> sh_max: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = wid.x; let tid = lid.x; let base = g * 128u;
  var local_max = 0.0;
  let idx0 = base + tid; let idx1 = base + tid + 64u;
  if (idx0 < K) { local_max = max(local_max, abs(x[idx0])); }
  if (idx1 < K) { local_max = max(local_max, abs(x[idx1])); }
  sh_max[tid] = local_max;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (tid < s) { sh_max[tid] = max(sh_max[tid], sh_max[tid + s]); }
    workgroupBarrier();
  }
  let gmax = sh_max[0]; let scale = select(gmax / 127.0, 1.0, gmax == 0.0);
  if (tid == 0u) { scale_x[g] = scale; }
  let pidx = base + tid * 4u;
  if (pidx < K) {
    let q0 = clamp(i32(round(x[pidx] / scale)), -128, 127) & 0xff;
    let q1 = clamp(i32(round(x[pidx + 1u] / scale)), -128, 127) & 0xff;
    let q2 = clamp(i32(round(x[pidx + 2u] / scale)), -128, 127) & 0xff;
    let q3 = clamp(i32(round(x[pidx + 3u] / scale)), -128, 127) & 0xff;
    x_q[g * 32u + tid] = u32(q0 | (q1 << 8u) | (q2 << 16u) | (q3 << 24u));
  }
}`;
var DYN_QUANT_X_T = `
requires immediate_address_space;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> x_q: array<u32>;
@group(0) @binding(2) var<storage, read_write> scale_x: array<f32>;
var<immediate> m: vec2<u32>; // K, T
var<workgroup> sh_max: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = wid.x; let t = wid.y; let tid = lid.x; let K = m.x; let T = m.y;
  if (t >= T) { return; }
  let row_base = t * K; let base = row_base + g * 128u;
  var local_max = 0.0;
  let idx0 = base + tid; let idx1 = base + tid + 64u;
  if (g * 128u + tid < K) { local_max = max(local_max, abs(x[idx0])); }
  if (g * 128u + tid + 64u < K) { local_max = max(local_max, abs(x[idx1])); }
  sh_max[tid] = local_max;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (tid < s) { sh_max[tid] = max(sh_max[tid], sh_max[tid + s]); }
    workgroupBarrier();
  }
  let gmax = sh_max[0]; let scale = select(gmax / 127.0, 1.0, gmax == 0.0);
  let groupsPerRow = K / 128u;
  if (tid == 0u) { scale_x[t * groupsPerRow + g] = scale; }
  let pidx = base + tid * 4u;
  if (g * 128u + tid * 4u < K) {
    let q0 = clamp(i32(round(x[pidx] / scale)), -128, 127) & 0xff;
    let q1 = clamp(i32(round(x[pidx + 1u] / scale)), -128, 127) & 0xff;
    let q2 = clamp(i32(round(x[pidx + 2u] / scale)), -128, 127) & 0xff;
    let q3 = clamp(i32(round(x[pidx + 3u] / scale)), -128, 127) & 0xff;
    x_q[t * (K / 4u) + g * 32u + tid] = u32(q0 | (q1 << 8u) | (q2 << 16u) | (q3 << 24u));
  }
}`;
var GEMV4_W4A8 = /* @__PURE__ */ __name((hasDP4a, wgSize = 64) => `
enable subgroups;
${hasDP4a ? `
enable packed_4x8_integer_dot_product;
` : ""}
requires immediate_address_space;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> w: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;
@group(0) @binding(5) var<storage,read> loraD: array<f32>;
@group(0) @binding(6) var<storage,read> loraB: array<f32>;
@group(0) @binding(7) var<storage,read_write> y: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

var<workgroup> part: array<f32, ${wgSize}>;
@compute @workgroup_size(${wgSize})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + ${wgSize}u) {
    let word = w[rb+c]; let bk = c*8u;
    let sc_w = scale[sbase + (bk >> 7u)];
    let sc_x = scale_x[bk >> 7u];
    let w0 = (i32(word << 28u) >> 28u) & 0xff;
    let w1 = (i32(word << 24u) >> 28u) & 0xff;
    let w2 = (i32(word << 20u) >> 28u) & 0xff;
    let w3 = (i32(word << 16u) >> 28u) & 0xff;
    let w4 = (i32(word << 12u) >> 28u) & 0xff;
    let w5 = (i32(word << 8u)  >> 28u) & 0xff;
    let w6 = (i32(word << 4u)  >> 28u) & 0xff;
    let w7 = (i32(word)        >> 28u) & 0xff;
    let pw0 = u32(w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u));
    let pw1 = u32(w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u));
    let px0 = x_q[c * 2u];
    let px1 = x_q[c * 2u + 1u];
    let sum = dot4I8Packed(pw0, px0) + dot4I8Packed(pw1, px1);
    acc = acc + f32(sum) * sc_w * sc_x;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (${wgSize}u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    if (m.hasBias == 1u) { o = o + bias[n]; }
    if (m.hasLora == 1u) { var dl = 0.0; for (var r = 0u; r < m.rank; r = r + 1u) { dl = dl + loraD[r] * loraB[r*m.N + n]; } o = o + m.scaleLo * dl; }
    y[n] = o;
  }
}
`, "GEMV4_W4A8");
var GEMV4_ADD_W4A8 = /* @__PURE__ */ __name((hasDP4a, wgSize = 64) => `
enable subgroups;
${hasDP4a ? `
enable packed_4x8_integer_dot_product;
` : ""}
requires immediate_address_space;
struct Meta { K:u32, N:u32, rank:u32, hasBias:u32, hasLora:u32, gridX:u32, scaleLo:f32, gpr:u32 };
@group(0) @binding(0) var<storage,read> x_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> w: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;
@group(0) @binding(5) var<storage,read> loraD: array<f32>;
@group(0) @binding(6) var<storage,read> loraB: array<f32>;
@group(0) @binding(7) var<storage,read_write> y: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

var<workgroup> part: array<f32, ${wgSize}>;
@compute @workgroup_size(${wgSize})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + ${wgSize}u) {
    let word = w[rb+c]; let bk = c*8u;
    let sc_w = scale[sbase + (bk >> 7u)];
    let sc_x = scale_x[bk >> 7u];
    let w0 = (i32(word << 28u) >> 28u) & 0xff;
    let w1 = (i32(word << 24u) >> 28u) & 0xff;
    let w2 = (i32(word << 20u) >> 28u) & 0xff;
    let w3 = (i32(word << 16u) >> 28u) & 0xff;
    let w4 = (i32(word << 12u) >> 28u) & 0xff;
    let w5 = (i32(word << 8u)  >> 28u) & 0xff;
    let w6 = (i32(word << 4u)  >> 28u) & 0xff;
    let w7 = (i32(word)        >> 28u) & 0xff;
    let pw0 = u32(w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u));
    let pw1 = u32(w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u));
    let px0 = x_q[c * 2u];
    let px1 = x_q[c * 2u + 1u];
    let sum = dot4I8Packed(pw0, px0) + dot4I8Packed(pw1, px1);
    acc = acc + f32(sum) * sc_w * sc_x;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (${wgSize}u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    if (m.hasBias == 1u) { o = o + bias[n]; }
    if (m.hasLora == 1u) { var dl = 0.0; for (var r = 0u; r < m.rank; r = r + 1u) { dl = dl + loraD[r] * loraB[r*m.N + n]; } o = o + m.scaleLo * dl; }
    y[n] = y[n] + o;
  }
}
`, "GEMV4_ADD_W4A8");
var QKV_GEMV4_W4A8 = /* @__PURE__ */ __name((hasDP4a, wgSize = 64) => `
enable subgroups;
${hasDP4a ? `
enable packed_4x8_integer_dot_product;
` : ""}
requires immediate_address_space;
struct Meta { K:u32, totalN:u32, qN:u32, kN:u32, vN:u32, gpr:u32, gridX:u32, p0:u32 };
@group(0) @binding(0) var<storage,read> x_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> w: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;
@group(0) @binding(5) var<storage,read_write> qOut: array<f32>;
@group(0) @binding(6) var<storage,read_write> kOut: array<f32>;
@group(0) @binding(7) var<storage,read_write> vOut: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

var<workgroup> part: array<f32, ${wgSize}>;
@compute @workgroup_size(${wgSize})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.totalN) { return; }
  let K8 = m.K/8u; let rb = n*K8; let sbase = n*m.gpr;
  var acc = 0.0;
  for (var c = tid; c < K8; c = c + ${wgSize}u) {
    let word = w[rb+c]; let bk = c*8u;
    let sc_w = scale[sbase + (bk >> 7u)];
    let sc_x = scale_x[bk >> 7u];
    let w0 = (i32(word << 28u) >> 28u) & 0xff;
    let w1 = (i32(word << 24u) >> 28u) & 0xff;
    let w2 = (i32(word << 20u) >> 28u) & 0xff;
    let w3 = (i32(word << 16u) >> 28u) & 0xff;
    let w4 = (i32(word << 12u) >> 28u) & 0xff;
    let w5 = (i32(word << 8u)  >> 28u) & 0xff;
    let w6 = (i32(word << 4u)  >> 28u) & 0xff;
    let w7 = (i32(word)        >> 28u) & 0xff;
    let pw0 = u32(w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u));
    let pw1 = u32(w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u));
    let px0 = x_q[c * 2u];
    let px1 = x_q[c * 2u + 1u];
    let sum = dot4I8Packed(pw0, px0) + dot4I8Packed(pw1, px1);
    acc = acc + f32(sum) * sc_w * sc_x;
  }
  let ssum = subgroupAdd(acc);
  if (sgid == 0u) { part[tid / sgsz] = ssum; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (${wgSize}u + sgsz - 1u) / sgsz; var o = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o = o + part[i]; }
    o = o + bias[n];
    if (n < m.qN) {
      qOut[n] = o;
    } else if (n < m.qN + m.kN) {
      kOut[n - m.qN] = o;
    } else {
      vOut[n - m.qN - m.kN] = o;
    }
  }
}
`, "QKV_GEMV4_W4A8");
var GATE_UP_SILU_GEMV4_W4A8 = /* @__PURE__ */ __name((hasDP4a, wgSize = 64) => `
enable subgroups;
${hasDP4a ? `
enable packed_4x8_integer_dot_product;
` : ""}
requires immediate_address_space;
struct Meta { K:u32, N:u32, gpr:u32, gridX:u32, gateRank:u32, upRank:u32, hasGateLora:u32, hasUpLora:u32, gateScaleLo:f32, upScaleLo:f32, p0:f32, p1:f32 };
@group(0) @binding(0) var<storage,read> x_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> w: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read_write> y: array<f32>;
@group(0) @binding(5) var<storage,read> gateD: array<f32>;
@group(0) @binding(6) var<storage,read> gateB: array<f32>;
@group(0) @binding(7) var<storage,read> upD: array<f32>;
@group(0) @binding(8) var<storage,read> upB: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

var<workgroup> partG: array<f32, ${wgSize}>;
var<workgroup> partU: array<f32, ${wgSize}>;
@compute @workgroup_size(${wgSize})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let n = wid.x + wid.y * m.gridX; let tid = lid.x;
  if (n >= m.N) { return; }
  let K8 = m.K/8u; let rbG = n*K8; let rbU = (m.N + n)*K8;
  let sbG = n*m.gpr; let sbU = (m.N + n)*m.gpr;
  var accG = 0.0; var accU = 0.0;
  for (var c = tid; c < K8; c = c + ${wgSize}u) {
    let wg = w[rbG+c]; let wu = w[rbU+c];
    let bk = c*8u;
    let scG = scale[sbG + (bk >> 7u)]; let scU = scale[sbU + (bk >> 7u)];
    let sc_x = scale_x[bk >> 7u];
    let wg0 = (i32(wg << 28u) >> 28u) & 0xff;
    let wg1 = (i32(wg << 24u) >> 28u) & 0xff;
    let wg2 = (i32(wg << 20u) >> 28u) & 0xff;
    let wg3 = (i32(wg << 16u) >> 28u) & 0xff;
    let wg4 = (i32(wg << 12u) >> 28u) & 0xff;
    let wg5 = (i32(wg << 8u)  >> 28u) & 0xff;
    let wg6 = (i32(wg << 4u)  >> 28u) & 0xff;
    let wg7 = (i32(wg)        >> 28u) & 0xff;
    let pwg0 = u32(wg0 | (wg1 << 8u) | (wg2 << 16u) | (wg3 << 24u));
    let pwg1 = u32(wg4 | (wg5 << 8u) | (wg6 << 16u) | (wg7 << 24u));
    let wu0 = (i32(wu << 28u) >> 28u) & 0xff;
    let wu1 = (i32(wu << 24u) >> 28u) & 0xff;
    let wu2 = (i32(wu << 20u) >> 28u) & 0xff;
    let wu3 = (i32(wu << 16u) >> 28u) & 0xff;
    let wu4 = (i32(wu << 12u) >> 28u) & 0xff;
    let wu5 = (i32(wu << 8u)  >> 28u) & 0xff;
    let wu6 = (i32(wu << 4u)  >> 28u) & 0xff;
    let wu7 = (i32(wu)        >> 28u) & 0xff;
    let pwu0 = u32(wu0 | (wu1 << 8u) | (wu2 << 16u) | (wu3 << 24u));
    let pwu1 = u32(wu4 | (wu5 << 8u) | (wu6 << 16u) | (wu7 << 24u));
    let px0 = x_q[c * 2u];
    let px1 = x_q[c * 2u + 1u];
    let sumG = dot4I8Packed(pwg0, px0) + dot4I8Packed(pwg1, px1);
    let sumU = dot4I8Packed(pwu0, px0) + dot4I8Packed(pwu1, px1);
    accG = accG + f32(sumG) * scG * sc_x;
    accU = accU + f32(sumU) * scU * sc_x;
  }
  let sg = subgroupAdd(accG); let su = subgroupAdd(accU);
  if (sgid == 0u) { partG[tid / sgsz] = sg; partU[tid / sgsz] = su; }
  workgroupBarrier();
  if (tid == 0u) {
    let nsg = (${wgSize}u + sgsz - 1u) / sgsz; var gate = 0.0; var up = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { gate = gate + partG[i]; up = up + partU[i]; }
    if (m.hasGateLora == 1u) {
      var dl = 0.0; for (var r = 0u; r < m.gateRank; r = r + 1u) { dl = dl + gateD[r] * gateB[r*m.N + n]; }
      gate = gate + m.gateScaleLo * dl;
    }
    if (m.hasUpLora == 1u) {
      var dl = 0.0; for (var r = 0u; r < m.upRank; r = r + 1u) { dl = dl + upD[r] * upB[r*m.N + n]; }
      up = up + m.upScaleLo * dl;
    }
    y[n] = (gate / (1.0 + exp(-gate))) * up;
  }
}
`, "GATE_UP_SILU_GEMV4_W4A8");
var GEMM4_W4A8 = /* @__PURE__ */ __name((hasDP4a) => `
enable subgroups;
${hasDP4a ? `
enable packed_4x8_integer_dot_product;
` : ""}
requires immediate_address_space;
struct Meta { K:u32, N:u32, T:u32, gpr:u32, hasBias:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> A_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> W: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;
@group(0) @binding(5) var<storage,read_write> Y: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

const BM = 16u; const BN = 64u;
var<workgroup> As_q: array<u32, 32>;
var<workgroup> As_scale: array<f32, 16>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tTile = wid.y * BM; let col = wid.x * BN + lid.x; let valid = col < m.N;
  let K8 = m.K/8u; let rb = col*K8;
  var acc: array<f32, 16>;
  for (var i = 0u; i < BM; i = i + 1u) { acc[i] = 0.0; }
  let groupsPerRow = m.K / 128u;
  for (var c = 0u; c < K8; c = c + 1u) {
    if (lid.x < BM * 2u) {
      let tt = lid.x / 2u; let trow = tTile + tt; let wordIdx = lid.x % 2u;
      As_q[lid.x] = select(0u, A_q[trow * (m.K / 4u) + c * 2u + wordIdx], trow < m.T);
    }
    if (lid.x < BM) {
      let trow = tTile + lid.x;
      As_scale[lid.x] = select(0.0, scale_x[trow * groupsPerRow + ((c * 8u) >> 7u)], trow < m.T);
    }
    workgroupBarrier();
    if (valid) {
      let word = W[rb + c]; let sc_w = scale[col*m.gpr + ((c*8u) >> 7u)];
      let w0 = (i32(word << 28u) >> 28u) & 0xff;
      let w1 = (i32(word << 24u) >> 28u) & 0xff;
      let w2 = (i32(word << 20u) >> 28u) & 0xff;
      let w3 = (i32(word << 16u) >> 28u) & 0xff;
      let w4 = (i32(word << 12u) >> 28u) & 0xff;
      let w5 = (i32(word << 8u)  >> 28u) & 0xff;
      let w6 = (i32(word << 4u)  >> 28u) & 0xff;
      let w7 = (i32(word)        >> 28u) & 0xff;
      let pw0 = u32(w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u));
      let pw1 = u32(w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u));
      for (var t = 0u; t < BM; t = t + 1u) {
        let px0 = As_q[t * 2u]; let px1 = As_q[t * 2u + 1u];
        let sum = dot4I8Packed(pw0, px0) + dot4I8Packed(pw1, px1);
        acc[t] = acc[t] + f32(sum) * sc_w * As_scale[t];
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    let bv = select(0.0, bias[col], m.hasBias == 1u);
    for (var t = 0u; t < BM; t = t + 1u) { let trow = tTile + t; if (trow < m.T) { Y[trow*m.N + col] = acc[t] + bv; } }
  }
}
`, "GEMM4_W4A8");
var GEMM4_ADD_T_W4A8 = /* @__PURE__ */ __name((hasDP4a) => `
enable subgroups;
${hasDP4a ? `
enable packed_4x8_integer_dot_product;
` : ""}
requires immediate_address_space;
struct Meta { K:u32, N:u32, T:u32, gpr:u32, hasBias:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> A_q: array<u32>;
@group(0) @binding(1) var<storage,read> scale_x: array<f32>;
@group(0) @binding(2) var<storage,read> W: array<u32>;
@group(0) @binding(3) var<storage,read> scale: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;
@group(0) @binding(5) var<storage,read_write> Y: array<f32>;
var<immediate> m: Meta;

${hasDP4a ? "" : `
fn dot4I8Packed(a: u32, b: u32) -> i32 {
  let va = unpack4xI8(a);
  let vb = unpack4xI8(b);
  return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}
`}

const BM = 16u; const BN = 64u;
var<workgroup> As_q: array<u32, 32>;
var<workgroup> As_scale: array<f32, 16>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tTile = wid.y * BM; let col = wid.x * BN + lid.x; let valid = col < m.N;
  let K8 = m.K/8u; let rb = col*K8;
  var acc: array<f32, 16>;
  for (var i = 0u; i < BM; i = i + 1u) { acc[i] = 0.0; }
  let groupsPerRow = m.K / 128u;
  for (var c = 0u; c < K8; c = c + 1u) {
    if (lid.x < BM * 2u) {
      let tt = lid.x / 2u; let trow = tTile + tt; let wordIdx = lid.x % 2u;
      As_q[lid.x] = select(0u, A_q[trow * (m.K / 4u) + c * 2u + wordIdx], trow < m.T);
    }
    if (lid.x < BM) {
      let trow = tTile + lid.x;
      As_scale[lid.x] = select(0.0, scale_x[trow * groupsPerRow + ((c * 8u) >> 7u)], trow < m.T);
    }
    workgroupBarrier();
    if (valid) {
      let word = W[rb + c]; let sc_w = scale[col*m.gpr + ((c*8u) >> 7u)];
      let w0 = (i32(word << 28u) >> 28u) & 0xff;
      let w1 = (i32(word << 24u) >> 28u) & 0xff;
      let w2 = (i32(word << 20u) >> 28u) & 0xff;
      let w3 = (i32(word << 16u) >> 28u) & 0xff;
      let w4 = (i32(word << 12u) >> 28u) & 0xff;
      let w5 = (i32(word << 8u)  >> 28u) & 0xff;
      let w6 = (i32(word << 4u)  >> 28u) & 0xff;
      let w7 = (i32(word)        >> 28u) & 0xff;
      let pw0 = u32(w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u));
      let pw1 = u32(w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u));
      for (var t = 0u; t < BM; t = t + 1u) {
        let px0 = As_q[t * 2u]; let px1 = As_q[t * 2u + 1u];
        let sum = dot4I8Packed(pw0, px0) + dot4I8Packed(pw1, px1);
        acc[t] = acc[t] + f32(sum) * sc_w * As_scale[t];
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    let bv = select(0.0, bias[col], m.hasBias == 1u);
    for (var t = 0u; t < BM; t = t + 1u) {
      let trow = tTile + t;
      if (trow < m.T) { Y[trow*m.N + col] = Y[trow*m.N + col] + acc[t] + bv; }
    }
  }
}
`, "GEMM4_ADD_T_W4A8");
var WRITE_KV_PAGE = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read> k_src: array<f32>;
@group(0) @binding(1) var<storage,read> v_src: array<f32>;
@group(0) @binding(2) var<storage,read_write> kc: array<f32>;
@group(0) @binding(3) var<storage,read_write> vc: array<f32>;
@group(0) @binding(4) var<storage,read> block_table: array<u32>;
var<immediate> m: vec4<u32>; // pos, seq_id, max_blocks, kvd
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; let pos = m.x; let seq_id = m.y; let max_blocks = m.z; let kvd = m.w;
  if (idx >= kvd) { return; }
  let page_idx = block_table[seq_id * max_blocks + (pos / 16u)];
  let page_offset = pos % 16u;
  let physical_pos = page_idx * 16u + page_offset;
  let dst_offset = physical_pos * kvd + idx;
  kc[dst_offset] = k_src[idx];
  vc[dst_offset] = v_src[idx];
}`;
var WRITE_KV_PAGE_BATCH = `
requires immediate_address_space;
struct KVBatchMeta { T:u32, seq_id:u32, max_blocks:u32, kvd:u32, off:u32 };
@group(0) @binding(0) var<storage,read> k_src: array<f32>;
@group(0) @binding(1) var<storage,read> v_src: array<f32>;
@group(0) @binding(2) var<storage,read_write> kc: array<f32>;
@group(0) @binding(3) var<storage,read_write> vc: array<f32>;
@group(0) @binding(4) var<storage,read> block_table: array<u32>;
var<immediate> m: KVBatchMeta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; let T = m.T; let seq_id = m.seq_id; let max_blocks = m.max_blocks; let kvd = m.kvd; let off = m.off;
  let total = T * kvd; if (idx >= total) { return; }
  let t = idx / kvd; let d = idx % kvd;
  let page_idx = block_table[seq_id * max_blocks + ((off + t) / 16u)];
  let page_offset = (off + t) % 16u;
  let physical_pos = page_idx * 16u + page_offset;
  let dst_offset = physical_pos * kvd + d;
  kc[dst_offset] = k_src[idx];
  vc[dst_offset] = v_src[idx];
}`;
var ATTN_PARTIAL_PAGED = `
enable subgroups;
requires immediate_address_space;
struct Meta { nHeads:u32, nKV:u32, ctx:u32, hd:u32, nsplit:u32, chunk:u32, seq_id:u32, max_blocks:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> pm: array<f32>;
@group(0) @binding(4) var<storage,read_write> pz: array<f32>;
@group(0) @binding(5) var<storage,read_write> po: array<f32>;
@group(0) @binding(6) var<storage,read> block_table: array<u32>;
var<immediate> m: Meta;
var<workgroup> sc: array<f32,128>;
var<workgroup> red: array<f32,32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let s = wid.y; let tid = lid.x;
  let nHeads = m.nHeads; let nKV = m.nKV; let ctx = m.ctx; let hd = m.hd;
  let nsplit = m.nsplit; let chunk = m.chunk; let seq_id = m.seq_id; let max_blocks = m.max_blocks;
  let kvh = h / (nHeads / nKV);
  let qbase = h*hd; let stride = nKV*hd; let hoff = kvh*hd; let scale = 1.0/sqrt(f32(hd));
  let nsg = (128u + sgsz - 1u) / sgsz;
  let t0 = s*chunk; var t1 = t0 + chunk; if (t1 > ctx) { t1 = ctx; }
  let t = t0 + tid; var sv = -1e30;
  if (t < t1) {
    var dot = 0.0;
    let page_idx = block_table[seq_id * max_blocks + (t / 16u)];
    let page_offset = t % 16u;
    let kb = (page_idx * 16u + page_offset) * stride + hoff;
    for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qbase+d]*kc[kb+d]; }
    sv = dot*scale;
  }
  let sgm = subgroupMax(sv); if (sgid == 0u) { red[tid/sgsz] = sgm; }
  workgroupBarrier();
  var M = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { M = max(M, red[i]); }
  workgroupBarrier();
  var ev = 0.0; if (t < t1) { ev = exp(sv - M); } sc[tid] = ev;
  let sgs = subgroupAdd(ev); if (sgid == 0u) { red[tid/sgsz] = sgs; }
  workgroupBarrier();
  var Z = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { Z = Z + red[i]; }
  workgroupBarrier();
  let len = t1 - t0; let pbase = (h*nsplit + s)*hd;
  for (var d = tid; d < hd; d = d + 128u) {
    var acc = 0.0;
    for (var tt = 0u; tt < len; tt = tt + 1u) {
      let t_curr = t0 + tt;
      let page_idx = block_table[seq_id * max_blocks + (t_curr / 16u)];
      let page_offset = t_curr % 16u;
      let physical_t = page_idx * 16u + page_offset;
      acc = acc + sc[tt]*vc[physical_t*stride + hoff + d];
    }
    po[pbase + d] = acc;
  }
  if (tid == 0u) { pm[h*nsplit + s] = M; pz[h*nsplit + s] = Z; }
}`;
var ATTN_PREFILL_PAGED = `
enable subgroups;
requires immediate_address_space;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32, seq_id:u32, max_blocks:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
@group(0) @binding(4) var<storage,read> block_table: array<u32>;
var<immediate> m: Meta;
var<workgroup> ps: array<f32,256>;
var<workgroup> acc: array<f32,128>;
var<workgroup> red: array<f32,64>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let t = wid.y; let tid = lid.x; let nHeads = m.nHeads; let nKV = m.nKV; let hd = m.hd;
  let ctx = t + 1u; let kvh = h / (nHeads / nKV);
  let qbase = t*nHeads*hd + h*hd; let stride = nKV*hd; let hoff = kvh*hd; let scl = 1.0/sqrt(f32(hd));
  let nsg = (256u + sgsz - 1u) / sgsz;
  let seq_id = m.seq_id; let max_blocks = m.max_blocks;
  for (var d = tid; d < hd; d = d + 256u) { acc[d] = 0.0; }
  var mrun = -1e30; var lrun = 0.0;
  let nblk = (ctx + 255u) / 256u;
  for (var blk = 0u; blk < nblk; blk = blk + 1u) {
    let kbase = blk*256u; let kk = kbase + tid;
    var s = -1e30;
    if (kk < ctx) {
      var dot = 0.0;
      let page_idx = block_table[seq_id * max_blocks + (kk / 16u)];
      let page_offset = kk % 16u;
      let kb = (page_idx * 16u + page_offset)*stride + hoff;
      for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qbase+d]*kc[kb+d]; }
      s = dot*scl;
    }
    let sgm = subgroupMax(s); if (sgid == 0u) { red[tid/sgsz] = sgm; }
    workgroupBarrier();
    var bm = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { bm = max(bm, red[i]); }
    let mnew = max(mrun, bm); let corr = exp(mrun - mnew);
    var p = 0.0; if (kk < ctx) { p = exp(s - mnew); }
    ps[tid] = p;
    workgroupBarrier();
    let sgs = subgroupAdd(p); if (sgid == 0u) { red[tid/sgsz] = sgs; }
    workgroupBarrier();
    var bs = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { bs = bs + red[i]; }
    lrun = lrun*corr + bs;
    let bcount = min(256u, ctx - kbase);
    for (var d = tid; d < hd; d = d + 256u) {
      var aa = acc[d]*corr;
      for (var j = 0u; j < bcount; j = j + 1u) {
        let t_curr = kbase + j;
        let page_idx = block_table[seq_id * max_blocks + (t_curr / 16u)];
        let page_offset = t_curr % 16u;
        let physical_t = page_idx * 16u + page_offset;
        aa = aa + ps[j]*vc[physical_t*stride + hoff + d];
      }
      acc[d] = aa;
    }
    mrun = mnew;
    workgroupBarrier();
  }
  let invL = 1.0/lrun;
  for (var d = tid; d < hd; d = d + 256u) { o[qbase + d] = acc[d]*invL; }
}`;
var ATTN_PREFILL_BLOCK_PAGED = `
enable subgroups;
requires immediate_address_space;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32, qStart:u32, ctx:u32, seq_id:u32, max_blocks:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
@group(0) @binding(4) var<storage,read> block_table: array<u32>;
var<immediate> m: Meta;
const BQ = 4u; const BK = 128u;
var<workgroup> ps: array<f32, 512>;
var<workgroup> acc: array<f32, 512>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  let h = wid.x; let qBlock = wid.y; let tid = lid.x; let hd = m.hd;
  let kvh = h / (m.nHeads / m.nKV); let stride = m.nKV * hd; let hoff = kvh * hd;
  let nsg = (128u + sgsz - 1u) / sgsz; let scl = 1.0 / sqrt(f32(hd));
  let seq_id = m.seq_id; let max_blocks = m.max_blocks;
  var mrun: array<f32, 4>; var lrun: array<f32, 4>;
  for (var r = 0u; r < BQ; r = r + 1u) { mrun[r] = -1e30; lrun[r] = 0.0; }
  for (var i = tid; i < BQ*hd; i = i + 128u) { acc[i] = 0.0; }
  workgroupBarrier();
  let nblk = (m.ctx + BK - 1u) / BK;
  for (var blk = 0u; blk < nblk; blk = blk + 1u) {
    let kbase = blk * BK; let kk = kbase + tid;
    var score: array<f32, 4>;
    var validQ: array<bool, 4>;
    var dot: array<f32, 4>;
    var corrRun: array<f32, 4>;
    for (var r = 0u; r < BQ; r = r + 1u) {
      let qt = qBlock * BQ + r; let absQ = m.qStart + qt;
      validQ[r] = qt < m.T && kk < m.ctx && kk <= absQ;
      dot[r] = 0.0; score[r] = -1e30;
    }
    if (kk < m.ctx) {
      let page_idx = block_table[seq_id * max_blocks + (kk / 16u)];
      let page_offset = kk % 16u;
      let kb = (page_idx * 16u + page_offset)*stride + hoff;
      for (var d = 0u; d < hd; d = d + 1u) {
        let kval = kc[kb+d];
        for (var r = 0u; r < BQ; r = r + 1u) {
          let qt = qBlock * BQ + r;
          if (validQ[r]) { dot[r] = dot[r] + q[qt*m.nHeads*hd + h*hd + d] * kval; }
        }
      }
      for (var r = 0u; r < BQ; r = r + 1u) {
        if (validQ[r]) { score[r] = dot[r] * scl; }
      }
    }
    for (var r = 0u; r < BQ; r = r + 1u) {
      let s = score[r];
      let sgm = subgroupMax(s);
      if (sgid == 0u) { red[r*32u + tid/sgsz] = sgm; }
      workgroupBarrier();
      var bm = -1e30; for (var i = 0u; i < nsg; i = i + 1u) { bm = max(bm, red[r*32u+i]); }
      let mnew = max(mrun[r], bm); let corr = exp(mrun[r] - mnew);
      corrRun[r] = corr;
      var p = 0.0; if (validQ[r]) { p = exp(s - mnew); }
      ps[r*BK + tid] = p;
      workgroupBarrier();
      let sgs = subgroupAdd(p);
      if (sgid == 0u) { red[r*32u + tid/sgsz] = sgs; }
      workgroupBarrier();
      var bs = 0.0; for (var i = 0u; i < nsg; i = i + 1u) { bs = bs + red[r*32u+i]; }
      lrun[r] = lrun[r] * corr + bs;
      mrun[r] = mnew;
      workgroupBarrier();
    }
    let bcount = min(BK, m.ctx - kbase);
    for (var d = tid; d < hd; d = d + 128u) {
      var aa: array<f32, 4>;
      for (var r = 0u; r < BQ; r = r + 1u) { aa[r] = acc[r*hd+d] * corrRun[r]; }
      for (var j = 0u; j < bcount; j = j + 1u) {
        let t_curr = kbase + j;
        let page_idx = block_table[seq_id * max_blocks + (t_curr / 16u)];
        let page_offset = t_curr % 16u;
        let physical_t = page_idx * 16u + page_offset;
        let vv = vc[physical_t*stride + hoff + d];
        for (var r = 0u; r < BQ; r = r + 1u) { aa[r] = aa[r] + ps[r*BK+j] * vv; }
      }
      for (var r = 0u; r < BQ; r = r + 1u) { acc[r*hd+d] = aa[r]; }
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < BQ; r = r + 1u) {
    let qt = qBlock * BQ + r;
    if (qt < m.T) {
      let invL = 1.0 / lrun[r]; let ob = qt*m.nHeads*hd + h*hd;
      for (var d = tid; d < hd; d = d + 128u) { o[ob+d] = acc[r*hd+d] * invL; }
    }
  }
}`;
var GEMV4_QKV_ROPE_RMS = `
enable subgroups;
requires immediate_address_space;
struct Meta { 
  K: u32, totalPairs: u32, qPairs: u32, kPairs: u32, vPairs: u32, gpr: u32, gridX: u32, 
  pos: u32, headDim: u32, eps: f32,
  qN: u32, kN: u32
};

@group(0) @binding(0) var<storage,read> hidden: array<f32>;      
@group(0) @binding(1) var<storage,read> rms_g: array<f32>;       
@group(0) @binding(2) var<storage,read> w: array<u32>;           
@group(0) @binding(3) var<storage,read> scale: array<f32>;       
@group(0) @binding(4) var<storage,read> bias: array<f32>;        
@group(0) @binding(5) var<storage,read> cosT: array<f32>;
@group(0) @binding(6) var<storage,read> sinT: array<f32>;
@group(0) @binding(7) var<storage,read_write> qOut: array<f32>;  
@group(0) @binding(8) var<storage,read_write> kOut: array<f32>;  
@group(0) @binding(9) var<storage,read_write> vOut: array<f32>;  
var<immediate> m: Meta;

var<workgroup> partSum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgsz: u32, @builtin(subgroup_invocation_id) sgid: u32) {
  
  let pair_idx = wid.x + wid.y * m.gridX;
  if (pair_idx >= m.totalPairs) { return; }
  let tid = lid.x;
  
  var s = 0.0;
  for (var k = tid; k < m.K; k = k + 64u) { let v = hidden[k]; s = s + v*v; }
  let ssum = subgroupAdd(s);
  if (sgid == 0u) { partSum[tid / sgsz] = ssum; }
  workgroupBarrier();
  
  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; var red = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { red = red + partSum[i]; }
    partSum[0] = inverseSqrt(red / f32(m.K) + m.eps);
  }
  workgroupBarrier();
  let inv = partSum[0];

  let half = m.headDim / 2u;
  var n0: u32; var n1: u32;
  var isQ = false; var isK = false; var isV = false;
  var out_idx0: u32; var out_idx1: u32;
  var rope_j: u32 = 0u;

  if (pair_idx < m.qPairs) {
    isQ = true;
    let h = pair_idx / half; let j = pair_idx % half;
    n0 = h * m.headDim + j;
    n1 = n0 + half;
    out_idx0 = n0; out_idx1 = n1;
    rope_j = j;
  } else if (pair_idx < m.qPairs + m.kPairs) {
    isK = true;
    let p = pair_idx - m.qPairs;
    let h = p / half; let j = p % half;
    n0 = m.qN + h * m.headDim + j;
    n1 = n0 + half;
    out_idx0 = h * m.headDim + j; out_idx1 = out_idx0 + half;
    rope_j = j;
  } else {
    isV = true;
    let p = pair_idx - m.qPairs - m.kPairs;
    n0 = m.qN + m.kN + p * 2u;
    n1 = n0 + 1u;
    out_idx0 = p * 2u; out_idx1 = out_idx0 + 1u;
  }

  let K8 = m.K / 8u;
  let rb0 = n0 * K8; let rb1 = n1 * K8;
  let sbase0 = n0 * m.gpr; let sbase1 = n1 * m.gpr;

  var acc0 = 0.0; var acc1 = 0.0;
  
  for (var c = tid; c < K8; c = c + 64u) {
    let w0 = w[rb0 + c]; let w1 = w[rb1 + c];
    let bk = c * 8u;
    let sc0 = scale[sbase0 + (bk >> 7u)]; let sc1 = scale[sbase1 + (bk >> 7u)];
    
    // We compute normalized X on the fly
    let x0 = hidden[bk] * inv * rms_g[bk];
    let x1 = hidden[bk+1u] * inv * rms_g[bk+1u];
    let x2 = hidden[bk+2u] * inv * rms_g[bk+2u];
    let x3 = hidden[bk+3u] * inv * rms_g[bk+3u];
    let x4 = hidden[bk+4u] * inv * rms_g[bk+4u];
    let x5 = hidden[bk+5u] * inv * rms_g[bk+5u];
    let x6 = hidden[bk+6u] * inv * rms_g[bk+6u];
    let x7 = hidden[bk+7u] * inv * rms_g[bk+7u];

    var p0 = 0.0; var p1 = 0.0;
    p0 = p0 + x0 * f32(i32(w0 << 28u) >> 28u); p1 = p1 + x0 * f32(i32(w1 << 28u) >> 28u);
    p0 = p0 + x1 * f32(i32(w0 << 24u) >> 28u); p1 = p1 + x1 * f32(i32(w1 << 24u) >> 28u);
    p0 = p0 + x2 * f32(i32(w0 << 20u) >> 28u); p1 = p1 + x2 * f32(i32(w1 << 20u) >> 28u);
    p0 = p0 + x3 * f32(i32(w0 << 16u) >> 28u); p1 = p1 + x3 * f32(i32(w1 << 16u) >> 28u);
    p0 = p0 + x4 * f32(i32(w0 << 12u) >> 28u); p1 = p1 + x4 * f32(i32(w1 << 12u) >> 28u);
    p0 = p0 + x5 * f32(i32(w0 << 8u)  >> 28u); p1 = p1 + x5 * f32(i32(w1 << 8u)  >> 28u);
    p0 = p0 + x6 * f32(i32(w0 << 4u)  >> 28u); p1 = p1 + x6 * f32(i32(w1 << 4u)  >> 28u);
    p0 = p0 + x7 * f32(i32(w0)        >> 28u); p1 = p1 + x7 * f32(i32(w1)        >> 28u);
    
    acc0 = acc0 + p0 * sc0;
    acc1 = acc1 + p1 * sc1;
  }

  let ssum0 = subgroupAdd(acc0); let ssum1 = subgroupAdd(acc1);
  if (sgid == 0u) { partSum[tid / sgsz] = ssum0; partSum[32u + tid / sgsz] = ssum1; }
  workgroupBarrier();

  if (tid == 0u) {
    let nsg = (64u + sgsz - 1u) / sgsz; 
    var o0 = 0.0; var o1 = 0.0;
    for (var i = 0u; i < nsg; i = i + 1u) { o0 = o0 + partSum[i]; o1 = o1 + partSum[32u + i]; }
    
    o0 = o0 + bias[n0];
    o1 = o1 + bias[n1];

    if (isQ || isK) {
      let off = m.pos * m.headDim + rope_j;
      let c = cosT[off]; let s = sinT[off];
      let rl = fma(o0, c, 0.0) + fma(-o1, s, 0.0);
      let rh = fma(o1, c, 0.0) + fma(o0, s, 0.0);
      o0 = rl; o1 = rh;
    }

    if (isQ) { qOut[out_idx0] = o0; qOut[out_idx1] = o1; }
    else if (isK) { kOut[out_idx0] = o0; kOut[out_idx1] = o1; }
    else { vOut[out_idx0] = o0; vOut[out_idx1] = o1; }
  }
}`;

// src/qwgpu/model_schema.js
var arrEq = /* @__PURE__ */ __name((a, b) => a.length === b.length && a.every((v, i) => v === b[i]), "arrEq");
function projDesc(layer, subpath, outDim, inDim, { bias = false } = {}) {
  const name = `model.layers.${layer}.${subpath}.weight`;
  const m = subpath.match(/^(self_attn|mlp)\.(.+)$/);
  const loraKey = `layers.${layer}.${m[1]}.${m[2]}`;
  return {
    name,
    role: "projection",
    quant: "int4",
    shape: [outDim, inDim],
    loraKey,
    biasName: bias ? name.replace(/\.weight$/, ".bias") : null
  };
}
__name(projDesc, "projDesc");
function f32Desc(name, shape, role = "f32") {
  return { name, role, quant: "f32", shape };
}
__name(f32Desc, "f32Desc");
function createQwenSchema(cfg) {
  if (!cfg.tieWordEmbeddings && cfg.tieWordEmbeddings !== void 0) {
    throw new Error("QwenWGPU currently requires tied input/output embeddings");
  }
  const H = cfg.hiddenSize;
  const QD = cfg.numHeads * cfg.headDim;
  const KVD = cfg.numKVHeads * cfg.headDim;
  const I = cfg.intermediateSize;
  const tensors = [];
  const layers = [];
  const add = /* @__PURE__ */ __name((d) => {
    tensors.push(d);
    return d;
  }, "add");
  const embed = add({ name: "model.embed_tokens.weight", role: "embedding", quant: "int8", shape: [cfg.vocabSize, H] });
  const finalNorm = add(f32Desc("model.norm.weight", [H], "final_norm"));
  for (let i = 0; i < cfg.numLayers; i++) {
    const p = `model.layers.${i}`;
    const layer = {
      index: i,
      inputNorm: add(f32Desc(`${p}.input_layernorm.weight`, [H], "input_norm")),
      postAttentionNorm: add(f32Desc(`${p}.post_attention_layernorm.weight`, [H], "post_attention_norm")),
      projections: {},
      biases: {}
    };
    layer.projections.q = add(projDesc(i, "self_attn.q_proj", QD, H, { bias: !!cfg.attentionBias }));
    layer.projections.k = add(projDesc(i, "self_attn.k_proj", KVD, H, { bias: !!cfg.attentionBias }));
    layer.projections.v = add(projDesc(i, "self_attn.v_proj", KVD, H, { bias: !!cfg.attentionBias }));
    layer.projections.o = add(projDesc(i, "self_attn.o_proj", H, QD));
    layer.projections.gate = add(projDesc(i, "mlp.gate_proj", I, H));
    layer.projections.up = add(projDesc(i, "mlp.up_proj", I, H));
    layer.projections.down = add(projDesc(i, "mlp.down_proj", H, I));
    for (const key of ["q", "k", "v"]) {
      const proj = layer.projections[key];
      if (proj.biasName) {
        const bias = add(f32Desc(proj.biasName, [proj.shape[0]], `${key}_bias`));
        layer.biases[key] = bias;
      }
    }
    layers.push(layer);
  }
  const byName = new Map(tensors.map((t) => [t.name, t]));
  const expectedNames = new Set(byName.keys());
  return {
    cfg,
    tensors,
    byName,
    expectedNames,
    layers,
    embed,
    finalNorm,
    projectionDescs: tensors.filter((t) => t.role === "projection"),
    validateTensor(name, shape) {
      const desc = byName.get(name);
      if (!desc) return null;
      if (!arrEq(shape, desc.shape)) {
        throw new Error(`shape mismatch for ${name}: got [${shape.join(",")}], expected [${desc.shape.join(",")}]`);
      }
      return desc;
    },
    assertComplete(seen) {
      const missing = [];
      for (const name of expectedNames) if (!seen.has(name)) missing.push(name);
      if (missing.length) {
        const sample = missing.slice(0, 12).join(", ");
        throw new Error(`missing ${missing.length} required tensor(s): ${sample}${missing.length > 12 ? ", \u2026" : ""}`);
      }
    }
  };
}
__name(createQwenSchema, "createQwenSchema");
function moduleKeyFromTensorName(name) {
  const m = name.match(/layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+?)(_proj)?\.(lora_[ABab])/i);
  if (!m) return null;
  return `layers.${m[1]}.${m[2]}.${m[3].replace(/_proj$/, "")}_proj`;
}
__name(moduleKeyFromTensorName, "moduleKeyFromTensorName");

// src/qwgpu/dispatch_plan.js
function createDispatchPlan(schema) {
  return {
    embed: schema.embed,
    finalNorm: schema.finalNorm,
    layers: schema.layers.map((layer) => ({
      index: layer.index,
      inputNorm: layer.inputNorm.name,
      postAttentionNorm: layer.postAttentionNorm.name,
      q: {
        weight: layer.projections.q.name,
        bias: layer.biases.q?.name || null,
        loraKey: layer.projections.q.loraKey
      },
      k: {
        weight: layer.projections.k.name,
        bias: layer.biases.k?.name || null,
        loraKey: layer.projections.k.loraKey
      },
      v: {
        weight: layer.projections.v.name,
        bias: layer.biases.v?.name || null,
        loraKey: layer.projections.v.loraKey
      },
      o: {
        weight: layer.projections.o.name,
        bias: null,
        loraKey: layer.projections.o.loraKey
      },
      gate: {
        weight: layer.projections.gate.name,
        bias: null,
        loraKey: layer.projections.gate.loraKey
      },
      up: {
        weight: layer.projections.up.name,
        bias: null,
        loraKey: layer.projections.up.loraKey
      },
      down: {
        weight: layer.projections.down.name,
        bias: null,
        loraKey: layer.projections.down.loraKey
      }
    }))
  };
}
__name(createDispatchPlan, "createDispatchPlan");

// src/qwgpu/safetensors_loader.js
function decodeBf16ToF32(u8, numel) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, numel);
  const out = new Float32Array(numel);
  const o32 = new Uint32Array(out.buffer);
  for (let i = 0; i < numel; i++) o32[i] = u16[i] << 16;
  return out;
}
__name(decodeBf16ToF32, "decodeBf16ToF32");
function decodeF16ToF32(u8, numel) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, numel);
  const out = new Float32Array(numel);
  for (let i = 0; i < numel; i++) {
    const h = u16[i], s = (h & 32768) >> 15, e = (h & 31744) >> 10, f = h & 1023;
    if (e === 0) out[i] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    else if (e === 31) out[i] = f ? NaN : s ? -Infinity : Infinity;
    else out[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}
__name(decodeF16ToF32, "decodeF16ToF32");
function decodeF32(u8, numel) {
  return new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + numel * 4));
}
__name(decodeF32, "decodeF32");
var DECODERS = {
  BF16: decodeBf16ToF32,
  F16: decodeF16ToF32,
  FP16: decodeF16ToF32,
  F32: decodeF32,
  FP32: decodeF32
};
async function loadIndex(reader) {
  try {
    const idx = JSON.parse(await reader.text("model.safetensors.index.json"));
    return { weightMap: idx.weight_map || {}, shards: [...new Set(Object.values(idx.weight_map || {}))] };
  } catch {
    return { weightMap: null, shards: ["model.safetensors"] };
  }
}
__name(loadIndex, "loadIndex");
function shardPlan(shards, weightMap, names) {
  if (!weightMap || !names) return new Map(shards.map((shard) => [shard, null]));
  const plan = /* @__PURE__ */ new Map();
  for (const name of names) {
    const shard = weightMap[name];
    if (!shard) continue;
    if (!plan.has(shard)) plan.set(shard, /* @__PURE__ */ new Set());
    plan.get(shard).add(name);
  }
  return plan;
}
__name(shardPlan, "shardPlan");
async function streamSafetensors(source, { names = null, onTensor, onProgress = /* @__PURE__ */ __name(() => {
}, "onProgress") } = {}) {
  if (!onTensor) throw new Error("streamSafetensors requires onTensor");
  const reader = typeof source === "string" ? urlReader(source) : source;
  const { weightMap, shards } = await loadIndex(reader);
  const plan = shardPlan(shards, weightMap, names);
  let visited = 0;
  const total = names?.size || 0;
  for (const [shard, wantedInShard] of plan) {
    const lenBuf = await reader.range(shard, 0, 8);
    const headerLen = Number(new DataView(lenBuf).getBigUint64(0, true));
    const hdrBuf = await reader.range(shard, 8, 8 + headerLen);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(hdrBuf)));
    const dataStart = 8 + headerLen;
    const allNames = Object.keys(header).filter((k) => k !== "__metadata__");
    const tensorNames = wantedInShard ? allNames.filter((n) => wantedInShard.has(n)) : names ? allNames.filter((n) => names.has(n)) : allNames;
    for (const name of tensorNames) {
      const t = header[name];
      if (!t) continue;
      const dtype = String(t.dtype || "").toUpperCase();
      const dec = DECODERS[dtype];
      if (!dec) throw new Error(`unsupported dtype ${dtype} for ${name}`);
      const numel = t.shape.reduce((a, b) => a * b, 1);
      const [s, e] = t.data_offsets;
      const buf = await reader.range(shard, dataStart + s, dataStart + e);
      const data = dec(new Uint8Array(buf), numel);
      await onTensor({ name, shape: t.shape, dtype, data, shard });
      visited++;
      onProgress(name, total ? Math.min(0.95, visited / total) : 0.3);
    }
  }
}
__name(streamSafetensors, "streamSafetensors");

// src/qwgpu/quantize.js
function quantizeInt8RowMajor(f322, outDim, inDim) {
  const scale = new Float32Array(outDim);
  const q = new Int8Array(outDim * inDim);
  for (let o = 0; o < outDim; o++) {
    const base = o * inDim;
    let amax = 0;
    for (let i = 0; i < inDim; i++) {
      const a = Math.abs(f322[base + i]);
      if (a > amax) amax = a;
    }
    const s = amax > 0 ? amax / 127 : 1;
    scale[o] = s;
    const inv = 1 / s;
    for (let i = 0; i < inDim; i++) {
      let v = Math.round(f322[base + i] * inv);
      if (v > 127) v = 127;
      else if (v < -128) v = -128;
      q[base + i] = v;
    }
  }
  const packed = new Uint32Array(outDim * inDim / 4);
  const u8 = new Uint8Array(q.buffer);
  for (let w = 0; w < packed.length; w++) {
    packed[w] = u8[w * 4] | u8[w * 4 + 1] << 8 | u8[w * 4 + 2] << 16 | u8[w * 4 + 3] << 24;
  }
  return { packed, scale, outDim, inDim };
}
__name(quantizeInt8RowMajor, "quantizeInt8RowMajor");
function quantizeInt4Group(f322, outDim, inDim, group = 128) {
  const groupsPerRow = inDim / group;
  const scale = new Float32Array(outDim * groupsPerRow);
  const q = new Int8Array(outDim * inDim);
  for (let o = 0; o < outDim; o++) {
    for (let g = 0; g < groupsPerRow; g++) {
      const base = o * inDim + g * group;
      let amax = 0;
      for (let i = 0; i < group; i++) {
        const a = Math.abs(f322[base + i]);
        if (a > amax) amax = a;
      }
      const s = amax > 0 ? amax / 7 : 1;
      scale[o * groupsPerRow + g] = s;
      const inv = 1 / s;
      for (let i = 0; i < group; i++) {
        let v = Math.round(f322[base + i] * inv);
        if (v > 7) v = 7;
        else if (v < -8) v = -8;
        q[base + i] = v;
      }
    }
  }
  const packed = new Uint32Array(outDim * inDim / 8);
  for (let w = 0; w < packed.length; w++) {
    let acc = 0;
    for (let j = 0; j < 8; j++) acc |= (q[w * 8 + j] & 15) << j * 4;
    packed[w] = acc >>> 0;
  }
  return { packed, scale, groupsPerRow };
}
__name(quantizeInt4Group, "quantizeInt4Group");

// src/qwgpu/model_uploader.js
var ModelUploader = class {
  static {
    __name(this, "ModelUploader");
  }
  constructor({ schema, q, q4, bufs, uploadF32, uploadU32, groupSize = 128 }) {
    this.schema = schema;
    this.q = q;
    this.q4 = q4;
    this.bufs = bufs;
    this.uploadF32 = uploadF32;
    this.uploadU32 = uploadU32;
    this.groupSize = groupSize;
    this.seen = /* @__PURE__ */ new Set();
  }
  visit({ name, shape, data }) {
    const desc = this.schema.validateTensor(name, shape);
    if (!desc) return;
    if (this.seen.has(name)) throw new Error(`duplicate tensor ${name}`);
    if (desc.quant === "int8") {
      const { packed, scale } = quantizeInt8RowMajor(data, shape[0], shape[1]);
      this.q[name] = { w: this.uploadU32(packed), scale: this.uploadF32(scale), N: shape[0], K: shape[1] };
    } else if (desc.quant === "int4") {
      const { packed, scale, groupsPerRow } = quantizeInt4Group(data, shape[0], shape[1], this.groupSize);
      this.q4[name] = {
        w: this.uploadU32(packed),
        scale: this.uploadF32(scale),
        N: shape[0],
        K: shape[1],
        gpr: groupsPerRow,
        desc
      };
    } else if (desc.quant === "f32") {
      this.bufs[name] = this.uploadF32(data);
    } else {
      throw new Error(`unsupported quant mode ${desc.quant} for ${name}`);
    }
    this.seen.add(name);
  }
  finalize() {
    this.schema.assertComplete(this.seen);
  }
};

// src/qwgpu/buffer_pool.js
var GPUBufferPool = class {
  static {
    __name(this, "GPUBufferPool");
  }
  constructor(device, { cacheBindGroups = true } = {}) {
    this.dev = device;
    this.cacheBindGroups = cacheBindGroups;
    this.uniformPool = [];
    this.uniformIdx = 0;
    this.staticUniforms = /* @__PURE__ */ new Map();
    this.bindGroups = /* @__PURE__ */ new Map();
    this.sensitiveBindGroups = /* @__PURE__ */ new Set();
    this.bufferIds = /* @__PURE__ */ new WeakMap();
    this.pipelineIds = /* @__PURE__ */ new WeakMap();
    this.nextBufferId = 1;
    this.nextPipelineId = 1;
    this._stats = this._emptyStats();
  }
  /*
   * TECHNIQUE: Bind group caching (opt-in per call site)
   *   Frequently reused (pipeline + buffer set) combinations are stored in a Map.
   *   Avoids repeated GPU bind group creation on the hot GEMV / attention paths.
   *   Sensitive / one-shot groups are deliberately not cached.
   */
  _emptyStats() {
    return {
      buffersCreated: 0,
      dynamicUniformWrites: 0,
      staticUniformHits: 0,
      staticUniformMisses: 0,
      bindGroupHits: 0,
      bindGroupMisses: 0,
      uncachedBindGroups: 0
    };
  }
  resetStats() {
    this._stats = this._emptyStats();
  }
  stats() {
    return {
      ...this._stats,
      uniformPoolSize: this.uniformPool.length,
      staticUniforms: this.staticUniforms.size,
      bindGroups: this.bindGroups.size
    };
  }
  buffer(size, usage) {
    this._stats.buffersCreated++;
    return this.dev.createBuffer({ size, usage });
  }
  uploadF32(arr, usage) {
    const b = this.buffer(arr.byteLength, usage);
    this.dev.queue.writeBuffer(b, 0, arr);
    return b;
  }
  uploadU32(arr, usage) {
    const b = this.buffer(arr.byteLength, usage);
    this.dev.queue.writeBuffer(b, 0, arr);
    return b;
  }
  dynamicUniform(arr, usage) {
    let b = this.uniformPool[this.uniformIdx];
    if (!b) {
      b = this.buffer(32, usage);
      this.uniformPool[this.uniformIdx] = b;
    }
    this.uniformIdx++;
    this._stats.dynamicUniformWrites++;
    this.dev.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return b;
  }
  resetUniforms() {
    this.uniformIdx = 0;
  }
  staticUniform(key, arr, usage) {
    let b = this.staticUniforms.get(key);
    if (!b) {
      this._stats.staticUniformMisses++;
      b = this.buffer(32, usage);
      this.dev.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
      this.staticUniforms.set(key, b);
    } else this._stats.staticUniformHits++;
    return b;
  }
  idForBuffer(buffer) {
    let id = this.bufferIds.get(buffer);
    if (!id) {
      id = this.nextBufferId++;
      this.bufferIds.set(buffer, id);
    }
    return id;
  }
  idForPipeline(pipe) {
    let id = this.pipelineIds.get(pipe);
    if (!id) {
      id = this.nextPipelineId++;
      this.pipelineIds.set(pipe, id);
    }
    return id;
  }
  uncachedBindGroup(pipe, buffers) {
    this._stats.uncachedBindGroups++;
    return this.dev.createBindGroup({
      label: pipe.__name ? `${pipe.__name}:bg:${buffers.length}` : void 0,
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } }))
    });
  }
  cachedBindGroup(pipe, buffers, key, { sensitive = false } = {}) {
    if (!this.cacheBindGroups || !key) return this.uncachedBindGroup(pipe, buffers);
    const fullKey = `${this.idForPipeline(pipe)}:${key}:${buffers.map((b) => this.idForBuffer(b)).join(",")}`;
    let bg = this.bindGroups.get(fullKey);
    if (!bg) {
      this._stats.bindGroupMisses++;
      bg = this.uncachedBindGroup(pipe, buffers);
      this.bindGroups.set(fullKey, bg);
      if (sensitive) this.sensitiveBindGroups.add(fullKey);
    } else this._stats.bindGroupHits++;
    return bg;
  }
  clearSensitiveBindGroups() {
    for (const key of this.sensitiveBindGroups) this.bindGroups.delete(key);
    this.sensitiveBindGroups.clear();
  }
};

// src/qwgpu/runtime.js
var STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
var UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
var QwenWGPU = class {
  static {
    __name(this, "QwenWGPU");
  }
  // opts: { maxCtx, maxPrefillT, decodeBatchSize, samplingTopK } — context
  // window + batched-prefill cap (default 8192 each; KV cache grows linearly).
  constructor(device, cfg, opts = {}) {
    this.dev = device;
    this.cfg = cfg;
    this.lora = null;
    this.bufs = {};
    this.opts = opts;
    this.features = this._normalizeFeatures(opts);
    this.pool = new GPUBufferPool(device, { cacheBindGroups: opts.cacheBindGroups !== false });
    this._loraEpoch = 0;
    this.lastDispatchCount = 0;
    this.packedBytes = 0;
    this.workgroupAutotunePromise = null;
    this._argmaxReadBusy = false;
    this._topKReadBusy = false;
  }
  _normalizeFeatures(opts = {}) {
    const prefillAttention = opts.prefillAttention || "block";
    if (!["row", "block"].includes(prefillAttention))
      throw new Error(`unsupported prefillAttention ${prefillAttention}`);
    return {
      // fuseRMSNormQKVRoPE: fused RMSNorm + int4 QKV GEMV + RoPE for no-LoRA decode
      // (one workgroup per (head,rot) pair; verified logitDiff 0 vs PyTorch ref).
      // fuseQKV selects the alternate qkvGemv4 path and stays OFF by default since
      // the fused-RMS path already covers the fast no-LoRA decode; LoRA layers are
      // routed to the unfused gemv4x3 + ropeQK path automatically (see step()).
      fuseQKV: opts.fuseQKV === true,
      fuseRoPE: opts.fuseRoPE !== false,
      fuseMLP: opts.fuseMLP !== false,
      fuseResidual: opts.fuseResidual !== false,
      prefillAttention,
      prefillChunkSize: Math.max(0, opts.prefillChunkSize || 0),
      actQuant: !!opts.actQuant,
      // Default OFF: the GEMV4_QKV_ROPE_RMS kernel still computes zero outputs even
      // with the corrected (totalPairs) dispatch — there is a deeper bug in the
      // fused kernel itself. The unfused gemv4x3 + ropeQK decode is verified
      // logitDiff 0 vs the PyTorch ref, so it stays the default until the fused
      // kernel is debugged. The wrapper dispatch is now correct for that work.
      fuseRMSNormQKVRoPE: opts.fuseRMSNormQKVRoPE === true,
      pagedAttention: !!opts.pagedAttention
    };
  }
  setFeatureFlags(flags = {}) {
    this.features = this._normalizeFeatures({ ...this.features, ...flags });
    this.pool.clearSensitiveBindGroups();
  }
  featureFlags() {
    return { ...this.features };
  }
  // Phase 3 (f16): when shader-f16 is available we can switch hot kernels to f16
  // storage/compute for bandwidth wins. Stub for now; real kernel variants + selection
  // will be added. Evaluation: compare f16 vs f32 logits within tolerance + bench speedup.
  hasF16Compute() {
    return !!this.hasF16;
  }
  setUseF16(v) {
    this._useF16 = !!v && this.hasF16Compute();
  }
  usingF16() {
    return !!this._useF16;
  }
  // Phase 4: allow caller / autotuner to override workgroup size after build if desired.
  // Note: affects *future* pipes / re-pipes; existing pipes keep their specialization.
  setWorkgroupSize(wg) {
    if (wg && wg > 0) this.workgroupSize = wg | 0;
  }
  // Basic load-time / on-demand workgroup autotuner (Phase 4).
  // Tries a few WG sizes for simple override-supporting kernels (add / rms for now).
  // Uses wall time + onSubmittedWorkDone for broad compatibility.
  // Returns a map of best sizes; optionally hot-swaps the pipe for 'add'.
  async autotuneWorkgroups(opts = {}) {
    const iters = opts.iters || 6;
    const cands = opts.candidates || [32, 64, 128, 256];
    const results = {};
    const useTS = this.hasTimestampQuery;
    const timeKernel = /* @__PURE__ */ __name(async (spec, pipe, label) => {
      const n = spec.n;
      const a = this._buf(n * 4);
      const g = this._buf(n * 4);
      const y = this._buf(n * 4);
      const buffers = spec.buffers(a, y, g);
      const imm = spec.imm(n);
      let gpuMs = 0;
      let usedGPU = false;
      if (useTS) {
        const qs = this.dev.createQuerySet({ type: "timestamp", count: 2 });
        const resolveBuf = this._buf(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
        const readBuf = this._buf(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const tWall0 = typeof performance !== "undefined" ? performance.now() : Date.now();
        for (let i = 0; i < iters; i++) {
          const enc = this.dev.createCommandEncoder();
          const bg = this._bg(pipe, buffers);
          const p = enc.beginComputePass({
            timestampWrites: {
              querySet: qs,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1
            }
          });
          p.setPipeline(pipe);
          if (bg) p.setBindGroup(0, bg);
          if (imm) p.setImmediates(0, imm);
          p.dispatchWorkgroups(Math.ceil(n / (pipe.__wg || 256)), 1);
          p.end();
          enc.resolveQuerySet(qs, 0, 2, resolveBuf, 0);
          enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 16);
          this.dev.queue.submit([enc.finish()]);
          if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
          await readBuf.mapAsync(GPUMapMode.READ);
          const t = new BigInt64Array(readBuf.getMappedRange());
          const us = Number(t[1] - t[0]) / 1e3;
          gpuMs += us;
          readBuf.unmap();
        }
        const wallMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - tWall0;
        resolveBuf.destroy?.();
        readBuf.destroy?.();
        qs.destroy?.();
        usedGPU = true;
        a.destroy?.();
        g.destroy?.();
        y.destroy?.();
        return gpuMs / iters / 1e3;
      }
      const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
      for (let i = 0; i < iters; i++) {
        const enc = this.dev.createCommandEncoder();
        const bg = this._bg(pipe, buffers);
        this._dispatch(enc, pipe, bg, Math.ceil(n / (pipe.__wg || 256)), 1, label + ":bench", imm);
        this.dev.queue.submit([enc.finish()]);
        if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
      }
      const ms = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
      a.destroy?.();
      g.destroy?.();
      y.destroy?.();
      return ms / iters;
    }, "timeKernel");
    const kernels = [
      { name: "add", src: ADD, n: 8192, buffers: /* @__PURE__ */ __name((a, y) => [a, y], "buffers"), imm: /* @__PURE__ */ __name((n) => new Uint32Array([n]), "imm") },
      { name: "rms", src: RMSNORM, n: 4096, buffers: /* @__PURE__ */ __name((a, y, g) => [a, g, y], "buffers"), imm: /* @__PURE__ */ __name((n) => new Float32Array([n, this.cfg.rmsNormEps]), "imm") },
      { name: "silu", src: SILUMUL, n: 8192, buffers: /* @__PURE__ */ __name((a, y) => [a, y], "buffers"), imm: /* @__PURE__ */ __name((n) => new Uint32Array([n]), "imm") }
    ];
    for (const k of kernels) {
      try {
        let best = { wg: 256, ms: Infinity };
        for (const wg of cands) {
          const p = this._pipe(k.src, `${k.name}:autotune:${wg}`, { WG: wg });
          p.__wg = wg;
          const ms = await timeKernel(k, p, `${k.name}${wg}`);
          results[`${k.name}:${wg}`] = ms;
          if (ms < best.ms) best = { wg, ms };
        }
        results[`best${k.name[0].toUpperCase()}${k.name.slice(1)}`] = best;
        if (opts.apply && this.pipes[k.name]) {
          this.pipes[k.name] = this._pipe(k.src, k.name, { WG: best.wg });
          this.pipes[k.name].__wg = best.wg;
        }
      } catch (e) {
        results[`${k.name}Error`] = String(e);
      }
    }
    this.bestWorkgroupSizes = {
      add: results.bestAdd?.wg,
      rms: results.bestRms?.wg,
      silu: results.bestSilu?.wg,
      source: useTS ? "gpu-ts" : "wall"
    };
    console.log("[autotune] WG microbench results (ms/iter, source=" + (useTS ? "gpu-ts" : "wall") + "):", results);
    return results;
  }
  _buf(size, usage = STORAGE) {
    return this.pool.buffer(size, usage);
  }
  _f32(arr, usage = STORAGE) {
    return this.pool.uploadF32(arr, usage);
  }
  _u32(arr) {
    return this.pool.uploadU32(arr, STORAGE);
  }
  _uni(arr) {
    return this.pool.dynamicUniform(arr, UNIFORM);
  }
  _staticUni(key, arr) {
    return this.pool.staticUniform(key, arr, UNIFORM);
  }
  _resetUni() {
    this.pool.resetUniforms();
    this.lastDispatchCount = 0;
  }
  _pipe(code, name, overrides = null) {
    const processedCode = typeof code === "string" ? code.replaceAll("WG_SIZE", this.workgroupSize || 64) : code;
    const m = this.dev.createShaderModule({
      label: name || void 0,
      code: processedCode
    });
    const comp = { module: m, entryPoint: "main" };
    if (overrides && typeof overrides === "object") comp.constants = overrides;
    const pipe = this.dev.createComputePipeline({
      label: name ? `${name}-pipeline` : void 0,
      layout: "auto",
      compute: comp
    });
    if (overrides?.WG) pipe.__wg = overrides.WG;
    if (name) pipe.__name = name;
    return pipe;
  }
  /*
   * TECHNIQUE: Specialization via pipeline constants (overrides)
   *   Workgroup size and other small values are passed as pipeline-overridable
   *   constants instead of uniforms or JS branches. Allows the shader compiler
   *   to specialize the binary (better than runtime if).
   */
  // `source` is a base URL string OR a reader { range, text } (e.g. hfReader/fileReader).
  async build(source, onProgress = () => {
  }) {
    const shaderCompileStart = performance.now();
    const dev = this.dev, c = this.cfg;
    this.CHUNK = 128;
    this._initRuntimeOptions();
    this.maxCtx = this.opts.maxCtx || 8192;
    this.maxPrefillT = Math.min(this.opts.maxPrefillT || 8192, this.maxCtx);
    const isAppleSilicon = this.dev.limits.minStorageBufferOffsetAlignment === 4;
    const isIntelArc = this.dev.limits.minStorageBufferOffsetAlignment === 256;
    this.workgroupSize = isAppleSilicon || isIntelArc ? 32 : 64;
    onProgress && onProgress(`workgroup size chosen: ${this.workgroupSize} (apple/intel bias toward 32)`, 0);
    let hasDP4a = false;
    if (typeof navigator !== "undefined" && navigator.gpu?.wgslLanguageFeatures?.has?.("packed_4x8_integer_dot_product")) {
      dev.pushErrorScope("validation");
      try {
        dev.createShaderModule({
          code: `enable packed_4x8_integer_dot_product; @compute @workgroup_size(1) fn main() {}`
        });
        const error = await dev.popErrorScope();
        if (!error) {
          hasDP4a = true;
        }
      } catch (e) {
        await dev.popErrorScope();
      }
    }
    this.hasDP4a = hasDP4a;
    const hasF16 = this.dev.features.has("shader-f16");
    this.hasF16 = hasF16;
    this.hasTimestampQuery = this.dev.features.has("timestamp-query");
    this.pam = new PagedAttentionManager(this.maxCtx);
    this.pipes = {
      gemv: this._pipe(GEMV, "gemv"),
      loraA: this._pipe(LORA_A, "loraA"),
      loraABatch: this._pipe(LORA_A_BATCH, "loraABatch"),
      loraBAdd: this._pipe(LORA_B_ADD, "loraBAdd"),
      loraBAddT: this._pipe(LORA_B_ADD_T, "loraBAddT"),
      rms: this._pipe(RMSNORM, "rms", { WG: this.workgroupSize || 256 }),
      rmsF16: hasF16 ? this._pipe(RMSNORM_F16, "rmsF16", { WG: this.workgroupSize || 256 }) : null,
      rope: this._pipe(ROPE, "rope"),
      ropeF16: hasF16 ? this._pipe(ROPE_F16, "ropeF16") : null,
      ropeQK: this._pipe(ROPE_QK, "ropeQK"),
      ropeQKF16: hasF16 ? this._pipe(ROPE_QK_F16, "ropeQKF16") : null,
      ropeT: this._pipe(ROPE_T, "ropeT"),
      ropeTF16: hasF16 ? this._pipe(ROPE_T_F16, "ropeTF16") : null,
      attnP: this._pipe(ATTN_PARTIAL, "attnP", { WG: 128 }),
      attnPF16: hasF16 ? this._pipe(ATTN_PARTIAL_F16, "attnPF16", { WG: 128 }) : null,
      attnC: this._pipe(ATTN_COMBINE, "attnC", { WG: 128 }),
      attnCF16: hasF16 ? this._pipe(ATTN_COMBINE_F16, "attnCF16", { WG: 128 }) : null,
      add: this._pipe(ADD, "add", { WG: this.workgroupSize || 256 }),
      silu: this._pipe(SILUMUL, "silu", { WG: this.workgroupSize || 256 }),
      addF16: hasF16 ? this._pipe(ADD_F16, "addF16", { WG: this.workgroupSize || 256 }) : null,
      siluF16: hasF16 ? this._pipe(SILUMUL_F16, "siluF16", { WG: this.workgroupSize || 256 }) : null,
      embed: this._pipe(EMBED, "embed"),
      embedBuf: this._pipe(EMBED_BUF, "embedBuf"),
      argmax: this._pipe(ARGMAX, "argmax"),
      gemv4: this._pipe(GEMV4, "gemv4"),
      gemv4Add: this._pipe(GEMV4_ADD, "gemv4Add"),
      qkvGemv4: this._pipe(QKV_GEMV4, "qkvGemv4"),
      gateUpSiluGemv4: this._pipe(GATE_UP_SILU_GEMV4, "gateUpSiluGemv4"),
      topkSelect: this._pipe(TOPK_SELECT, "topkSelect"),
      sampleTopK: this._pipe(SAMPLE_TOPK, "sampleTopK"),
      gemm4: this._pipe(GEMM4, "gemm4"),
      gemm4AddT: this._pipe(GEMM4_ADD_T, "gemm4AddT"),
      rmsT: this._pipe(RMSNORM_T, "rmsT", { WG: this.workgroupSize || 256 }),
      rmsTF16: hasF16 ? this._pipe(RMSNORM_T_F16, "rmsTF16", { WG: this.workgroupSize || 256 }) : null,
      embedT: this._pipe(EMBED_T, "embedT"),
      attnPrefill: this._pipe(ATTN_PREFILL, "attnPrefill"),
      attnPrefillBlock: this._pipe(ATTN_PREFILL_BLOCK, "attnPrefillBlock"),
      dynQuant: this._pipe(DYN_QUANT_X, "dynQuant"),
      dynQuantT: this._pipe(DYN_QUANT_X_T, "dynQuantT"),
      gemv4W4A8: this._pipe(GEMV4_W4A8(hasDP4a, this.workgroupSize), "gemv4W4A8"),
      gemv4AddW4A8: this._pipe(GEMV4_ADD_W4A8(hasDP4a, this.workgroupSize), "gemv4AddW4A8"),
      qkvGemv4W4A8: this._pipe(QKV_GEMV4_W4A8(hasDP4a, this.workgroupSize), "qkvGemv4W4A8"),
      gateUpSiluGemv4W4A8: this._pipe(GATE_UP_SILU_GEMV4_W4A8(hasDP4a, this.workgroupSize), "gateUpSiluGemv4W4A8"),
      gemm4W4A8: this._pipe(GEMM4_W4A8(hasDP4a), "gemm4W4A8"),
      gemm4AddTW4A8: this._pipe(GEMM4_ADD_T_W4A8(hasDP4a), "gemm4AddTW4A8"),
      rmsNormQkvRope: this._pipe(GEMV4_QKV_ROPE_RMS, "rmsNormQkvRope"),
      writeKvPage: this._pipe(WRITE_KV_PAGE, "writeKvPage"),
      writeKvPageBatch: this._pipe(WRITE_KV_PAGE_BATCH, "writeKvPageBatch"),
      attnPartialPaged: this._pipe(ATTN_PARTIAL_PAGED, "attnPartialPaged"),
      attnPrefillPaged: this._pipe(ATTN_PREFILL_PAGED, "attnPrefillPaged"),
      attnPrefillBlockPaged: this._pipe(ATTN_PREFILL_BLOCK_PAGED, "attnPrefillBlockPaged")
    };
    this.shaderCompileMs = performance.now() - shaderCompileStart;
    if (hasF16) {
      this.setUseF16(true);
      onProgress("f16 compute enabled (add/silu/rms/rope/attn-partial/combine paths)", 0);
    }
    if (this.hasTimestampQuery) {
      onProgress("timestamp-query available (precise GPU timing + autotune)", 0);
    }
    onProgress("streaming + quantizing weights", 0);
    this.schema = createQwenSchema(c);
    this.plan = createDispatchPlan(this.schema);
    this.q = {};
    this.q4 = {};
    this.qkv = [];
    this.gateUp = [];
    const uploader = new ModelUploader({
      schema: this.schema,
      q: this.q,
      q4: this.q4,
      bufs: this.bufs,
      uploadF32: /* @__PURE__ */ __name((arr) => this._f32(arr), "uploadF32"),
      uploadU32: /* @__PURE__ */ __name((arr) => this._u32(arr), "uploadU32")
    });
    if (source === "mock") {
      for (const name of this.schema.expectedNames) {
        const desc = this.schema.tensors.find((t) => t.name === name);
        const shape = desc.shape;
        const numel = shape.reduce((a, b) => a * b, 1);
        const type = desc.quant === "int8" ? "I8" : "F32";
        uploader.visit({ name, shape, data: new Uint8Array(numel * (type === "I8" ? 1 : 4)), type });
      }
    } else {
      await streamSafetensors(source, {
        names: this.schema.expectedNames,
        onProgress,
        onTensor: /* @__PURE__ */ __name(async (tensor) => {
          uploader.visit(tensor);
          if (uploader.seen.size % 48 === 0) await new Promise((r) => setTimeout(r, 0));
        }, "onTensor")
      });
    }
    uploader.finalize();
    await this._buildPackedProjectionBuffers();
    this._buildRope(this.maxCtx);
    this.kc = [], this.vc = [];
    const kvSize = c.numKVHeads * this.maxCtx * c.headDim * 4;
    for (let i = 0; i < c.numLayers; i++) {
      this.kc.push(this._buf(kvSize));
      this.vc.push(this._buf(kvSize));
    }
    const H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize;
    const NSPLITMAX = Math.ceil(this.maxCtx / this.CHUNK);
    this.s = {
      hidden: this._buf(H * 4),
      normed: this._buf(H * 4),
      q: this._buf(qd * 4),
      k: this._buf(kvd * 4),
      v: this._buf(kvd * 4),
      attn: this._buf(qd * 4),
      tmp: this._buf(Math.max(qd, I) * 4),
      tmp2: this._buf(I * 4),
      logits: this._buf(c.vocabSize * 4),
      dummy: this._buf(64),
      loraD: this._buf(256 * 4),
      loraD2: this._buf(256 * 4),
      amax: this._buf(4),
      pm: this._buf(c.numHeads * NSPLITMAX * 4),
      pz: this._buf(c.numHeads * NSPLITMAX * 4),
      po: this._buf(c.numHeads * NSPLITMAX * c.headDim * 4),
      idsBuf: this._buf(this.decodeBatchCapacity * 4),
      sampleIds: this._buf(this.maxSamplingTopK * 4),
      sampleVals: this._buf(this.maxSamplingTopK * 4),
      sampled: this._buf(4),
      // single u32 chosen by GPU sampler (Phase 5)
      x_q: this._buf(Math.max(qd, I) * 4),
      scale_x: this._buf(256 * 4),
      blockTableBuf: this._buf(this.pam.maxBlocksPerSeq * 4, STORAGE | GPUBufferUsage.COPY_DST)
    };
    this.idsRead = this._buf(this.decodeBatchCapacity * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.argmaxRead = this._buf(4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sampleIdsRead = this._buf(this.maxSamplingTopK * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sampleValsRead = this._buf(this.maxSamplingTopK * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sampledRead = this._buf(4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.sT = null;
    this.sTcap = 0;
    this._initStaticUniforms();
    if (this.decodeBatchMode === "auto") {
      onProgress("autotuning decode batch", 0.98);
      await this.autotuneDecodeBatch();
    }
    onProgress("ready", 1);
    if (!this._didAutoWG) {
      this._didAutoWG = true;
      this.workgroupAutotunePromise = this.autotuneWorkgroups({ iters: 2, apply: true }).catch((e) => ({
        error: String(e)
      }));
    }
    return this;
  }
  _initRuntimeOptions() {
    const opts = this.opts;
    this.decodeBatchMode = opts.decodeBatchSize === "auto" ? "auto" : "fixed";
    this.decodeBatchCandidates = (opts.decodeBatchCandidates || [1, 2, 4, 8, 16, 32]).map((x) => Math.max(1, Math.floor(Number(x) || 0))).filter(Boolean);
    const requested = opts.decodeBatchSize === void 0 || opts.decodeBatchSize === "auto" ? 16 : Math.max(1, Math.floor(Number(opts.decodeBatchSize)));
    this.maxDecodeBatchSize = Math.max(
      1,
      Math.floor(Number(opts.maxDecodeBatchSize || Math.max(requested, ...this.decodeBatchCandidates, 16)))
    );
    this.decodeBatchCapacity = Math.min(this.maxDecodeBatchSize, Math.max(requested, ...this.decodeBatchCandidates));
    this.MAXBATCH = Math.min(requested, this.decodeBatchCapacity);
    this.decodeBatchWarmupTokens = Math.max(0, Math.floor(Number(opts.decodeBatchWarmupTokens ?? 4)));
    this.decodeBatchWarmupSize = Math.min(
      this.decodeBatchCapacity,
      Math.max(1, Math.floor(Number(opts.decodeBatchWarmupSize ?? 4)))
    );
    this.decodeBatchMaxLatencyMs = Number(opts.decodeBatchMaxLatencyMs ?? 250);
    this.samplingTopK = Math.max(1, Math.floor(Number(opts.samplingTopK ?? 40)));
    this.maxSamplingTopK = Math.max(this.samplingTopK, Math.floor(Number(opts.maxSamplingTopK ?? 64)));
    this.decodeBatchTuning = {
      selected: this.MAXBATCH,
      candidates: [],
      reason: this.decodeBatchMode === "auto" ? "pending" : "fixed"
    };
  }
  _buildRope(maxSeq) {
    const { headDim, ropeTheta } = this.cfg;
    const half = headDim / 2;
    const cos = new Float32Array(maxSeq * headDim), sin = new Float32Array(maxSeq * headDim);
    for (let p = 0; p < maxSeq; p++)
      for (let i = 0; i < half; i++) {
        const a = p / Math.pow(ropeTheta, 2 * i / headDim);
        const cc = Math.cos(a), ss = Math.sin(a);
        cos[p * headDim + i] = cc;
        cos[p * headDim + half + i] = cc;
        sin[p * headDim + i] = ss;
        sin[p * headDim + half + i] = ss;
      }
    this.ropeCos = this._f32(cos);
    this.ropeSin = this._f32(sin);
    this._ropeRow = headDim * 4;
  }
  _initStaticUniforms() {
    const c = this.cfg;
    const rms = new ArrayBuffer(8);
    const rmsDv = new DataView(rms);
    rmsDv.setFloat32(0, c.hiddenSize, true);
    rmsDv.setFloat32(4, c.rmsNormEps, true);
    this.u = {
      rmsHidden: this._staticUni(`rms:${c.hiddenSize}:${c.rmsNormEps}`, new Uint8Array(rms)),
      addHidden: this._staticUni(`u32:${c.hiddenSize}`, new Uint32Array([c.hiddenSize])),
      siluIntermediate: this._staticUni(`u32:${c.intermediateSize}`, new Uint32Array([c.intermediateSize])),
      embedBuf: this._staticUni(`embedBuf:${c.hiddenSize}`, new Uint32Array([c.hiddenSize])),
      argmax: this._staticUni(`argmax:${c.vocabSize}`, new Uint32Array([c.vocabSize]))
    };
  }
  async _buildPackedProjectionBuffers() {
    const enc = this.dev.createCommandEncoder();
    const copy = /* @__PURE__ */ __name((src, dst, dstOffset, bytes) => enc.copyBufferToBuffer(src, 0, dst, dstOffset, bytes), "copy");
    this.packedBytes = 0;
    for (const L of this.plan.layers) {
      const q = this.q4[L.q.weight], k = this.q4[L.k.weight], v = this.q4[L.v.weight];
      if (q.K !== k.K || q.K !== v.K || q.gpr !== k.gpr || q.gpr !== v.gpr)
        throw new Error(`layer ${L.index} qkv packing requires matching K/gpr`);
      const totalN = q.N + k.N + v.N;
      const wBytes = totalN * (q.K / 8) * 4;
      const scaleBytes = totalN * q.gpr * 4;
      const biasBytes = totalN * 4;
      const w = this._buf(wBytes);
      const scale = this._buf(scaleBytes);
      const bias = this._buf(biasBytes);
      enc.clearBuffer(bias);
      let wOff = 0, sOff = 0, bOff = 0;
      for (const part of [L.q, L.k, L.v]) {
        const qq = this.q4[part.weight];
        const rowsW = qq.N * (qq.K / 8) * 4;
        const rowsS = qq.N * qq.gpr * 4;
        copy(qq.w, w, wOff, rowsW);
        wOff += rowsW;
        copy(qq.scale, scale, sOff, rowsS);
        sOff += rowsS;
        if (part.bias) copy(this.bufs[part.bias], bias, bOff, qq.N * 4);
        bOff += qq.N * 4;
      }
      this.qkv[L.index] = { w, scale, bias, K: q.K, qN: q.N, kN: k.N, vN: v.N, totalN, gpr: q.gpr };
      this.packedBytes += wBytes + scaleBytes + biasBytes;
      const gate = this.q4[L.gate.weight], up = this.q4[L.up.weight];
      if (gate.K !== up.K || gate.N !== up.N || gate.gpr !== up.gpr)
        throw new Error(`layer ${L.index} gate/up packing requires matching shape`);
      const guWBytes = (gate.N + up.N) * (gate.K / 8) * 4;
      const guScaleBytes = (gate.N + up.N) * gate.gpr * 4;
      const guW = this._buf(guWBytes);
      const guScale = this._buf(guScaleBytes);
      copy(gate.w, guW, 0, gate.N * (gate.K / 8) * 4);
      copy(up.w, guW, gate.N * (gate.K / 8) * 4, up.N * (up.K / 8) * 4);
      copy(gate.scale, guScale, 0, gate.N * gate.gpr * 4);
      copy(up.scale, guScale, gate.N * gate.gpr * 4, up.N * up.gpr * 4);
      this.gateUp[L.index] = { w: guW, scale: guScale, K: gate.K, N: gate.N, gpr: gate.gpr };
      this.packedBytes += guWBytes + guScaleBytes;
    }
    this.dev.queue.submit([enc.finish()]);
    await this.dev.queue.onSubmittedWorkDone();
  }
  memoryFootprintBytes() {
    const c = this.cfg;
    const kvBytes = c.numLayers * 2 * c.numKVHeads * this.maxCtx * c.headDim * 4;
    const decodeScratchBytes = c.hiddenSize * 2 * 4 + (c.numHeads * c.headDim + 2 * c.numKVHeads * c.headDim + c.numHeads * c.headDim) * 4 + (Math.max(c.numHeads * c.headDim, c.intermediateSize) + c.intermediateSize + c.vocabSize) * 4;
    const prefillScratchBytes = this.sTcap ? this.sTcap * (3 * c.hiddenSize + c.numHeads * c.headDim + 2 * c.numKVHeads * c.headDim + c.numHeads * c.headDim + 2 * c.intermediateSize) * 4 : 0;
    return { kvBytes, decodeScratchBytes, prefillScratchBytes, packedBytes: this.packedBytes };
  }
  _gemvMeta(q, biasBuf, mod) {
    const gx = Math.min(q.N, 65535);
    const bytes = new Uint8Array(32);
    const dv = new DataView(bytes.buffer);
    dv.setUint32(0, q.K, true);
    dv.setUint32(4, q.N, true);
    dv.setUint32(8, mod ? mod.rank : 0, true);
    dv.setUint32(12, biasBuf ? 1 : 0, true);
    dv.setUint32(16, mod ? 1 : 0, true);
    dv.setUint32(20, gx, true);
    dv.setFloat32(24, mod ? mod.scale : 0, true);
    return {
      gx,
      gy: Math.ceil(q.N / gx),
      bytes
    };
  }
  _gemv4Meta(q, biasBuf, mod) {
    const gx = Math.min(q.N, 65535);
    const bytes = new Uint8Array(32);
    const dv = new DataView(bytes.buffer);
    dv.setUint32(0, q.K, true);
    dv.setUint32(4, q.N, true);
    dv.setUint32(8, mod ? mod.rank : 0, true);
    dv.setUint32(12, biasBuf ? 1 : 0, true);
    dv.setUint32(16, mod ? 1 : 0, true);
    dv.setUint32(20, gx, true);
    dv.setFloat32(24, mod ? mod.scale : 0, true);
    dv.setUint32(28, q.gpr, true);
    return {
      gx,
      gy: Math.ceil(q.N / gx),
      bytes
    };
  }
  setLora(adapter) {
    this.lora = adapter;
    this._loraEpoch++;
    this.pool.clearSensitiveBindGroups();
  }
  // {modules: {key:{A,B,rank,scale}}}  A:[K][rank], B:[rank][N] f32 GPUBuffers
  clearLora() {
    this.lora = null;
    this._loraEpoch++;
    this.pool.clearSensitiveBindGroups();
  }
  // Called after an in-place mutation of the active adapter's A/B buffers (e.g. an
  // optimizer step during training). Bumps the LoRA epoch so cached bind groups that
  // referenced the old contents are dropped and inference re-binds the mutated buffers.
  invalidateLora() {
    this._loraEpoch++;
    this.pool.clearSensitiveBindGroups();
  }
  _bg(pipe, buffers) {
    return this.pool.uncachedBindGroup(pipe, buffers);
  }
  _bgCached(pipe, buffers, key, opts) {
    return this.pool.cachedBindGroup(pipe, buffers, key, opts);
  }
  _dispatch(enc, pipe, bg, gx, gy = 1, cat, imm = null) {
    this.lastDispatchCount++;
    let ts;
    if (this.prof && this.prof.idx < this.prof.cap) {
      const i = this.prof.idx++;
      this.prof.cats.push(cat || "misc");
      ts = { querySet: this.prof.qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
    }
    const p = enc.beginComputePass(ts ? { timestampWrites: ts } : void 0);
    p.setPipeline(pipe);
    if (bg) p.setBindGroup(0, bg);
    if (imm) {
      if (Array.isArray(imm)) {
        let off = 0;
        for (const part of imm) {
          p.setImmediates(off, part);
          off += part.byteLength || part.length * (part.BYTES_PER_ELEMENT || 4);
        }
      } else {
        p.setImmediates(0, imm);
      }
    }
    p.dispatchWorkgroups(gx, gy);
    p.end();
  }
  enableProf(cap = 700) {
    this.prof = {
      qs: this.dev.createQuerySet({ type: "timestamp", count: cap * 2 }),
      cap,
      idx: 0,
      cats: [],
      resolve: this._buf(cap * 16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC),
      read: this._buf(cap * 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    };
  }
  async profToken(id, pos) {
    this._resetUni();
    this.prof.idx = 0;
    this.prof.cats = [];
    const enc = this.dev.createCommandEncoder();
    this.embedRow(enc, id);
    this.step(enc, id, pos);
    const n = this.prof.idx;
    enc.resolveQuerySet(this.prof.qs, 0, n * 2, this.prof.resolve, 0);
    enc.copyBufferToBuffer(this.prof.resolve, 0, this.prof.read, 0, n * 16);
    this.dev.queue.submit([enc.finish()]);
    await this.prof.read.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(this.prof.read.getMappedRange());
    const sums = {};
    for (let i = 0; i < n; i++) {
      const us = Number(t[2 * i + 1] - t[2 * i]) / 1e3;
      const c = this.prof.cats[i];
      sums[c] = (sums[c] || 0) + us;
    }
    this.prof.read.unmap();
    return sums;
  }
  poolStats() {
    return this.pool.stats();
  }
  // Phase 4 observability: best workgroup sizes chosen by autotune (or null if not run).
  getBestWorkgroupSizes() {
    return this.bestWorkgroupSizes ? { ...this.bestWorkgroupSizes } : null;
  }
  resetPoolStats() {
    this.pool.resetStats();
  }
  estimateKvCacheBytes() {
    const c = this.cfg;
    return c.numLayers * 2 * c.numKVHeads * this.maxCtx * c.headDim * 4;
  }
  estimatePrefillScratchBytes(T, loraRank = this._activeMaxLoraRank()) {
    const c = this.cfg, H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize;
    return T * H * 4 * 2 + T * qd * 4 * 2 + T * kvd * 4 * 2 + T * I * 4 * 2 + T * 4 + Math.max(1, T * Math.max(1, loraRank)) * 4;
  }
  greedyBatchSizeFor({ emitted = 0, remaining = Infinity, pos = 0 } = {}) {
    const interactive = emitted < this.decodeBatchWarmupTokens ? this.decodeBatchWarmupSize : this.MAXBATCH;
    return Math.max(0, Math.min(interactive, remaining, this.maxCtx - pos, this.decodeBatchCapacity));
  }
  async _resetAutotuneDecodeState(tokens, seedTokenId = 0) {
    const c = this.cfg, S = this.s, H = c.hiddenSize, hd = c.headDim, qd = c.numHeads * hd, kvd = c.numKVHeads * hd, I = c.intermediateSize;
    const nsplitMax = Math.ceil(this.maxCtx / this.CHUNK);
    const touchedTokens = Math.min(Math.max(0, Math.floor(tokens)), this.maxCtx);
    const enc = this.dev.createCommandEncoder();
    const clear = /* @__PURE__ */ __name((buf, bytes) => {
      if (bytes > 0) enc.clearBuffer(buf, 0, bytes);
    }, "clear");
    clear(S.hidden, H * 4);
    clear(S.normed, H * 4);
    clear(S.q, qd * 4);
    clear(S.k, kvd * 4);
    clear(S.v, kvd * 4);
    clear(S.attn, qd * 4);
    clear(S.tmp, Math.max(qd, I) * 4);
    clear(S.tmp2, I * 4);
    clear(S.logits, c.vocabSize * 4);
    clear(S.loraD, 256 * 4);
    clear(S.idsBuf, this.decodeBatchCapacity * 4);
    clear(S.pm, c.numHeads * nsplitMax * 4);
    clear(S.pz, c.numHeads * nsplitMax * 4);
    clear(S.po, c.numHeads * nsplitMax * hd * 4);
    const kvBytes = touchedTokens * kvd * 4;
    for (let i = 0; i < c.numLayers; i++) {
      clear(this.kc[i], kvBytes);
      clear(this.vc[i], kvBytes);
    }
    this.dev.queue.submit([enc.finish()]);
    this.dev.queue.writeBuffer(S.amax, 0, new Uint32Array([seedTokenId]));
    if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
  }
  async autotuneDecodeBatch() {
    const candidates = [...new Set(this.decodeBatchCandidates)].filter((k) => k >= 1 && k <= this.decodeBatchCapacity && k <= this.maxCtx).sort((a, b) => a - b);
    const rows = [];
    const resetTokens = candidates.length ? Math.max(...candidates) : 0;
    let selected = candidates[0] ?? this.MAXBATCH, best = Infinity;
    try {
      for (const k of candidates) {
        await this._resetAutotuneDecodeState(resetTokens);
        const t0 = performance.now();
        await this.decodeGreedyBatch(0, k);
        const ms = performance.now() - t0;
        const msPerToken = ms / k;
        rows.push({ k, ms, msPerToken });
        const latencyOk = !Number.isFinite(this.decodeBatchMaxLatencyMs) || ms <= this.decodeBatchMaxLatencyMs;
        if (latencyOk && msPerToken < best) {
          best = msPerToken;
          selected = k;
        }
      }
      if (!rows.some((r) => r.k === selected) && rows.length)
        selected = rows.reduce((a, b) => a.msPerToken <= b.msPerToken ? a : b).k;
      this.MAXBATCH = selected;
      this.decodeBatchTuning = {
        selected,
        candidates: rows,
        reason: "auto wall-clock decodeGreedyBatch with reset state"
      };
    } catch (e) {
      this.decodeBatchTuning = { selected: this.MAXBATCH, candidates: rows, reason: `auto failed: ${e.message}` };
    } finally {
      if (resetTokens > 0) {
        try {
          await this._resetAutotuneDecodeState(resetTokens);
        } catch {
        }
      }
    }
    return this.decodeBatchTuning;
  }
  // y = int8-GEMV(x, q) [+bias] [+lora]. q={w,scale,N,K}. moduleKey for LoRA lookup.
  gemv(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this._loraA(enc, xBuf, q, mod, this.s.loraD, moduleKey);
    const meta = this._gemvMeta(q, biasBuf, mod);
    const key = `gemv:${moduleKey || "base"}:${q.K}:${q.N}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv,
      [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf],
      key,
      { sensitive: !!mod }
    );
    this._dispatch(enc, this.pipes.gemv, bg, meta.gx, meta.gy, `gemv:${q.N}x${q.K}`, meta.bytes);
  }
  gemv4(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (this.debugCapture) console.log("VWG gemv4: " + moduleKey + " mod=" + !!mod);
    if (mod) this._loraA(enc, xBuf, q, mod, this.s.loraD, moduleKey);
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4:${moduleKey || "base"}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv4,
      [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf],
      key,
      { sensitive: !!mod }
    );
    this._dispatch(enc, this.pipes.gemv4, bg, meta.gx, meta.gy, `g4:${q.N}x${q.K}`, meta.bytes);
    if (mod) {
      if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj" && this.debugStep < this.debugT) {
        enc.copyBufferToBuffer(yBuf, 0, this.debugBufs.ySeq, this.debugStep * q.N * 4, q.N * 4);
        this.debugStep++;
      }
    }
  }
  _loraA(enc, xBuf, q, mod, dBuf, moduleKey, label = "loraA") {
    const imm = new Uint32Array([q.K, mod.rank]);
    this._dispatch(
      enc,
      this.pipes.loraA,
      this._bgCached(this.pipes.loraA, [xBuf, mod.A, dBuf], `${label}:${moduleKey}:${this._loraEpoch}`, {
        sensitive: true
      }),
      mod.rank,
      1,
      label,
      imm
    );
    if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj" && this.debugStep < this.debugT) {
      enc.copyBufferToBuffer(xBuf, 0, this.debugBufs.xSeq, this.debugStep * q.K * 4, q.K * 4);
      enc.copyBufferToBuffer(dBuf, 0, this.debugBufs.dSeq, this.debugStep * mod.rank * 4, mod.rank * 4);
    }
  }
  _loraBAdd(enc, yBuf, q, mod, dBuf, moduleKey) {
    const meta = new ArrayBuffer(32);
    const dv = new DataView(meta);
    dv.setUint32(0, q.N, true);
    dv.setUint32(4, mod.rank, true);
    dv.setFloat32(16, mod.scale, true);
    const bg = this._bgCached(
      this.pipes.loraBAdd,
      [dBuf, mod.B, yBuf],
      `loraBAdd:${moduleKey}:${this._loraEpoch}`,
      { sensitive: true }
    );
    this._dispatch(enc, this.pipes.loraBAdd, bg, Math.ceil(q.N / 256), 1, "loraB", new Uint8Array(meta));
    if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj" && this.debugStep < this.debugT) {
      enc.copyBufferToBuffer(yBuf, 0, this.debugBufs.ySeq, this.debugStep * q.N * 4, q.N * 4);
      this.debugStep++;
    }
  }
  gemv4Add(enc, xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this._loraA(enc, xBuf, q, mod, this.s.loraD, moduleKey);
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4add:${moduleKey || "base"}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv4Add,
      [xBuf, q.w, q.scale, biasBuf || this.s.dummy, this.s.loraD, mod ? mod.B : this.s.dummy, yBuf],
      key,
      { sensitive: !!mod }
    );
    this._dispatch(enc, this.pipes.gemv4Add, bg, meta.gx, meta.gy, `g4add:${q.N}x${q.K}`, meta.bytes);
  }
  dynQuant(enc, xBuf, x_qBuf, scale_xBuf, K) {
    const numGroups = Math.ceil(K / 128);
    const imm = new Uint32Array([K]);
    const bg = this._bg(this.pipes.dynQuant, [xBuf, x_qBuf, scale_xBuf]);
    this._dispatch(enc, this.pipes.dynQuant, bg, numGroups, 1, "dynQuant", imm);
  }
  dynQuantT(enc, xBuf, x_qBuf, scale_xBuf, K, T) {
    const numGroups = Math.ceil(K / 128);
    const imm = new Uint32Array([K, T]);
    const bg = this._bg(this.pipes.dynQuantT, [xBuf, x_qBuf, scale_xBuf]);
    this._dispatch(enc, this.pipes.dynQuantT, bg, numGroups, T, "dynQuantT", imm);
  }
  gemv4W4A8(enc, xBuf, x_qBuf, scale_xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this._loraA(enc, xBuf, q, mod, this.s.loraD, moduleKey);
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4_w4a8:${moduleKey || "base"}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv4W4A8,
      [
        x_qBuf,
        scale_xBuf,
        q.w,
        q.scale,
        biasBuf || this.s.dummy,
        this.s.loraD,
        mod ? mod.B : this.s.dummy,
        yBuf
      ],
      key,
      { sensitive: !!mod }
    );
    this._dispatch(enc, this.pipes.gemv4W4A8, bg, meta.gx, meta.gy, `g4w4a8:${q.N}x${q.K}`, meta.bytes);
  }
  gemv4AddW4A8(enc, xBuf, x_qBuf, scale_xBuf, q, yBuf, biasBuf, moduleKey) {
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this._loraA(enc, xBuf, q, mod, this.s.loraD, moduleKey);
    const meta = this._gemv4Meta(q, biasBuf, mod);
    const key = `gemv4add_w4a8:${moduleKey || "base"}:${q.K}:${q.N}:${q.gpr}:${biasBuf ? 1 : 0}:${mod ? this._loraEpoch : 0}`;
    const bg = this._bgCached(
      this.pipes.gemv4AddW4A8,
      [
        x_qBuf,
        scale_xBuf,
        q.w,
        q.scale,
        biasBuf || this.s.dummy,
        this.s.loraD,
        mod ? mod.B : this.s.dummy,
        yBuf
      ],
      key,
      { sensitive: !!mod }
    );
    this._dispatch(enc, this.pipes.gemv4AddW4A8, bg, meta.gx, meta.gy, `g4addw4a8:${q.N}x${q.K}`, meta.bytes);
  }
  qkvGemv4W4A8(enc, xBuf, x_qBuf, scale_xBuf, packed, qBuf, kBuf, vBuf, L) {
    const gx = Math.min(packed.totalN, 65535);
    const imm = new Uint32Array([packed.K, packed.totalN, packed.qN, packed.kN, packed.vN, packed.gpr, gx, 0]);
    const bg = this._bgCached(
      this.pipes.qkvGemv4W4A8,
      [x_qBuf, scale_xBuf, packed.w, packed.scale, packed.bias, qBuf, kBuf, vBuf],
      `qkv_w4a8:${L.index}`,
      { sensitive: false }
    );
    this._dispatch(
      enc,
      this.pipes.qkvGemv4W4A8,
      bg,
      gx,
      Math.ceil(packed.totalN / gx),
      `qkvw4a8:${packed.totalN}x${packed.K}`,
      imm
    );
    for (const [part, out] of [
      [L.q, qBuf],
      [L.k, kBuf],
      [L.v, vBuf]
    ]) {
      const mod = this.lora?.modules?.[part.loraKey];
      if (!mod) continue;
      const q = this.q4[part.weight];
      this._loraA(enc, xBuf, q, mod, this.s.loraD, part.loraKey);
      this._loraBAdd(enc, out, q, mod, this.s.loraD, part.loraKey);
    }
  }
  _gateUpImmediate(packed, gx, gateMod, upMod) {
    const imm = new Uint32Array(12);
    imm.set([
      packed.K,
      packed.N,
      packed.gpr,
      gx,
      gateMod ? gateMod.rank : 0,
      upMod ? upMod.rank : 0,
      gateMod ? 1 : 0,
      upMod ? 1 : 0
    ]);
    const f322 = new Float32Array(imm.buffer);
    f322[8] = gateMod ? gateMod.scale : 0;
    f322[9] = upMod ? upMod.scale : 0;
    return imm;
  }
  gateUpSiluGemv4W4A8(enc, xBuf, x_qBuf, scale_xBuf, packed, yBuf, L) {
    const gate = this.q4[L.gate.weight], up = this.q4[L.up.weight];
    const gateMod = this.lora?.modules?.[L.gate.loraKey];
    const upMod = this.lora?.modules?.[L.up.loraKey];
    if (gateMod) this._loraA(enc, xBuf, gate, gateMod, this.s.loraD, L.gate.loraKey, "loraA:gate");
    if (upMod) this._loraA(enc, xBuf, up, upMod, this.s.loraD2, L.up.loraKey, "loraA:up");
    const gx = Math.min(packed.N, 65535);
    const imm = this._gateUpImmediate(packed, gx, gateMod, upMod);
    const bg = this._bgCached(
      this.pipes.gateUpSiluGemv4W4A8,
      [
        x_qBuf,
        scale_xBuf,
        packed.w,
        packed.scale,
        yBuf,
        this.s.loraD,
        gateMod ? gateMod.B : this.s.dummy,
        this.s.loraD2,
        upMod ? upMod.B : this.s.dummy
      ],
      `gu_w4a8:${L.index}:${this._loraEpoch}:${gateMod ? 1 : 0}:${upMod ? 1 : 0}`,
      { sensitive: !!(gateMod || upMod) }
    );
    this._dispatch(
      enc,
      this.pipes.gateUpSiluGemv4W4A8,
      bg,
      gx,
      Math.ceil(packed.N / gx),
      `guw4a8:${packed.N}x${packed.K}`,
      imm
    );
  }
  gemm4W4A8(enc, aBuf, a_qBuf, scale_xBuf, q, yBuf, T, biasBuf, moduleKey) {
    const imm = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4W4A8, [a_qBuf, scale_xBuf, q.w, q.scale, biasBuf || this.s.dummy, yBuf]);
    this._dispatch(enc, this.pipes.gemm4W4A8, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), "gemm4W4A8", imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }
  gemm4AddTW4A8(enc, aBuf, a_qBuf, scale_xBuf, q, yBuf, T, biasBuf, moduleKey) {
    const imm = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4AddTW4A8, [
      a_qBuf,
      scale_xBuf,
      q.w,
      q.scale,
      biasBuf || this.s.dummy,
      yBuf
    ]);
    this._dispatch(enc, this.pipes.gemm4AddTW4A8, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), "gemm4AddTW4A8", imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }
  // Fused decode: RMSNorm + int4 QKV GEMV + RoPE in one dispatch. The kernel
  // assigns ONE workgroup per (head, rotation) pair, so it must be launched with
  // totalPairs = (qN+kN+vN)/2 workgroups and the matching grid width — the prior
  // `20`-workgroup launch (+ element-count meta) left most Q/K/V outputs unwritten
  // and produced garbage tokens. The kernel normalizes x on the fly and has no
  // `normed` output, so this path is for the NO-LoRA case only; callers must route
  // LoRA-bearing layers to the unfused gemv4x3 path (which can add the adapter).
  rmsNormQkvRope(enc, xBuf, layerIndex, pos) {
    const c = this.cfg, L = this.plan.layers[layerIndex];
    const packed = this.qkv[L.index];
    const qPairs = packed.qN / 2, kPairs = packed.kN / 2, vPairs = packed.vN / 2;
    const totalPairs = qPairs + kPairs + vPairs;
    const gx = Math.min(totalPairs, 65535);
    const meta = new Uint32Array([
      packed.K,
      totalPairs,
      qPairs,
      kPairs,
      vPairs,
      packed.gpr,
      gx,
      pos,
      c.headDim,
      ...new Uint32Array(new Float32Array([c.rmsNormEps, packed.qN, packed.kN]).buffer)
    ]);
    const bg = this._bg(
      this.pipes.rmsNormQkvRope,
      [
        xBuf,
        this.bufs[L.inputNorm],
        packed.w,
        packed.scale,
        packed.bias,
        this.ropeCos,
        this.ropeSin,
        this.s.q,
        this.s.k,
        this.s.v
      ]
    );
    this._dispatch(enc, this.pipes.rmsNormQkvRope, bg, gx, Math.ceil(totalPairs / gx), "rmsNormQkvRope", meta);
  }
  writeKvPage(enc, kBuf, vBuf, kcBuf, vcBuf, pos, layerIndex) {
    const c = this.cfg;
    const kvd = c.numKVHeads * c.headDim;
    this.pam.ensureBlocks(0, pos + 1);
    const btArr = this.pam.getBlockTableArray(0);
    this.dev.queue.writeBuffer(this.s.blockTableBuf, 0, btArr);
    const meta = new Uint32Array([pos, 0, this.pam.maxBlocksPerSeq, kvd]);
    const bg = this._bg(this.pipes.writeKvPage, [kBuf, vBuf, kcBuf, vcBuf, this.s.blockTableBuf]);
    this._dispatch(enc, this.pipes.writeKvPage, bg, Math.ceil(kvd / 256), 1, "writeKvPage", meta);
  }
  writeKvPageBatch(enc, kBuf, vBuf, kcBuf, vcBuf, T, off, layerIndex) {
    const c = this.cfg;
    const kvd = c.numKVHeads * c.headDim;
    this.pam.ensureBlocks(0, off + T);
    const btArr = this.pam.getBlockTableArray(0);
    this.dev.queue.writeBuffer(this.s.blockTableBuf, 0, btArr);
    const meta = new Uint32Array([T, 0, this.pam.maxBlocksPerSeq, kvd, off]);
    const bg = this._bg(this.pipes.writeKvPageBatch, [kBuf, vBuf, kcBuf, vcBuf, this.s.blockTableBuf]);
    this._dispatch(enc, this.pipes.writeKvPageBatch, bg, Math.ceil(T * kvd / 256), 1, "writeKvPageBatch", meta);
  }
  attnPaged(enc, qBuf, kc, vc, oBuf, ctx) {
    const c = this.cfg, S = this.s;
    const nsplit = Math.ceil(ctx / this.CHUNK);
    const bgP = this._bg(this.pipes.attnPartialPaged, [
      qBuf,
      kc,
      vc,
      S.pm,
      S.pz,
      S.po,
      S.blockTableBuf
    ]);
    const immP = new Uint32Array([c.numHeads, c.numKVHeads, ctx, c.headDim, nsplit, this.CHUNK, 0, this.pam.maxBlocksPerSeq]);
    this._dispatch(enc, this.pipes.attnPartialPaged, bgP, c.numHeads, nsplit, "attnP_paged", immP);
    const useF16C = this.usingF16() && this.pipes.attnCF16;
    const pipeC = useF16C ? this.pipes.attnCF16 : this.pipes.attnC;
    const bgC = this._bg(pipeC, [
      S.pm,
      S.pz,
      S.po,
      oBuf
    ]);
    const immC = new Uint32Array([c.numHeads, c.headDim, nsplit, 0]);
    this._dispatch(enc, pipeC, bgC, c.numHeads, 1, useF16C ? "attnCF16" : "attnC", immC);
  }
  attnPrefillPaged(enc, qBuf, kc, vc, oBuf, T, qStart = 0, ctx = T) {
    const c = this.cfg;
    if (this.features.prefillAttention === "block" || qStart !== 0 || ctx !== T) {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T, qStart, ctx, 0, this.pam.maxBlocksPerSeq]);
      this._dispatch(
        enc,
        this.pipes.attnPrefillBlockPaged,
        this._bg(this.pipes.attnPrefillBlockPaged, [qBuf, kc, vc, oBuf, this.s.blockTableBuf]),
        c.numHeads,
        Math.ceil(T / 4),
        "attnPrefillBlockPaged",
        imm
      );
    } else {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T, 0, this.pam.maxBlocksPerSeq, 0, 0]);
      this._dispatch(
        enc,
        this.pipes.attnPrefillPaged,
        this._bg(this.pipes.attnPrefillPaged, [
          qBuf,
          kc,
          vc,
          oBuf,
          this.s.blockTableBuf
        ]),
        c.numHeads,
        T,
        "attnPrefillPaged",
        imm
      );
    }
  }
  qkvGemv4(enc, xBuf, packed, qBuf, kBuf, vBuf, L) {
    const gx = Math.min(packed.totalN, 65535);
    const imm = new Uint32Array([packed.K, packed.totalN, packed.qN, packed.kN, packed.vN, packed.gpr, gx, 0]);
    const bg = this._bgCached(
      this.pipes.qkvGemv4,
      [xBuf, packed.w, packed.scale, packed.bias, qBuf, kBuf, vBuf],
      `qkv:${L.index}`,
      { sensitive: false }
    );
    this._dispatch(enc, this.pipes.qkvGemv4, bg, gx, Math.ceil(packed.totalN / gx), `qkv:${packed.totalN}x${packed.K}`, imm);
    for (const [part, out] of [
      [L.q, qBuf],
      [L.k, kBuf],
      [L.v, vBuf]
    ]) {
      const mod = this.lora?.modules?.[part.loraKey];
      if (!mod) continue;
      const q = this.q4[part.weight];
      this._loraA(enc, xBuf, q, mod, this.s.loraD, part.loraKey);
      this._loraBAdd(enc, out, q, mod, this.s.loraD, part.loraKey);
    }
  }
  fusedRmsQkvRope(enc, hiddenBuf, inputNormBuf, packed, qBuf, kBuf, vBuf, pos, L) {
    const qPairs = packed.qN / 2;
    const kPairs = packed.kN / 2;
    const vPairs = packed.vN / 2;
    const totalPairs = qPairs + kPairs + vPairs;
    const gx = Math.min(totalPairs, 65535);
    const meta = new Uint32Array([
      packed.K,
      totalPairs,
      qPairs,
      kPairs,
      vPairs,
      packed.gpr,
      gx,
      pos,
      this.cfg.headDim,
      ...new Uint32Array(new Float32Array([this.cfg.rmsNormEps, packed.qN, packed.kN]).buffer)
    ]);
    const bg = this._bg(
      this.pipes.rmsNormQkvRope,
      [
        hiddenBuf,
        inputNormBuf,
        packed.w,
        packed.scale,
        packed.bias,
        this.ropeCos,
        this.ropeSin,
        qBuf,
        kBuf,
        vBuf
      ]
    );
    this._dispatch(
      enc,
      this.pipes.rmsNormQkvRope,
      bg,
      gx,
      Math.ceil(totalPairs / gx),
      `fusedQkvRope:${totalPairs}x${packed.K}`,
      meta
    );
  }
  gateUpSiluGemv4(enc, xBuf, packed, yBuf, L) {
    const gate = this.q4[L.gate.weight], up = this.q4[L.up.weight];
    const gateMod = this.lora?.modules?.[L.gate.loraKey];
    const upMod = this.lora?.modules?.[L.up.loraKey];
    if (gateMod) this._loraA(enc, xBuf, gate, gateMod, this.s.loraD, L.gate.loraKey, "loraA:gate");
    if (upMod) this._loraA(enc, xBuf, up, upMod, this.s.loraD2, L.up.loraKey, "loraA:up");
    const gx = Math.min(packed.N, 65535);
    const imm = this._gateUpImmediate(packed, gx, gateMod, upMod);
    const bg = this._bgCached(
      this.pipes.gateUpSiluGemv4,
      [
        xBuf,
        packed.w,
        packed.scale,
        yBuf,
        this.s.loraD,
        gateMod ? gateMod.B : this.s.dummy,
        this.s.loraD2,
        upMod ? upMod.B : this.s.dummy
      ],
      `gu:${L.index}:${this._loraEpoch}:${gateMod ? 1 : 0}:${upMod ? 1 : 0}`,
      { sensitive: !!(gateMod || upMod) }
    );
    this._dispatch(enc, this.pipes.gateUpSiluGemv4, bg, gx, Math.ceil(packed.N / gx), `gu:${packed.N}x${packed.K}`, imm);
  }
  rms(enc, xBuf, gBuf, yBuf, K) {
    const imm = new Float32Array([K, this.cfg.rmsNormEps]);
    const useF16 = this.usingF16() && this.pipes.rmsF16;
    const pipe = useF16 ? this.pipes.rmsF16 : this.pipes.rms;
    const key = `rms:${K}${useF16 ? ":f16" : ""}`;
    this._dispatch(enc, pipe, this._bgCached(pipe, [xBuf, gBuf, yBuf], key), 1, 1, useF16 ? "rmsF16" : "rms", imm);
  }
  rope(enc, xBuf, pos, nHeads) {
    const useF16 = this.usingF16() && this.pipes.ropeF16;
    const pipe = useF16 ? this.pipes.ropeF16 : this.pipes.rope;
    this._dispatch(
      enc,
      pipe,
      this._bg(pipe, [
        xBuf,
        this.ropeCos,
        this.ropeSin
      ]),
      Math.ceil(nHeads * (this.cfg.headDim / 2) / 256),
      1,
      useF16 ? "ropeF16" : "rope",
      new Uint32Array([nHeads, this.cfg.headDim, pos])
    );
  }
  ropeQK(enc, qBuf, kBuf, pos) {
    const c = this.cfg;
    const pairs = (c.numHeads + c.numKVHeads) * (c.headDim / 2);
    const useF16 = this.usingF16() && this.pipes.ropeQKF16;
    const pipe = useF16 ? this.pipes.ropeQKF16 : this.pipes.ropeQK;
    this._dispatch(
      enc,
      pipe,
      this._bg(pipe, [
        qBuf,
        kBuf,
        this.ropeCos,
        this.ropeSin
      ]),
      Math.ceil(pairs / 256),
      1,
      useF16 ? "ropeQKF16" : "ropeQK",
      new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, pos])
    );
  }
  attn(enc, qBuf, kc, vc, oBuf, ctx) {
    const c = this.cfg, S = this.s;
    const nsplit = Math.ceil(ctx / this.CHUNK);
    const useF16P = this.usingF16() && this.pipes.attnPF16;
    const pipeP = useF16P ? this.pipes.attnPF16 : this.pipes.attnP;
    const bgP = this._bg(pipeP, [
      qBuf,
      kc,
      vc,
      S.pm,
      S.pz,
      S.po
    ]);
    const immP = new Uint32Array([c.numHeads, c.numKVHeads, ctx, c.headDim, nsplit, this.CHUNK]);
    this._dispatch(enc, pipeP, bgP, c.numHeads, nsplit, useF16P ? "attnPF16" : "attnP", immP);
    const useF16C = this.usingF16() && this.pipes.attnCF16;
    const pipeC = useF16C ? this.pipes.attnCF16 : this.pipes.attnC;
    const bgC = this._bg(pipeC, [
      S.pm,
      S.pz,
      S.po,
      oBuf
    ]);
    const immC = new Uint32Array([c.numHeads, c.headDim, nsplit, 0]);
    this._dispatch(enc, pipeC, bgC, c.numHeads, 1, useF16C ? "attnCF16" : "attnC", immC);
  }
  // Decode one token at absolute position `pos`. Writes logits to s.logits. Returns nothing.
  step(enc, tokenId, pos) {
    const c = this.cfg, S = this.s, hd = c.headDim, kvd = c.numKVHeads * hd;
    for (let i = 0; i < c.numLayers; i++) {
      const L = this.plan.layers[i];
      const hasQkvLora = this.lora && (this.lora.modules[L.q.loraKey] || this.lora.modules[L.k.loraKey] || this.lora.modules[L.v.loraKey]);
      if (this.features.fuseRMSNormQKVRoPE && !hasQkvLora && !this.features.actQuant) {
        this.rmsNormQkvRope(enc, S.hidden, i, pos);
      } else {
        this.rms(enc, S.hidden, this.bufs[L.inputNorm], S.normed, c.hiddenSize);
        if (this.features.actQuant) {
          this.dynQuant(enc, S.normed, S.x_q, S.scale_x, c.hiddenSize);
          this.qkvGemv4W4A8(enc, S.normed, S.x_q, S.scale_x, this.qkv[L.index], S.q, S.k, S.v, L);
        } else {
          if (!hasQkvLora && this.features.fuseQKV) {
            this.fusedRmsQkvRope(enc, S.hidden, this.bufs[L.inputNorm], this.qkv[L.index], S.q, S.k, S.v, pos, L);
          } else if (this.features.fuseQKV) {
            this.qkvGemv4(enc, S.normed, this.qkv[L.index], S.q, S.k, S.v, L);
            if (this.features.fuseRoPE) this.ropeQK(enc, S.q, S.k, pos);
            else {
              this.rope(enc, S.q, pos, c.numHeads);
              this.rope(enc, S.k, pos, c.numKVHeads);
            }
          } else {
            this.gemv4(enc, S.normed, this.q4[L.q.weight], S.q, this.bufs[L.q.bias], L.q.loraKey);
            this.gemv4(enc, S.normed, this.q4[L.k.weight], S.k, this.bufs[L.k.bias], L.k.loraKey);
            this.gemv4(enc, S.normed, this.q4[L.v.weight], S.v, this.bufs[L.v.bias], L.v.loraKey);
            if (this.features.fuseRoPE) this.ropeQK(enc, S.q, S.k, pos);
            else {
              this.rope(enc, S.q, pos, c.numHeads);
              this.rope(enc, S.k, pos, c.numKVHeads);
            }
          }
        }
      }
      if (this.features.pagedAttention) {
        this.writeKvPage(enc, S.k, S.v, this.kc[i], this.vc[i], pos, i);
      } else {
        enc.copyBufferToBuffer(S.k, 0, this.kc[i], pos * kvd * 4, kvd * 4);
        enc.copyBufferToBuffer(S.v, 0, this.vc[i], pos * kvd * 4, kvd * 4);
      }
      if (this.features.pagedAttention) {
        this.attnPaged(enc, S.q, this.kc[i], this.vc[i], S.attn, pos + 1);
      } else {
        this.attn(enc, S.q, this.kc[i], this.vc[i], S.attn, pos + 1);
      }
      if (this.features.actQuant) {
        this.dynQuant(enc, S.attn, S.x_q, S.scale_x, c.hiddenSize);
        if (this.features.fuseResidual) {
          this.gemv4AddW4A8(enc, S.attn, S.x_q, S.scale_x, this.q4[L.o.weight], S.hidden, null, L.o.loraKey);
        } else {
          this.gemv4W4A8(enc, S.attn, S.x_q, S.scale_x, this.q4[L.o.weight], S.tmp, null, L.o.loraKey);
          this._addInto(enc, S.hidden, S.tmp, c.hiddenSize);
        }
      } else {
        if (this.features.fuseResidual) this.gemv4Add(enc, S.attn, this.q4[L.o.weight], S.hidden, null, L.o.loraKey);
        else {
          this.gemv4(enc, S.attn, this.q4[L.o.weight], S.tmp, null, L.o.loraKey);
          this._addInto(enc, S.hidden, S.tmp, c.hiddenSize);
        }
      }
      this.rms(enc, S.hidden, this.bufs[L.postAttentionNorm], S.normed, c.hiddenSize);
      if (this.features.actQuant) {
        this.dynQuant(enc, S.normed, S.x_q, S.scale_x, c.hiddenSize);
        this.gateUpSiluGemv4W4A8(enc, S.normed, S.x_q, S.scale_x, this.gateUp[L.index], S.tmp, L);
      } else {
        if (this.features.fuseMLP) {
          this.gateUpSiluGemv4(enc, S.normed, this.gateUp[L.index], S.tmp, L);
        } else {
          this.gemv4(enc, S.normed, this.q4[L.gate.weight], S.tmp, null, L.gate.loraKey);
          this.gemv4(enc, S.normed, this.q4[L.up.weight], S.tmp2, null, L.up.loraKey);
          this._siluMul(enc, S.tmp, S.tmp2, c.intermediateSize);
        }
      }
      if (this.features.actQuant) {
        this.dynQuant(enc, S.tmp, S.x_q, S.scale_x, c.intermediateSize);
        if (this.features.fuseResidual) {
          this.gemv4AddW4A8(enc, S.tmp, S.x_q, S.scale_x, this.q4[L.down.weight], S.hidden, null, L.down.loraKey);
        } else {
          this.gemv4W4A8(enc, S.tmp, S.x_q, S.scale_x, this.q4[L.down.weight], S.normed, null, L.down.loraKey);
          this._addInto(enc, S.hidden, S.normed, c.hiddenSize);
        }
      } else {
        if (this.features.fuseResidual)
          this.gemv4Add(enc, S.tmp, this.q4[L.down.weight], S.hidden, null, L.down.loraKey);
        else {
          this.gemv4(enc, S.tmp, this.q4[L.down.weight], S.normed, null, L.down.loraKey);
          this._addInto(enc, S.hidden, S.normed, c.hiddenSize);
        }
      }
    }
    this.rms(enc, S.hidden, this.bufs[this.plan.finalNorm.name], S.normed, c.hiddenSize);
    this.gemv(enc, S.normed, this.q[this.plan.embed.name], S.logits, null, null);
  }
  _addInto(enc, yBuf, aBuf, n) {
    const imm = new Uint32Array([n]);
    const useF16 = this.usingF16() && this.pipes.addF16;
    const pipe = useF16 ? this.pipes.addF16 : this.pipes.add;
    const bg = this._bgCached(pipe, [aBuf, yBuf], `add:${n}${useF16 ? ":f16" : ""}`);
    const wg = pipe.__wg || 256;
    this._dispatch(enc, pipe, bg, Math.min(Math.ceil(n / wg), 65535), 1, useF16 ? "addF16" : "add", imm);
  }
  _siluMul(enc, gateBuf, upBuf, n) {
    const imm = new Uint32Array([n]);
    const useF16 = this.usingF16() && this.pipes.siluF16;
    const pipe = useF16 ? this.pipes.siluF16 : this.pipes.silu;
    const bg = this._bgCached(pipe, [gateBuf, upBuf], `silu:${n}${useF16 ? ":f16" : ""}`);
    const wg = pipe.__wg || 256;
    this._dispatch(enc, pipe, bg, Math.min(Math.ceil(n / wg), 65535), 1, useF16 ? "siluF16" : "silu", imm);
  }
  embedRow(enc, id) {
    const e = this.q[this.plan.embed.name];
    const imm = new Uint32Array([id, this.cfg.hiddenSize]);
    this._dispatch(
      enc,
      this.pipes.embed,
      this._bg(this.pipes.embed, [e.w, e.scale, this.s.hidden]),
      Math.ceil(this.cfg.hiddenSize / 256),
      1,
      "embed",
      imm
    );
  }
  async argmaxLogits() {
    if (this._argmaxReadBusy)
      throw new Error("argmaxLogits() is already in flight; concurrent generation is not supported");
    this._argmaxReadBusy = true;
    const enc = this.dev.createCommandEncoder();
    const n = this.cfg.vocabSize || 0;
    this._dispatch(
      enc,
      this.pipes.argmax,
      this._bgCached(this.pipes.argmax, [this.s.logits, this.s.amax], "argmax"),
      1,
      1,
      "argmax",
      new Uint32Array([n])
    );
    enc.copyBufferToBuffer(this.s.amax, 0, this.argmaxRead, 0, 4);
    this.dev.queue.submit([enc.finish()]);
    if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
    try {
      await this.argmaxRead.mapAsync(GPUMapMode.READ);
      const id = new Uint32Array(this.argmaxRead.getMappedRange())[0];
      this.argmaxRead.unmap();
      return id;
    } finally {
      this._argmaxReadBusy = false;
    }
  }
  // Convenience for numeric comparison harnesses (Phase 3 f16 eval etc.).
  // Returns a fresh Float32Array copy of the current final logits buffer.
  async readLogits() {
    const n = this.cfg.vocabSize;
    if (!this._logitsRead) {
      this._logitsRead = this._buf(n * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    }
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.s.logits, 0, this._logitsRead, 0, n * 4);
    this.dev.queue.submit([enc.finish()]);
    if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
    await this._logitsRead.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this._logitsRead.getMappedRange()).slice();
    this._logitsRead.unmap();
    return out;
  }
  async topKLogits(k = this.samplingTopK) {
    if (this._topKReadBusy) throw new Error("topKLogits() is already in flight; concurrent sampling is not supported");
    this._topKReadBusy = true;
    try {
      k = Math.min(Math.max(1, Math.floor(k)), this.maxSamplingTopK, this.cfg.vocabSize);
      const enc = this.dev.createCommandEncoder();
      for (let i = 0; i < k; i++) {
        const imm = new Uint32Array([this.cfg.vocabSize, i]);
        this._dispatch(
          enc,
          this.pipes.topkSelect,
          this._bgCached(this.pipes.topkSelect, [this.s.logits, this.s.sampleIds, this.s.sampleVals], `topk:${i}`),
          1,
          1,
          "topk",
          imm
        );
      }
      enc.copyBufferToBuffer(this.s.sampleIds, 0, this.sampleIdsRead, 0, k * 4);
      enc.copyBufferToBuffer(this.s.sampleVals, 0, this.sampleValsRead, 0, k * 4);
      this.dev.queue.submit([enc.finish()]);
      await Promise.all([this.sampleIdsRead.mapAsync(GPUMapMode.READ), this.sampleValsRead.mapAsync(GPUMapMode.READ)]);
      const ids = Array.from(new Uint32Array(this.sampleIdsRead.getMappedRange(), 0, k));
      const vals = Array.from(new Float32Array(this.sampleValsRead.getMappedRange(), 0, k));
      return ids.map((id, i) => ({ id, logit: vals[i] }));
    } finally {
      if (this.sampleIdsRead.mapState !== "unmapped") this.sampleIdsRead.unmap();
      if (this.sampleValsRead.mapState !== "unmapped") this.sampleValsRead.unmap();
      this._topKReadBusy = false;
    }
  }
  // Phase 5: GPU-resident sampling (pure-GPU top-k + sample chaining).
  // Runs the iterative top-k selection dispatches directly into the GPU sampleIds/sampleVals
  // buffers, then immediately chains the SAMPLE_TOPK kernel in the same submission.
  // Only a single u32 (the chosen token) is ever read back from the GPU.
  // This eliminates the previous k-value readbacks for the sampling path.
  async sampleToken(temp = 1, r = typeof Math !== "undefined" ? Math.random() : 0.5) {
    if (this._topKReadBusy) throw new Error("sampleToken: top-k selection already in flight");
    this._topKReadBusy = true;
    const k = Math.min(this.samplingTopK, this.maxSamplingTopK, this.cfg.vocabSize);
    try {
      const enc = this.dev.createCommandEncoder();
      for (let i = 0; i < k; i++) {
        const imm2 = new Uint32Array([this.cfg.vocabSize, i]);
        this._dispatch(
          enc,
          this.pipes.topkSelect,
          this._bgCached(this.pipes.topkSelect, [this.s.logits, this.s.sampleIds, this.s.sampleVals], `topk:${i}`),
          1,
          1,
          "topk",
          imm2
        );
      }
      const bg = this._bg(this.pipes.sampleTopK, [
        this.s.sampleIds,
        this.s.sampleVals,
        this.s.sampled
      ]);
      const imm = new Uint32Array(4);
      imm[0] = k;
      const f322 = new Float32Array(imm.buffer);
      f322[2] = temp > 0 ? temp : 1;
      f322[3] = Math.max(0, Math.min(1, r));
      this._dispatch(enc, this.pipes.sampleTopK, bg, 1, 1, "sampleTopK", imm);
      enc.copyBufferToBuffer(this.s.sampled, 0, this.sampledRead, 0, 4);
      this.dev.queue.submit([enc.finish()]);
      if (this.dev.queue.onSubmittedWorkDone) await this.dev.queue.onSubmittedWorkDone();
      await this.sampledRead.mapAsync(GPUMapMode.READ);
      const id = new Uint32Array(this.sampledRead.getMappedRange())[0];
      this.sampledRead.unmap();
      return id;
    } finally {
      this._topKReadBusy = false;
    }
  }
  // Run one token end-to-end (embed + step) and submit.
  token(id, pos) {
    this._resetUni();
    const enc = this.dev.createCommandEncoder();
    this.embedRow(enc, id);
    this.step(enc, id, pos);
    this.dev.queue.submit([enc.finish()]);
  }
  // embed the token id held in s.amax (GPU-resident, from a prior argmax)
  embedFromBuf(enc) {
    const e = this.q[this.plan.embed.name];
    const imm = new Uint32Array([this.cfg.hiddenSize]);
    this._dispatch(
      enc,
      this.pipes.embedBuf,
      this._bgCached(this.pipes.embedBuf, [e.w, e.scale, this.s.hidden, this.s.amax], "embedBuf"),
      Math.ceil(this.cfg.hiddenSize / 256),
      1,
      "embed",
      imm
    );
  }
  // argmax(logits) -> s.amax, within the given encoder (no submit/readback)
  argmaxInto(enc) {
    const n = this.cfg.vocabSize || 0;
    this._dispatch(
      enc,
      this.pipes.argmax,
      this._bgCached(this.pipes.argmax, [this.s.logits, this.s.amax], "argmax"),
      1,
      1,
      "argmax",
      new Uint32Array([n])
    );
  }
  // GPU-resident batched GREEDY decode only: chains embed->step->argmax for K
  // tokens in ONE submit, reads back K ids once, and checks stop tokens only
  // after readback. It assumes s.amax already holds the current token id to
  // embed. Do not use for sampled decoding; sampled tokens must be written by
  // the CPU/GPU sampler one step at a time.
  async decodeBatch(startPos, K) {
    K = Math.min(K, this.decodeBatchCapacity, this.maxCtx - startPos);
    if (K <= 0) return [];
    this._resetUni();
    const enc = this.dev.createCommandEncoder();
    for (let k = 0; k < K; k++) {
      this.embedFromBuf(enc);
      this.step(enc, 0, startPos + k);
      this.argmaxInto(enc);
      enc.copyBufferToBuffer(this.s.amax, 0, this.s.idsBuf, k * 4, 4);
    }
    enc.copyBufferToBuffer(this.s.idsBuf, 0, this.idsRead, 0, K * 4);
    this.dev.queue.submit([enc.finish()]);
    await this.idsRead.mapAsync(GPUMapMode.READ);
    const ids = Array.from(new Uint32Array(this.idsRead.getMappedRange(), 0, K));
    this.idsRead.unmap();
    return ids;
  }
  async decodeGreedyBatch(startPos, K) {
    return this.decodeBatch(startPos, K);
  }
  // ---- PREFILL (T>1): process the whole prompt at once via tiled GEMM. If a LoRA
  // adapter has the projection module, add its batched delta immediately after base GEMM.
  gemm4(enc, aBuf, q, yBuf, T, biasBuf, moduleKey) {
    const imm = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4, [aBuf, q.w, q.scale, biasBuf || this.s.dummy, yBuf]);
    this._dispatch(enc, this.pipes.gemm4, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), "gemm4", imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }
  gemm4AddT(enc, aBuf, q, yBuf, T, biasBuf, moduleKey) {
    const imm = new Uint32Array([q.K, q.N, T, q.gpr, biasBuf ? 1 : 0, 0, 0, 0]);
    const bg = this._bg(this.pipes.gemm4AddT, [aBuf, q.w, q.scale, biasBuf || this.s.dummy, yBuf]);
    this._dispatch(enc, this.pipes.gemm4AddT, bg, Math.ceil(q.N / 64), Math.ceil(T / 16), "gemm4AddT", imm);
    const mod = this.lora?.modules?.[moduleKey];
    if (mod) this.loraBatchDelta(enc, aBuf, yBuf, q, T, mod, moduleKey);
  }
  loraBatchDelta(enc, xBuf, yBuf, q, T, mod, moduleKey) {
    if (this.debugCapture) console.log("VWG loraBatchDelta: " + moduleKey + " mod=" + !!mod);
    const imm = new Uint32Array([q.K, mod.rank, T, 0]);
    const bgA = this._bg(this.pipes.loraABatch, [xBuf, mod.A, this.sT.loraD]);
    this._dispatch(enc, this.pipes.loraABatch, bgA, mod.rank, T, "loraA:T", imm);
    if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj") {
      enc.copyBufferToBuffer(xBuf, 0, this.debugBufs.xBat, 0, T * q.K * 4);
      enc.copyBufferToBuffer(this.sT.loraD, 0, this.debugBufs.dBat, 0, T * mod.rank * 4);
    }
    const totalGroups = Math.ceil(T * q.N / 256);
    let gx = totalGroups;
    let gy = 1;
    if (gx > 65535) {
      gx = 256;
      gy = Math.ceil(totalGroups / 256);
    }
    const meta = new ArrayBuffer(32);
    const dv = new DataView(meta);
    dv.setUint32(0, T, true);
    dv.setUint32(4, q.N, true);
    dv.setUint32(8, mod.rank, true);
    dv.setUint32(12, gx, true);
    dv.setFloat32(16, mod.scale, true);
    const bgB = this._bg(this.pipes.loraBAddT, [this.sT.loraD, mod.B, yBuf]);
    this._dispatch(enc, this.pipes.loraBAddT, bgB, gx, gy, "loraB:T", new Uint8Array(meta));
    if (this.debugCapture && moduleKey === "layers.0.self_attn.q_proj") {
      enc.copyBufferToBuffer(yBuf, 0, this.debugBufs.yBat, 0, T * q.N * 4);
      this.debugCaptured = true;
    }
  }
  rmsT(enc, xBuf, gBuf, yBuf, T, K) {
    const imm = new Float32Array([K, this.cfg.rmsNormEps]);
    const useF16 = this.usingF16() && this.pipes.rmsTF16;
    const pipe = useF16 ? this.pipes.rmsTF16 : this.pipes.rmsT;
    this._dispatch(enc, pipe, this._bg(pipe, [xBuf, gBuf, yBuf]), T, 1, useF16 ? "rmsTF16" : "rmsT", imm);
  }
  ropeT(enc, xBuf, T, nHeads, pos0 = 0) {
    const hd = this.cfg.headDim;
    const imm = new Uint32Array([nHeads, hd, T, pos0]);
    const useF16 = this.usingF16() && this.pipes.ropeTF16;
    const pipe = useF16 ? this.pipes.ropeTF16 : this.pipes.ropeT;
    this._dispatch(
      enc,
      pipe,
      this._bg(pipe, [xBuf, this.ropeCos, this.ropeSin]),
      Math.ceil(T * nHeads * (hd / 2) / 256),
      1,
      useF16 ? "ropeTF16" : "ropeT",
      imm
    );
  }
  attnPrefill(enc, qBuf, kc, vc, oBuf, T, qStart = 0, ctx = T) {
    const c = this.cfg;
    if (this.features.prefillAttention === "block" || qStart !== 0 || ctx !== T) {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T, qStart, ctx, 0, 0]);
      this._dispatch(
        enc,
        this.pipes.attnPrefillBlock,
        this._bg(this.pipes.attnPrefillBlock, [qBuf, kc, vc, oBuf]),
        c.numHeads,
        Math.ceil(T / 4),
        "attnPrefillBlock",
        imm
      );
    } else {
      const imm = new Uint32Array([c.numHeads, c.numKVHeads, c.headDim, T]);
      this._dispatch(
        enc,
        this.pipes.attnPrefill,
        this._bg(this.pipes.attnPrefill, [qBuf, kc, vc, oBuf]),
        c.numHeads,
        Math.ceil(T / 4),
        "attnPrefill",
        imm
      );
    }
  }
  // (re)allocate prefill scratch sized to T (grows as needed; only paid when prefilling).
  _ensurePrefillScratch(T, loraRank = 0, idsCap = T) {
    if (this.sTcap >= T && (this.sTLoraRank || 0) >= loraRank && (this.sTidsCap || 0) >= idsCap) return;
    const need = this.estimatePrefillScratchBytes(T, loraRank);
    if (this.opts.maxPrefillScratchBytes && need > this.opts.maxPrefillScratchBytes) {
      throw new Error(
        `prefill scratch ${Math.ceil(need / 1048576)}MiB exceeds maxPrefillScratchBytes; lower maxPrefillT or use shorter prompt chunks`
      );
    }
    if (this.sT) for (const k in this.sT) this.sT[k].destroy();
    const c = this.cfg, H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize;
    this.sT = {
      hidden: this._buf(T * H * 4),
      normed: this._buf(T * H * 4),
      q: this._buf(T * qd * 4),
      k: this._buf(T * kvd * 4),
      v: this._buf(T * kvd * 4),
      attn: this._buf(T * qd * 4),
      tmp: this._buf(T * I * 4),
      tmp2: this._buf(T * I * 4),
      ids: this._buf(idsCap * 4),
      loraD: this._buf(Math.max(1, T * Math.max(1, loraRank)) * 4),
      x_q: this._buf(T * Math.max(H, I) * 4),
      scale_x: this._buf(T * Math.max(H, I) / 128 * 4)
    };
    this.sTcap = T;
    this.sTLoraRank = loraRank;
    this.sTidsCap = idsCap;
  }
  _activeMaxLoraRank() {
    let rank = 0;
    const mods = this.lora?.modules;
    if (!mods) return 0;
    for (const key of Object.keys(mods)) rank = Math.max(rank, mods[key].rank || 0);
    return rank;
  }
  // Prefill the prompt (positions 0..T-1). Leaves last-row logits in s.logits and the
  // KV cache populated, so decode continues from pos=T. T must be <= maxPrefillT.
  prefillBatch(ids) {
    const T = ids.length;
    if (T > this.maxPrefillT) throw new Error(`prompt ${T} > maxPrefillT ${this.maxPrefillT}`);
    if (T > this.maxCtx) throw new Error(`prompt ${T} > maxCtx ${this.maxCtx}`);
    const chunk = this.features.prefillChunkSize;
    if (chunk > 0 && T > chunk) return this._prefillChunked(ids, chunk);
    return this._prefillFull(ids);
  }
  _prefillFull(ids) {
    const c = this.cfg, S = this.s, T = ids.length, hd = c.headDim, kvd = c.numKVHeads * hd, H = c.hiddenSize;
    this._ensurePrefillScratch(T, this._activeMaxLoraRank());
    const ST = this.sT;
    this._resetUni();
    this.dev.queue.writeBuffer(ST.ids, 0, new Uint32Array(ids));
    const enc = this.dev.createCommandEncoder();
    const e = this.q[this.plan.embed.name];
    const imm = new Uint32Array([T, H, 0, 0]);
    this._dispatch(
      enc,
      this.pipes.embedT,
      this._bg(this.pipes.embedT, [e.w, e.scale, ST.hidden, ST.ids]),
      Math.min(Math.ceil(T * H / 256), 65535),
      1,
      "embedT",
      imm
    );
    for (let i = 0; i < c.numLayers; i++) {
      const L = this.plan.layers[i];
      this.rmsT(enc, ST.hidden, this.bufs[L.inputNorm], ST.normed, T, H);
      if (this.features.actQuant) {
        this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, T);
        this.gemm4W4A8(
          enc,
          ST.normed,
          ST.x_q,
          ST.scale_x,
          this.q4[L.q.weight],
          ST.q,
          T,
          this.bufs[L.q.bias],
          L.q.loraKey
        );
        this.gemm4W4A8(
          enc,
          ST.normed,
          ST.x_q,
          ST.scale_x,
          this.q4[L.k.weight],
          ST.k,
          T,
          this.bufs[L.k.bias],
          L.k.loraKey
        );
        this.gemm4W4A8(
          enc,
          ST.normed,
          ST.x_q,
          ST.scale_x,
          this.q4[L.v.weight],
          ST.v,
          T,
          this.bufs[L.v.bias],
          L.v.loraKey
        );
      } else {
        this.gemm4(enc, ST.normed, this.q4[L.q.weight], ST.q, T, this.bufs[L.q.bias], L.q.loraKey);
        this.gemm4(enc, ST.normed, this.q4[L.k.weight], ST.k, T, this.bufs[L.k.bias], L.k.loraKey);
        this.gemm4(enc, ST.normed, this.q4[L.v.weight], ST.v, T, this.bufs[L.v.bias], L.v.loraKey);
      }
      this.ropeT(enc, ST.q, T, c.numHeads);
      this.ropeT(enc, ST.k, T, c.numKVHeads);
      if (this.features.pagedAttention) {
        this.writeKvPageBatch(enc, ST.k, ST.v, this.kc[i], this.vc[i], T, 0, i);
      } else {
        enc.copyBufferToBuffer(ST.k, 0, this.kc[i], 0, T * kvd * 4);
        enc.copyBufferToBuffer(ST.v, 0, this.vc[i], 0, T * kvd * 4);
      }
      if (this.features.pagedAttention) {
        this.attnPrefillPaged(enc, ST.q, this.kc[i], this.vc[i], ST.attn, T, 0, T);
      } else {
        this.attnPrefill(enc, ST.q, this.kc[i], this.vc[i], ST.attn, T, 0, T);
      }
      if (this.features.actQuant) {
        this.dynQuantT(enc, ST.attn, ST.x_q, ST.scale_x, H, T);
        if (this.features.fuseResidual) {
          this.gemm4AddTW4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.hidden, T, null, L.o.loraKey);
        } else {
          this.gemm4W4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.tmp, T, null, L.o.loraKey);
          this._addInto(enc, ST.hidden, ST.tmp, T * H);
        }
      } else {
        if (this.features.fuseResidual)
          this.gemm4AddT(enc, ST.attn, this.q4[L.o.weight], ST.hidden, T, null, L.o.loraKey);
        else {
          this.gemm4(enc, ST.attn, this.q4[L.o.weight], ST.tmp, T, null, L.o.loraKey);
          this._addInto(enc, ST.hidden, ST.tmp, T * H);
        }
      }
      this.rmsT(enc, ST.hidden, this.bufs[L.postAttentionNorm], ST.normed, T, H);
      if (this.features.actQuant) {
        this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, T);
        this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.gate.weight], ST.tmp, T, null, L.gate.loraKey);
        this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.up.weight], ST.tmp2, T, null, L.up.loraKey);
      } else {
        this.gemm4(enc, ST.normed, this.q4[L.gate.weight], ST.tmp, T, null, L.gate.loraKey);
        this.gemm4(enc, ST.normed, this.q4[L.up.weight], ST.tmp2, T, null, L.up.loraKey);
      }
      this._siluMul(enc, ST.tmp, ST.tmp2, T * c.intermediateSize);
      if (this.features.actQuant) {
        this.dynQuantT(enc, ST.tmp, ST.x_q, ST.scale_x, c.intermediateSize, T);
        if (this.features.fuseResidual) {
          this.gemm4AddTW4A8(
            enc,
            ST.tmp,
            ST.x_q,
            ST.scale_x,
            this.q4[L.down.weight],
            ST.hidden,
            T,
            null,
            L.down.loraKey
          );
        } else {
          this.gemm4W4A8(enc, ST.tmp, ST.x_q, ST.scale_x, this.q4[L.down.weight], ST.normed, T, null, L.down.loraKey);
          this._addInto(enc, ST.hidden, ST.normed, T * H);
        }
      } else {
        if (this.features.fuseResidual)
          this.gemm4AddT(enc, ST.tmp, this.q4[L.down.weight], ST.hidden, T, null, L.down.loraKey);
        else {
          this.gemm4(enc, ST.tmp, this.q4[L.down.weight], ST.normed, T, null, L.down.loraKey);
          this._addInto(enc, ST.hidden, ST.normed, T * H);
        }
      }
    }
    enc.copyBufferToBuffer(ST.hidden, (T - 1) * H * 4, S.hidden, 0, H * 4);
    this.rms(enc, S.hidden, this.bufs[this.plan.finalNorm.name], S.normed, H);
    this.gemv(enc, S.normed, this.q[this.plan.embed.name], S.logits, null, null);
    this.dev.queue.submit([enc.finish()]);
  }
  _prefillChunked(ids, chunkSize) {
    const c = this.cfg, S = this.s, H = c.hiddenSize, hd = c.headDim, kvd = c.numKVHeads * hd;
    const T = ids.length;
    this._ensurePrefillScratch(Math.min(chunkSize, T), this._activeMaxLoraRank(), T);
    const ST = this.sT;
    this._resetUni();
    this.dev.queue.writeBuffer(ST.ids, 0, new Uint32Array(ids));
    const enc = this.dev.createCommandEncoder();
    const e = this.q[this.plan.embed.name];
    for (let off = 0; off < T; off += chunkSize) {
      const end = Math.min(T, off + chunkSize);
      const CT = end - off;
      this._dispatch(
        enc,
        this.pipes.embedT,
        this._bg(this.pipes.embedT, [e.w, e.scale, ST.hidden, ST.ids]),
        Math.min(Math.ceil(CT * H / 256), 65535),
        1,
        "embedT",
        new Uint32Array([CT, H, off, 0])
      );
      for (let i = 0; i < c.numLayers; i++) {
        const L = this.plan.layers[i];
        this.rmsT(enc, ST.hidden, this.bufs[L.inputNorm], ST.normed, CT, H);
        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, CT);
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.q.weight],
            ST.q,
            CT,
            this.bufs[L.q.bias],
            L.q.loraKey
          );
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.k.weight],
            ST.k,
            CT,
            this.bufs[L.k.bias],
            L.k.loraKey
          );
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.v.weight],
            ST.v,
            CT,
            this.bufs[L.v.bias],
            L.v.loraKey
          );
        } else {
          this.gemm4(enc, ST.normed, this.q4[L.q.weight], ST.q, CT, this.bufs[L.q.bias], L.q.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.k.weight], ST.k, CT, this.bufs[L.k.bias], L.k.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.v.weight], ST.v, CT, this.bufs[L.v.bias], L.v.loraKey);
        }
        this.ropeT(enc, ST.q, CT, c.numHeads, off);
        this.ropeT(enc, ST.k, CT, c.numKVHeads, off);
        if (this.features.pagedAttention) {
          this.writeKvPageBatch(enc, ST.k, ST.v, this.kc[i], this.vc[i], CT, off, i);
        } else {
          enc.copyBufferToBuffer(ST.k, 0, this.kc[i], off * kvd * 4, CT * kvd * 4);
          enc.copyBufferToBuffer(ST.v, 0, this.vc[i], off * kvd * 4, CT * kvd * 4);
        }
        if (this.features.pagedAttention) {
          this.attnPrefillPaged(enc, ST.q, this.kc[i], this.vc[i], ST.attn, CT, off, end);
        } else {
          this.attnPrefill(enc, ST.q, this.kc[i], this.vc[i], ST.attn, CT, off, end);
        }
        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.attn, ST.x_q, ST.scale_x, H, CT);
          if (this.features.fuseResidual) {
            this.gemm4AddTW4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.hidden, CT, null, L.o.loraKey);
          } else {
            this.gemm4W4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.tmp, CT, null, L.o.loraKey);
            this._addInto(enc, ST.hidden, ST.tmp, CT * H);
          }
        } else {
          if (this.features.fuseResidual)
            this.gemm4AddT(enc, ST.attn, this.q4[L.o.weight], ST.hidden, CT, null, L.o.loraKey);
          else {
            this.gemm4(enc, ST.attn, this.q4[L.o.weight], ST.tmp, CT, null, L.o.loraKey);
            this._addInto(enc, ST.hidden, ST.tmp, CT * H);
          }
        }
        this.rmsT(enc, ST.hidden, this.bufs[L.postAttentionNorm], ST.normed, CT, H);
        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, CT);
          this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.gate.weight], ST.tmp, CT, null, L.gate.loraKey);
          this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.up.weight], ST.tmp2, CT, null, L.up.loraKey);
        } else {
          this.gemm4(enc, ST.normed, this.q4[L.gate.weight], ST.tmp, CT, null, L.gate.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.up.weight], ST.tmp2, CT, null, L.up.loraKey);
        }
        this._siluMul(enc, ST.tmp, ST.tmp2, CT * c.intermediateSize);
        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.tmp, ST.x_q, ST.scale_x, c.intermediateSize, CT);
          if (this.features.fuseResidual) {
            this.gemm4AddTW4A8(
              enc,
              ST.tmp,
              ST.x_q,
              ST.scale_x,
              this.q4[L.down.weight],
              ST.hidden,
              CT,
              null,
              L.down.loraKey
            );
          } else {
            this.gemm4W4A8(
              enc,
              ST.tmp,
              ST.x_q,
              ST.scale_x,
              this.q4[L.down.weight],
              ST.normed,
              CT,
              null,
              L.down.loraKey
            );
            this._addInto(enc, ST.hidden, ST.normed, CT * H);
          }
        } else {
          if (this.features.fuseResidual)
            this.gemm4AddT(enc, ST.tmp, this.q4[L.down.weight], ST.hidden, CT, null, L.down.loraKey);
          else {
            this.gemm4(enc, ST.tmp, this.q4[L.down.weight], ST.normed, CT, null, L.down.loraKey);
            this._addInto(enc, ST.hidden, ST.normed, CT * H);
          }
        }
      }
      if (end === T) {
        enc.copyBufferToBuffer(ST.hidden, (CT - 1) * H * 4, S.hidden, 0, H * 4);
      }
    }
    this.rms(enc, S.hidden, this.bufs[this.plan.finalNorm.name], S.normed, H);
    this.gemv(enc, S.normed, this.q[this.plan.embed.name], S.logits, null, null);
    this.dev.queue.submit([enc.finish()]);
  }
  async speculativeDecode(draftModel, promptIds, maxNewTokens, onToken) {
    await this.prefillBatch(promptIds);
    await draftModel.prefillBatch(promptIds);
    let currentPos = promptIds.length;
    const generatedIds = [];
    let nextToken = await this.argmaxLogits();
    generatedIds.push(nextToken);
    if (onToken) onToken(nextToken);
    draftModel.dev.queue.writeBuffer(draftModel.s.amax, 0, new Uint32Array([nextToken]));
    this.dev.queue.writeBuffer(this.s.amax, 0, new Uint32Array([nextToken]));
    const gamma = 4;
    while (generatedIds.length < maxNewTokens) {
      const draftCandidates = await draftModel.decodeBatch(currentPos, gamma);
      if (draftCandidates.length === 0) break;
      const T = draftCandidates.length;
      this._resetUni();
      this._ensurePrefillScratch(T, this._activeMaxLoraRank());
      const ST = this.sT;
      const c = this.cfg, H = c.hiddenSize, kvd = c.numKVHeads * c.headDim;
      this.dev.queue.writeBuffer(ST.ids, 0, new Uint32Array(draftCandidates));
      const enc = this.dev.createCommandEncoder();
      const e = this.q[this.plan.embed.name];
      const embedUni = new Uint32Array([T, H, 0, 0]);
      this._dispatch(
        enc,
        this.pipes.embedT,
        this._bg(this.pipes.embedT, [e.w, e.scale, ST.hidden, ST.ids]),
        Math.min(Math.ceil(T * H / 256), 65535),
        1,
        "embedT",
        embedUni
      );
      for (let i = 0; i < c.numLayers; i++) {
        const L = this.plan.layers[i];
        this.rmsT(enc, ST.hidden, this.bufs[L.inputNorm], ST.normed, T, H);
        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, T);
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.q.weight],
            ST.q,
            T,
            this.bufs[L.q.bias],
            L.q.loraKey
          );
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.k.weight],
            ST.k,
            T,
            this.bufs[L.k.bias],
            L.k.loraKey
          );
          this.gemm4W4A8(
            enc,
            ST.normed,
            ST.x_q,
            ST.scale_x,
            this.q4[L.v.weight],
            ST.v,
            T,
            this.bufs[L.v.bias],
            L.v.loraKey
          );
        } else {
          this.gemm4(enc, ST.normed, this.q4[L.q.weight], ST.q, T, this.bufs[L.q.bias], L.q.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.k.weight], ST.k, T, this.bufs[L.k.bias], L.k.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.v.weight], ST.v, T, this.bufs[L.v.bias], L.v.loraKey);
        }
        this.ropeT(enc, ST.q, T, c.numHeads, currentPos);
        this.ropeT(enc, ST.k, T, c.numKVHeads, currentPos);
        if (this.features.pagedAttention) {
          this.writeKvPageBatch(enc, ST.k, ST.v, this.kc[i], this.vc[i], T, currentPos, i);
        } else {
          enc.copyBufferToBuffer(ST.k, 0, this.kc[i], currentPos * kvd * 4, T * kvd * 4);
          enc.copyBufferToBuffer(ST.v, 0, this.vc[i], currentPos * kvd * 4, T * kvd * 4);
        }
        if (this.features.pagedAttention) {
          this.attnPrefillPaged(enc, ST.q, this.kc[i], this.vc[i], ST.attn, T, currentPos, currentPos + T);
        } else {
          this.attnPrefill(enc, ST.q, this.kc[i], this.vc[i], ST.attn, T, currentPos, currentPos + T);
        }
        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.attn, ST.x_q, ST.scale_x, H, T);
          if (this.features.fuseResidual) {
            this.gemm4AddTW4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.hidden, T, null, L.o.loraKey);
          } else {
            this.gemm4W4A8(enc, ST.attn, ST.x_q, ST.scale_x, this.q4[L.o.weight], ST.tmp, T, null, L.o.loraKey);
            this._addInto(enc, ST.hidden, ST.tmp, T * H);
          }
        } else {
          if (this.features.fuseResidual)
            this.gemm4AddT(enc, ST.attn, this.q4[L.o.weight], ST.hidden, T, null, L.o.loraKey);
          else {
            this.gemm4(enc, ST.attn, this.q4[L.o.weight], ST.tmp, T, null, L.o.loraKey);
            this._addInto(enc, ST.hidden, ST.tmp, T * H);
          }
        }
        this.rmsT(enc, ST.hidden, this.bufs[L.postAttentionNorm], ST.normed, T, H);
        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.normed, ST.x_q, ST.scale_x, H, T);
          this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.gate.weight], ST.tmp, T, null, L.gate.loraKey);
          this.gemm4W4A8(enc, ST.normed, ST.x_q, ST.scale_x, this.q4[L.up.weight], ST.tmp2, T, null, L.up.loraKey);
        } else {
          this.gemm4(enc, ST.normed, this.q4[L.gate.weight], ST.tmp, T, null, L.gate.loraKey);
          this.gemm4(enc, ST.normed, this.q4[L.up.weight], ST.tmp2, T, null, L.up.loraKey);
        }
        this._siluMul(enc, ST.tmp, ST.tmp2, T * c.intermediateSize);
        if (this.features.actQuant) {
          this.dynQuantT(enc, ST.tmp, ST.x_q, ST.scale_x, c.intermediateSize, T);
          if (this.features.fuseResidual) {
            this.gemm4AddTW4A8(
              enc,
              ST.tmp,
              ST.x_q,
              ST.scale_x,
              this.q4[L.down.weight],
              ST.hidden,
              T,
              null,
              L.down.loraKey
            );
          } else {
            this.gemm4W4A8(enc, ST.tmp, ST.x_q, ST.scale_x, this.q4[L.down.weight], ST.normed, T, null, L.down.loraKey);
            this._addInto(enc, ST.hidden, ST.normed, T * H);
          }
        } else {
          if (this.features.fuseResidual)
            this.gemm4AddT(enc, ST.tmp, this.q4[L.down.weight], ST.hidden, T, null, L.down.loraKey);
          else {
            this.gemm4(enc, ST.tmp, this.q4[L.down.weight], ST.normed, T, null, L.down.loraKey);
            this._addInto(enc, ST.hidden, ST.normed, T * H);
          }
        }
      }
      if (!this.s.logitsT || this.sTcap < T) {
        if (this.s.logitsT) this.s.logitsT.destroy();
        this.s.logitsT = this._buf(T * c.vocabSize * 4);
        if (this.logitsTRead) this.logitsTRead.destroy();
        this.logitsTRead = this._buf(T * c.vocabSize * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      }
      for (let t = 0; t < T; t++) {
        enc.copyBufferToBuffer(ST.hidden, t * H * 4, this.s.hidden, 0, H * 4);
        this.rms(enc, this.s.hidden, this.bufs[this.plan.finalNorm.name], this.s.normed, H);
        this.gemv(enc, this.s.normed, this.q[this.plan.embed.name], this.s.logits, null, null);
        enc.copyBufferToBuffer(this.s.logits, 0, this.s.logitsT, t * c.vocabSize * 4, c.vocabSize * 4);
      }
      enc.copyBufferToBuffer(this.s.logitsT, 0, this.logitsTRead, 0, T * c.vocabSize * 4);
      this.dev.queue.submit([enc.finish()]);
      await this.logitsTRead.mapAsync(GPUMapMode.READ);
      const logitsArray = new Float32Array(this.logitsTRead.getMappedRange());
      let acceptedCount = 0;
      let targetToken = 0;
      for (let t = 0; t < T; t++) {
        let maxVal = -1e30;
        let argmaxId = 0;
        const offset = t * c.vocabSize;
        for (let v = 0; v < c.vocabSize; v++) {
          const l = logitsArray[offset + v];
          if (l > maxVal) {
            maxVal = l;
            argmaxId = v;
          }
        }
        targetToken = argmaxId;
        if (t < T) {
          if (draftCandidates[t] === targetToken) {
            acceptedCount++;
          } else {
            break;
          }
        }
      }
      this.logitsTRead.unmap();
      for (let a = 0; a < acceptedCount; a++) {
        generatedIds.push(draftCandidates[a]);
        if (onToken) onToken(draftCandidates[a]);
      }
      generatedIds.push(targetToken);
      if (onToken) onToken(targetToken);
      const nextPos = currentPos + acceptedCount + 1;
      this.dev.queue.writeBuffer(this.s.amax, 0, new Uint32Array([targetToken]));
      draftModel.dev.queue.writeBuffer(draftModel.s.amax, 0, new Uint32Array([targetToken]));
      if (this.features.pagedAttention) {
        this.pam.ensureBlocks(0, nextPos);
      }
      currentPos = nextPos;
    }
    return generatedIds;
  }
  // Simple high-level generation helper (Phase 5 wiring).
  // If opts.sample === true, uses the GPU sampler (sampleToken) with given temp;
  // otherwise falls back to argmax (greedy).
  // This makes sampleToken part of the real generation path.
  async generate(promptIds, maxNewTokens = 32, opts = {}) {
    const doSample = !!opts.sample;
    const temp = opts.temp != null && opts.temp > 0 ? opts.temp : 1;
    await this.prefillBatch(promptIds);
    const generatedIds = [];
    let pos = promptIds.length;
    let next = doSample ? await this.sampleToken(temp) : await this.argmaxLogits();
    generatedIds.push(next);
    if (opts.onToken) opts.onToken(next);
    this.dev.queue.writeBuffer(this.s.amax, 0, new Uint32Array([next]));
    while (generatedIds.length < maxNewTokens) {
      this._resetUni();
      const enc = this.dev.createCommandEncoder();
      this.embedFromBuf(enc);
      this.step(enc, 0, pos);
      this.dev.queue.submit([enc.finish()]);
      next = doSample ? await this.sampleToken(temp) : await this.argmaxLogits();
      generatedIds.push(next);
      if (opts.onToken) opts.onToken(next);
      this.dev.queue.writeBuffer(this.s.amax, 0, new Uint32Array([next]));
      pos += 1;
    }
    return generatedIds;
  }
  setupDebugCapture(T, K, rank, N) {
    this.debugCapture = true;
    this.debugT = T;
    this.debugK = K;
    this.debugRank = rank;
    this.debugN = N;
    this.debugStep = 0;
    this.debugCaptured = false;
    this.debugBufs = {
      xSeq: this._buf(T * K * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      dSeq: this._buf(T * rank * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      ySeq: this._buf(T * N * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      xBat: this._buf(T * K * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      dBat: this._buf(T * rank * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      yBat: this._buf(T * N * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    };
  }
  async readDebugCapture() {
    this.debugCapture = false;
    const bufs = this.debugBufs;
    if (!bufs) return null;
    await Promise.all([
      bufs.xSeq.mapAsync(GPUMapMode.READ),
      bufs.dSeq.mapAsync(GPUMapMode.READ),
      bufs.ySeq.mapAsync(GPUMapMode.READ),
      bufs.xBat.mapAsync(GPUMapMode.READ),
      bufs.dBat.mapAsync(GPUMapMode.READ),
      bufs.yBat.mapAsync(GPUMapMode.READ)
    ]);
    const res = {
      xSeq: new Float32Array(bufs.xSeq.getMappedRange()).slice(),
      dSeq: new Float32Array(bufs.dSeq.getMappedRange()).slice(),
      ySeq: new Float32Array(bufs.ySeq.getMappedRange()).slice(),
      xBat: new Float32Array(bufs.xBat.getMappedRange()).slice(),
      dBat: new Float32Array(bufs.dBat.getMappedRange()).slice(),
      yBat: new Float32Array(bufs.yBat.getMappedRange()).slice()
    };
    bufs.xSeq.unmap();
    bufs.xSeq.destroy();
    bufs.dSeq.unmap();
    bufs.dSeq.destroy();
    bufs.ySeq.unmap();
    bufs.ySeq.destroy();
    bufs.xBat.unmap();
    bufs.xBat.destroy();
    bufs.dBat.unmap();
    bufs.dBat.destroy();
    bufs.yBat.unmap();
    bufs.yBat.destroy();
    this.debugBufs = null;
    return res;
  }
};
var PagedAttentionManager = class {
  static {
    __name(this, "PagedAttentionManager");
  }
  constructor(maxCtx, pageSize = 16) {
    this.pageSize = pageSize;
    this.maxCtx = maxCtx;
    this.maxBlocksPerSeq = Math.ceil(maxCtx / pageSize);
    this.freeBlocks = [];
    this.seqBlocks = /* @__PURE__ */ new Map();
    const totalBlocks = this.maxBlocksPerSeq * 4;
    for (let i = 0; i < totalBlocks; i++) {
      this.freeBlocks.push(i);
    }
  }
  allocateSeq(seqId) {
    this.seqBlocks.set(seqId, []);
  }
  freeSeq(seqId) {
    const blocks = this.seqBlocks.get(seqId) || [];
    this.freeBlocks.push(...blocks);
    this.seqBlocks.delete(seqId);
  }
  ensureBlocks(seqId, numTokens) {
    const neededBlocks = Math.ceil(numTokens / this.pageSize);
    const blocks = this.seqBlocks.get(seqId);
    if (!blocks) throw new Error(`Sequence ${seqId} not allocated`);
    while (blocks.length < neededBlocks) {
      if (this.freeBlocks.length === 0) {
        const newBlock = blocks.length + 1e3;
        this.freeBlocks.push(newBlock);
      }
      blocks.push(this.freeBlocks.pop());
    }
    return blocks;
  }
  getBlockTableArray(seqId) {
    const blocks = this.seqBlocks.get(seqId) || [];
    const arr = new Uint32Array(this.maxBlocksPerSeq);
    arr.set(blocks);
    return arr;
  }
};

// src/services/device_service.js
async function initWebGPUDevice({ log: log2 = /* @__PURE__ */ __name(() => {
}, "log") } = {}) {
  log2("requesting WebGPU device\u2026");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter (use a WebGPU-capable browser)");
  if (!navigator.gpu.wgslLanguageFeatures?.has("immediate_address_space"))
    throw new Error("WGSL immediate_address_space is not available (upgrade to Chrome 149+)");
  if (!adapter.features.has("subgroups"))
    throw new Error(
      'GPU lacks the required "subgroups" feature. The current fast WGSL kernels require subgroups and no fallback kernel set is bundled.'
    );
  const hasSubgroupId = !!navigator.gpu.wgslLanguageFeatures?.has("subgroup_id");
  const hasLinearIndexing = !!navigator.gpu.wgslLanguageFeatures?.has("linear_indexing");
  const hasF16 = adapter.features.has("shader-f16");
  const hasTimestamp = adapter.features.has("timestamp-query");
  const reqFeatures = ["subgroups"];
  if (adapter.features.has("shader-f16")) reqFeatures.push("shader-f16");
  if (hasTimestamp) reqFeatures.push("timestamp-query");
  const dev = await adapter.requestDevice({
    requiredFeatures: reqFeatures,
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage
    }
  });
  dev.addEventListener?.("uncapturederror", (e) => console.error("GPUERR", e.error.message));
  log2(`WebGPU ready. maxBuffer=${(Number(adapter.limits.maxBufferSize) / 1e9).toFixed(2)}GB subgroupId=${hasSubgroupId} linearIdx=${hasLinearIndexing} f16=${hasF16} tsQuery=${hasTimestamp}`);
  return dev;
}
__name(initWebGPUDevice, "initWebGPUDevice");

// src/services/prompt_formatter.js
function chatML(messages) {
  let s = messages[0]?.role === "system" ? "" : "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
  for (const m of messages) s += `<|im_start|>${m.role}
${m.content}<|im_end|>
`;
  return s + "<|im_start|>assistant\n";
}
__name(chatML, "chatML");
function formatMessages(tokenizer, messages) {
  try {
    return tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
  } catch {
    return chatML(messages);
  }
}
__name(formatMessages, "formatMessages");

// src/services/model_session.js
async function buildTokenizer(reader) {
  const tj = JSON.parse(await reader.text("tokenizer.json"));
  const tc = JSON.parse(await reader.text("tokenizer_config.json"));
  const { PreTrainedTokenizer } = await import("@huggingface/transformers");
  return new PreTrainedTokenizer(tj, tc);
}
__name(buildTokenizer, "buildTokenizer");
function randomUnit() {
  if (globalThis.crypto?.getRandomValues) {
    const u = new Uint32Array(1);
    globalThis.crypto.getRandomValues(u);
    return u[0] / 4294967296;
  }
  return Math.random();
}
__name(randomUnit, "randomUnit");
function sampleTopK(candidates, { temperature, topP = 1 }) {
  if (!temperature || temperature <= 0) return candidates[0]?.id ?? 0;
  const best = candidates[0]?.logit ?? 0;
  const weighted = candidates.map((c2) => ({ id: c2.id, w: Math.exp((c2.logit - best) / temperature) }));
  let sum = weighted.reduce((a, c2) => a + c2.w, 0);
  if (topP > 0 && topP < 1 && weighted.length > 1 && sum > 0) {
    let csum = 0, keep = 0;
    for (; keep < weighted.length; keep++) {
      csum += weighted[keep].w / sum;
      if (csum >= topP) {
        keep++;
        break;
      }
    }
    weighted.length = Math.max(1, keep);
    sum = weighted.reduce((a, c2) => a + c2.w, 0);
  }
  let r = randomUnit() * sum, c = 0;
  for (const item of weighted) {
    c += item.w;
    if (r <= c) return item.id;
  }
  return weighted[weighted.length - 1]?.id ?? candidates[0]?.id ?? 0;
}
__name(sampleTopK, "sampleTopK");
var ModelSession = class {
  static {
    __name(this, "ModelSession");
  }
  constructor({ cfg = QWEN25_3B, log: log2 = /* @__PURE__ */ __name(() => {
  }, "log"), runtimeOptions = {} } = {}) {
    this.cfg = cfg;
    this.log = log2;
    this.runtimeOptions = { decodeBatchSize: "auto", samplingTopK: 40, ...runtimeOptions };
    this.dev = null;
    this.rt = null;
    this.tokenizer = null;
  }
  async loadWith(reader, label) {
    this.dev = await initWebGPUDevice({ log: this.log });
    this.log(`loading tokenizer from ${label}\u2026`);
    this.tokenizer = await buildTokenizer(reader);
    this.log(`tokenizer loaded. streaming + quantizing weights (int4) from ${label}\u2026`);
    const t0 = performance.now();
    this.rt = new QwenWGPU(this.dev, this.cfg, this.runtimeOptions);
    await this.rt.build(reader, (msg, frac) => this.log(`weights: ${msg} ${(frac * 100).toFixed(0)}%`));
    window.__rt = this.rt;
    window.__tokenizer = this.tokenizer;
    const tuning = this.rt.decodeBatchTuning;
    const tuned = tuning ? ` decodeBatch=${tuning.selected} (${tuning.reason})` : "";
    this.log(
      `READY in ${((performance.now() - t0) / 1e3).toFixed(1)}s \u2014 base loaded once; adapters hot-swap live.${tuned}`
    );
    return this;
  }
  async readLogits() {
    const n = this.cfg.vocabSize;
    const rb = this.dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.rt.s.logits, 0, rb, 0, n * 4);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rb.getMappedRange()).slice();
    rb.unmap();
    rb.destroy();
    return a;
  }
  async sampleNextToken({ temperature, topK = this.rt.samplingTopK, topP = 1 } = {}) {
    return sampleTopK(await this.rt.topKLogits(topK), { temperature, topP });
  }
  async *generate(messages, { maxTokens = 1024, temperature = 0, topK, topP = 1, stopIds = [151645, 151643] } = {}) {
    const rt = this.rt, tokenizer = this.tokenizer;
    const ids = tokenizer.encode(formatMessages(tokenizer, messages));
    if (ids.length <= rt.maxPrefillT) rt.prefillBatch(ids);
    else for (let p = 0; p < ids.length; p++) rt.token(ids[p], p);
    let pos = ids.length;
    const emit = /* @__PURE__ */ __name((id) => tokenizer.decode([id], { skip_special_tokens: true }), "emit");
    if (temperature > 0) {
      let next = await this.sampleNextToken({ temperature, topK, topP });
      for (let step = 0; step < maxTokens; step++) {
        if (stopIds.includes(next)) break;
        const d = emit(next);
        if (d) yield d;
        rt.token(next, pos);
        pos++;
        next = await this.sampleNextToken({ temperature, topK, topP });
      }
      return;
    }
    const first = await rt.argmaxLogits();
    if (stopIds.includes(first)) return;
    {
      const d = emit(first);
      if (d) yield d;
    }
    let emitted = 1;
    while (emitted < maxTokens && pos < rt.maxCtx) {
      const K = rt.greedyBatchSizeFor({ emitted, remaining: maxTokens - emitted, pos });
      const batch = await rt.decodeGreedyBatch(pos, K);
      pos += batch.length;
      let stop = false;
      for (const id of batch) {
        if (stopIds.includes(id)) {
          stop = true;
          break;
        }
        const d = emit(id);
        if (d) yield d;
        emitted++;
        if (emitted >= maxTokens) {
          stop = true;
          break;
        }
      }
      if (stop) break;
    }
  }
};

// src/qwgpu/backward_kernels.js
var GEMM_DX_INT4 = `
requires immediate_address_space;
struct Meta { T:u32, N:u32, K:u32, gpr:u32 };
@group(0) @binding(0) var<storage,read> dY: array<f32>;       // [T][N]
@group(0) @binding(1) var<storage,read> W: array<u32>;        // [N][K/8] int4
@group(0) @binding(2) var<storage,read> scaleW: array<f32>;   // [N][gpr]
@group(0) @binding(3) var<storage,read_write> dX: array<f32>; // [T][K]
var<immediate> m: Meta;
fn deq4(n: u32, k: u32, K8: u32) -> f32 {
  let word = W[n*K8 + (k >> 3u)];
  let shift = (k & 7u) * 4u;
  let nib = i32(word << (28u - shift)) >> 28u;
  return f32(nib) * scaleW[n*m.gpr + (k >> 7u)];
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.T * m.K; let stride = nwg.x * 256u; let K8 = m.K / 8u;
  for (var i = gid.x; i < total; i = i + stride) {
    let t = i / m.K; let k = i % m.K;
    var acc = 0.0;
    let yb = t * m.N;
    for (var n = 0u; n < m.N; n = n + 1u) { acc = acc + dY[yb + n] * deq4(n, k, K8); }
    dX[i] = dX[i] + acc;
  }
}`;
var LORA_DD = `
requires immediate_address_space;
struct Meta { T:u32, N:u32, rank:u32, p:u32, scale:f32, f0:f32, f1:f32, f2:f32 };
@group(0) @binding(0) var<storage,read> dY: array<f32>;       // [T][N]
@group(0) @binding(1) var<storage,read> B: array<f32>;        // [rank][N]
@group(0) @binding(2) var<storage,read_write> dD: array<f32>; // [T][rank]
var<immediate> m: Meta;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wid.x; let t = idx / m.rank; let r = idx % m.rank; let tid = lid.x;
  if (t >= m.T) { return; }
  var s = 0.0; let yb = t*m.N; let bb = r*m.N;
  for (var n = tid; n < m.N; n = n + 256u) { s = s + dY[yb + n] * B[bb + n]; }
  part[tid] = s; workgroupBarrier();
  for (var st = 128u; st > 0u; st = st/2u) { if (tid < st) { part[tid] = part[tid] + part[tid+st]; } workgroupBarrier(); }
  if (tid == 0u) { dD[t*m.rank + r] = m.scale * part[0]; }
}`;
var LORA_GRAD_A = `
requires immediate_address_space;
struct Meta { T:u32, K:u32, rank:u32, p:u32 };
@group(0) @binding(0) var<storage,read> dD: array<f32>;       // [T][rank]
@group(0) @binding(1) var<storage,read> X: array<f32>;        // [T][K]
@group(0) @binding(2) var<storage,read_write> dA: array<f32>; // [rank][K]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.rank * m.K; let stride = nwg.x * 256u;
  for (var i = gid.x; i < total; i = i + stride) {
    let r = i / m.K; let k = i % m.K;
    var acc = 0.0;
    for (var t = 0u; t < m.T; t = t + 1u) { acc = acc + dD[t*m.rank + r] * X[t*m.K + k]; }
    dA[i] = dA[i] + acc;
  }
}`;
var LORA_GRAD_B = `
requires immediate_address_space;
struct Meta { T:u32, N:u32, rank:u32, p:u32, scale:f32, f0:f32, f1:f32, f2:f32 };
@group(0) @binding(0) var<storage,read> D: array<f32>;        // [T][rank]
@group(0) @binding(1) var<storage,read> dY: array<f32>;       // [T][N]
@group(0) @binding(2) var<storage,read_write> dB: array<f32>; // [rank][N]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.rank * m.N; let stride = nwg.x * 256u;
  for (var i = gid.x; i < total; i = i + stride) {
    let r = i / m.N; let n = i % m.N;
    var acc = 0.0;
    for (var t = 0u; t < m.T; t = t + 1u) { acc = acc + D[t*m.rank + r] * dY[t*m.N + n]; }
    dB[i] = dB[i] + m.scale * acc;
  }
}`;
var LORA_DX_ADD = `
requires immediate_address_space;
struct Meta { T:u32, K:u32, rank:u32, p:u32 };
@group(0) @binding(0) var<storage,read> dD: array<f32>;       // [T][rank]
@group(0) @binding(1) var<storage,read> A: array<f32>;        // [rank][K]
@group(0) @binding(2) var<storage,read_write> dX: array<f32>; // [T][K]
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.T * m.K; let stride = nwg.x * 256u;
  for (var i = gid.x; i < total; i = i + stride) {
    let t = i / m.K; let k = i % m.K;
    var acc = 0.0;
    for (var r = 0u; r < m.rank; r = r + 1u) { acc = acc + dD[t*m.rank + r] * A[r*m.K + k]; }
    dX[i] = dX[i] + acc;
  }
}`;
var RMSNORM_BWD_T = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;        // [T][K]
@group(0) @binding(1) var<storage,read> g: array<f32>;        // [K]
@group(0) @binding(2) var<storage,read> dy: array<f32>;       // [T][K]
@group(0) @binding(3) var<storage,read_write> dx: array<f32>; // [T][K]
var<immediate> m: vec2<f32>;   // K, eps
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let K = u32(m.x); let base = wid.x * K;
  // sum of squares for inv
  var ss = 0.0;
  for (var k = tid; k < K; k = k + WG) { let v = x[base+k]; ss = ss + v*v; }
  red[tid] = ss; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); }
  let ms = red[0] / m.x;
  let inv = inverseSqrt(ms + m.y);
  workgroupBarrier();
  // c = sum dy*g*x
  var cc = 0.0;
  for (var k = tid; k < K; k = k + WG) { cc = cc + dy[base+k]*g[k]*x[base+k]; }
  red[tid] = cc; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); }
  let c = red[0];
  let inv3overK = inv*inv*inv / m.x;
  for (var k = tid; k < K; k = k + WG) {
    dx[base+k] = inv*g[k]*dy[base+k] - inv3overK * x[base+k] * c;
  }
}`;
var SWIGLU_BWD = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> gate: array<f32>;
@group(0) @binding(1) var<storage,read> up: array<f32>;
@group(0) @binding(2) var<storage,read> dOut: array<f32>;
@group(0) @binding(3) var<storage,read_write> dGate: array<f32>;
@group(0) @binding(4) var<storage,read_write> dUp: array<f32>;
var<immediate> n: u32;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * WG;
  for (var i = gid.x; i < n; i = i + stride) {
    let z = gate[i]; let sig = 1.0/(1.0+exp(-z)); let sl = z*sig;
    let d = dOut[i];
    dUp[i] = d * sl;
    dGate[i] = d * up[i] * (sig * (1.0 + z*(1.0 - sig)));
  }
}`;
var ROPE_BWD_T = `
requires immediate_address_space;
@group(0) @binding(0) var<storage,read_write> dx: array<f32>;   // [T][nHeads*headDim] gradient
@group(0) @binding(1) var<storage,read> cosT: array<f32>;
@group(0) @binding(2) var<storage,read> sinT: array<f32>;
var<immediate> m: vec4<u32>;   // nHeads, headDim, T, pos0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x; let H = m.x; let D = m.y; let T = m.z; let pos0 = m.w; let half = D/2u;
  let perRow = H*half; if (g >= T*perRow) { return; }
  let row = g / perRow; let r = g % perRow; let h = r / half; let j = r % half;
  let rb = row*H*D; let lo = rb + h*D + j; let hi = lo + half; let off = (pos0+row)*D + j;
  let c = cosT[off]; let s = sinT[off];
  let dl = dx[lo]; let dh = dx[hi];
  dx[lo] = c*dl + s*dh;
  dx[hi] = -s*dl + c*dh;
}`;
var ATTN_BWD_STATS = `
requires immediate_address_space;
override WG: u32 = 128u;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;     // [T][nHeads*hd]
@group(0) @binding(1) var<storage,read> kc: array<f32>;    // [T][nKV*hd]
@group(0) @binding(2) var<storage,read> o: array<f32>;     // [T][nHeads*hd] attn output
@group(0) @binding(3) var<storage,read> doo: array<f32>;   // [T][nHeads*hd] grad of attn output
@group(0) @binding(4) var<storage,read_write> lse: array<f32>;   // [nHeads*T]
@group(0) @binding(5) var<storage,read_write> delta: array<f32>; // [nHeads*T]
var<immediate> m: Meta;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x; let t = wid.y; let tid = lid.x;
  let hd = m.hd; let nKV = m.nKV; let kvh = h / (m.nHeads / nKV);
  let qb = t*m.nHeads*hd + h*hd; let kvstride = nKV*hd; let hoff = kvh*hd;
  let scl = 1.0 / sqrt(f32(hd));
  // running max
  var lmax = -1e30;
  for (var j = tid; j <= t; j = j + WG) {
    var dot = 0.0; let kb = j*kvstride + hoff;
    for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qb+d]*kc[kb+d]; }
    lmax = max(lmax, dot*scl);
  }
  red[tid] = lmax; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = max(red[tid], red[tid+s]); } workgroupBarrier(); }
  let M = red[0];
  workgroupBarrier();
  var lsum = 0.0;
  for (var j = tid; j <= t; j = j + WG) {
    var dot = 0.0; let kb = j*kvstride + hoff;
    for (var d = 0u; d < hd; d = d + 1u) { dot = dot + q[qb+d]*kc[kb+d]; }
    lsum = lsum + exp(dot*scl - M);
  }
  red[tid] = lsum; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); }
  // delta
  var dl = 0.0;
  for (var d = tid; d < hd; d = d + WG) { dl = dl + doo[qb+d]*o[qb+d]; }
  // reuse red after sum captured
  let Z = red[0];
  workgroupBarrier();
  red[tid] = dl; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); }
  if (tid == 0u) { lse[h*m.T + t] = M + log(Z); delta[h*m.T + t] = red[0]; }
}`;
var ATTN_BWD_DQ = `
requires immediate_address_space;
override WG: u32 = 128u;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read> doo: array<f32>;
@group(0) @binding(4) var<storage,read> lse: array<f32>;
@group(0) @binding(5) var<storage,read> delta: array<f32>;
@group(0) @binding(6) var<storage,read_write> dq: array<f32>;
var<immediate> m: Meta;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x; let t = wid.y; let d = lid.x;
  let hd = m.hd; let nKV = m.nKV; let kvh = h / (m.nHeads / nKV);
  let qb = t*m.nHeads*hd + h*hd; let kvstride = nKV*hd; let hoff = kvh*hd;
  let scl = 1.0 / sqrt(f32(hd));
  let lse_t = lse[h*m.T + t]; let delta_t = delta[h*m.T + t];
  // Guard every storage read behind (d < hd): WGSL select() is eager and would
  // still evaluate the buffer load for inactive lanes (OOB when hd < WG). Barriers
  // stay at uniform control flow so the reductions remain valid.
  let inHd = d < hd;
  var acc = 0.0;
  for (var j = 0u; j <= t; j = j + 1u) {
    let kb = j*kvstride + hoff;
    // s = scl * dot(q, k_j)
    var sv = 0.0; if (inHd) { sv = q[qb+d] * kc[kb+d]; }
    red[d] = sv; workgroupBarrier();
    for (var s = WG/2u; s > 0u; s = s/2u) { if (d < s) { red[d] = red[d] + red[d+s]; } workgroupBarrier(); }
    let sval = red[0] * scl;
    workgroupBarrier();
    // dp = dot(do, v_j)
    var dpv = 0.0; if (inHd) { dpv = doo[qb+d] * vc[kb+d]; }
    red[d] = dpv; workgroupBarrier();
    for (var s = WG/2u; s > 0u; s = s/2u) { if (d < s) { red[d] = red[d] + red[d+s]; } workgroupBarrier(); }
    let dp = red[0];
    workgroupBarrier();
    let p = exp(sval - lse_t);
    let ds = p * (dp - delta_t);
    if (inHd) { acc = acc + ds * kc[kb+d]; }
  }
  if (inHd) { dq[qb+d] = dq[qb+d] + scl * acc; }
}`;
var ATTN_BWD_DKV = `
requires immediate_address_space;
override WG: u32 = 128u;
struct Meta { nHeads:u32, nKV:u32, hd:u32, T:u32 };
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read> doo: array<f32>;
@group(0) @binding(4) var<storage,read> lse: array<f32>;
@group(0) @binding(5) var<storage,read> delta: array<f32>;
@group(0) @binding(6) var<storage,read_write> dk: array<f32>;
@group(0) @binding(7) var<storage,read_write> dv: array<f32>;
var<immediate> m: Meta;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let kvh = wid.x; let j = wid.y; let d = lid.x;
  let hd = m.hd; let nKV = m.nKV; let group = m.nHeads / nKV;
  let kvstride = nKV*hd; let hoff = kvh*hd; let kb = j*kvstride + hoff;
  let scl = 1.0 / sqrt(f32(hd));
  // Guard storage reads behind (d < hd) \u2014 see ATTN_BWD_DQ note on eager select().
  let inHd = d < hd;
  var dkacc = 0.0; var dvacc = 0.0;
  for (var hi = 0u; hi < group; hi = hi + 1u) {
    let h = kvh*group + hi;
    for (var t = j; t < m.T; t = t + 1u) {
      let qb = t*m.nHeads*hd + h*hd;
      var sv = 0.0; if (inHd) { sv = q[qb+d] * kc[kb+d]; }
      red[d] = sv; workgroupBarrier();
      for (var s = WG/2u; s > 0u; s = s/2u) { if (d < s) { red[d] = red[d] + red[d+s]; } workgroupBarrier(); }
      let sval = red[0] * scl;
      workgroupBarrier();
      var dpv = 0.0; if (inHd) { dpv = doo[qb+d] * vc[kb+d]; }
      red[d] = dpv; workgroupBarrier();
      for (var s = WG/2u; s > 0u; s = s/2u) { if (d < s) { red[d] = red[d] + red[d+s]; } workgroupBarrier(); }
      let dp = red[0];
      workgroupBarrier();
      let p = exp(sval - lse[h*m.T + t]);
      let ds = p * (dp - delta[h*m.T + t]);
      if (inHd) {
        dkacc = dkacc + scl * ds * q[qb+d];
        dvacc = dvacc + p * doo[qb+d];
      }
    }
  }
  if (inHd) { dk[kb+d] = dk[kb+d] + dkacc; dv[kb+d] = dv[kb+d] + dvacc; }
}`;
var LOGITS_GEMM_I8 = `
requires immediate_address_space;
struct Meta { T:u32, vocab:u32, K:u32, tOff:u32 };
@group(0) @binding(0) var<storage,read> normed: array<f32>;   // [T][K] (full-seq buffer, offset by tOff)
@group(0) @binding(1) var<storage,read> E: array<u32>;        // [vocab][K/4] int8
@group(0) @binding(2) var<storage,read> scaleE: array<f32>;   // [vocab]
@group(0) @binding(3) var<storage,read_write> logits: array<f32>; // [Tblock][vocab]
var<immediate> m: Meta;
fn sx8(v: u32) -> i32 {
  return i32(v << 24u) >> 24u;
}
fn unpack4xI8(x: u32) -> vec4<i32> {
  return vec4<i32>(
    sx8(x & 0xffu),
    sx8((x >> 8u) & 0xffu),
    sx8((x >> 16u) & 0xffu),
    sx8((x >> 24u) & 0xffu)
  );
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.T * m.vocab; let stride = nwg.x * 256u; let K4 = m.K / 4u;
  for (var i = gid.x; i < total; i = i + stride) {
    let t = i / m.vocab; let v = i % m.vocab;
    let nb = (m.tOff + t) * m.K; let eb = v * K4;
    var acc = 0.0;
    for (var c = 0u; c < K4; c = c + 1u) {
      let p = unpack4xI8(E[eb + c]); let kk = c*4u;
      acc = acc + normed[nb+kk]*f32(p.x) + normed[nb+kk+1u]*f32(p.y)
                + normed[nb+kk+2u]*f32(p.z) + normed[nb+kk+3u]*f32(p.w);
    }
    logits[i] = acc * scaleE[v];
  }
}`;
var CE_SOFTMAX_GRAD = `
requires immediate_address_space;
override WG: u32 = 256u;
struct Meta { vocab:u32, tOff:u32, lossScale:f32, p:u32 };
@group(0) @binding(0) var<storage,read_write> logits: array<f32>; // [bt][vocab] -> dLogits
@group(0) @binding(1) var<storage,read> labels: array<u32>;       // [T] token id (global)
@group(0) @binding(2) var<storage,read> mask: array<f32>;         // [T] 1 train / 0 skip (global)
@group(0) @binding(3) var<storage,read_write> lossOut: array<f32>;// [T] (global)
var<immediate> m: Meta;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lt = wid.x; let tid = lid.x; let base = lt*m.vocab;
  let gt = m.tOff + lt;            // global token index for target/mask/loss
  let mk = mask[gt];
  // max
  var mx = -1e30;
  for (var v = tid; v < m.vocab; v = v + WG) { mx = max(mx, logits[base+v]); }
  red[tid] = mx; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = max(red[tid], red[tid+s]); } workgroupBarrier(); }
  let M = red[0]; workgroupBarrier();
  // sum exp
  var sm = 0.0;
  for (var v = tid; v < m.vocab; v = v + WG) { sm = sm + exp(logits[base+v] - M); }
  red[tid] = sm; workgroupBarrier();
  for (var s = WG/2u; s > 0u; s = s/2u) { if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); }
  let Z = red[0];
  let tgt = labels[gt];
  if (tid == 0u) {
    let ltgt = logits[base + tgt];
    lossOut[gt] = mk * (log(Z) - (ltgt - M));
  }
  // dLogits = mask*lossScale*(p - onehot)
  let invZ = 1.0 / Z; let g = mk * m.lossScale;
  for (var v = tid; v < m.vocab; v = v + WG) {
    var p = exp(logits[base+v] - M) * invZ;
    if (v == tgt) { p = p - 1.0; }
    logits[base+v] = g * p;
  }
}`;
var DHIDDEN_FROM_DLOGITS_I8 = `
requires immediate_address_space;
struct Meta { T:u32, vocab:u32, K:u32, tOff:u32 };
@group(0) @binding(0) var<storage,read> dLogits: array<f32>;  // [Tblock][vocab]
@group(0) @binding(1) var<storage,read> E: array<u32>;        // [vocab][K/4] int8
@group(0) @binding(2) var<storage,read> scaleE: array<f32>;   // [vocab]
@group(0) @binding(3) var<storage,read_write> dHidden: array<f32>; // [T][K] (offset tOff)
var<immediate> m: Meta;
fn sx8(v: u32) -> i32 {
  return i32(v << 24u) >> 24u;
}
fn unpack4xI8(x: u32) -> vec4<i32> {
  return vec4<i32>(
    sx8(x & 0xffu),
    sx8((x >> 8u) & 0xffu),
    sx8((x >> 16u) & 0xffu),
    sx8((x >> 24u) & 0xffu)
  );
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = m.T * m.K; let stride = nwg.x * 256u; let K4 = m.K / 4u;
  for (var i = gid.x; i < total; i = i + stride) {
    let t = i / m.K; let k = i % m.K;
    let lb = t * m.vocab;
    var acc = 0.0;
    let word_idx = k >> 2u; let lane = k & 3u;
    for (var v = 0u; v < m.vocab; v = v + 1u) {
      let p = unpack4xI8(E[v*K4 + word_idx]);
      var b: i32; if (lane==0u){b=p.x;} else if (lane==1u){b=p.y;} else if (lane==2u){b=p.z;} else {b=p.w;}
      acc = acc + dLogits[lb + v] * scaleE[v] * f32(b);
    }
    dHidden[(m.tOff + t)*m.K + k] = dHidden[(m.tOff + t)*m.K + k] + acc;
  }
}`;
var ADAMW_STEP = `
requires immediate_address_space;
struct Meta { n:u32, p:u32, lr:f32, beta1:f32, beta2:f32, eps:f32, wd:f32, gScale:f32, b1c:f32, b2c:f32, f0:f32, f1:f32 };
@group(0) @binding(0) var<storage,read_write> param: array<f32>;
@group(0) @binding(1) var<storage,read> grad: array<f32>;
@group(0) @binding(2) var<storage,read_write> mBuf: array<f32>;
@group(0) @binding(3) var<storage,read_write> vBuf: array<f32>;
var<immediate> m: Meta;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 256u;
  for (var i = gid.x; i < m.n; i = i + stride) {
    let gr = grad[i] * m.gScale;
    let mm = m.beta1 * mBuf[i] + (1.0 - m.beta1) * gr;
    let vv = m.beta2 * vBuf[i] + (1.0 - m.beta2) * gr * gr;
    mBuf[i] = mm; vBuf[i] = vv;
    let mhat = mm / m.b1c; let vhat = vv / m.b2c;
    param[i] = param[i] - m.lr * (mhat / (sqrt(vhat) + m.eps) + m.wd * param[i]);
  }
}`;
var SUMSQ = `
requires immediate_address_space;
override WG: u32 = 256u;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read_write> out: array<f32>;  // [1]
var<immediate> n: u32;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; var s = 0.0;
  for (var i = tid; i < n; i = i + WG) { let v = x[i]; s = s + v*v; }
  red[tid] = s; workgroupBarrier();
  for (var st = WG/2u; st > 0u; st = st/2u) { if (tid < st) { red[tid] = red[tid] + red[tid+st]; } workgroupBarrier(); }
  if (tid == 0u) { out[0] = out[0] + red[0]; }
}`;

// src/qwgpu/trainer.js
var STORAGE2 = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
var READBACK = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;
var nowMs = /* @__PURE__ */ __name(() => globalThis.performance?.now?.() ?? Date.now(), "nowMs");
var ALL_PROJ = ["q", "k", "v", "o", "gate", "up", "down"];
function createTrainableAdapter(rt, opts = {}) {
  const rank = Math.max(1, Math.floor(opts.rank ?? 16));
  const alpha = opts.alpha ?? rank * 2;
  const scale = opts.scale ?? alpha / rank;
  const targets = opts.targetModules ?? ALL_PROJ;
  const stddev = opts.stddev ?? 1 / Math.sqrt(rank);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const gauss = /* @__PURE__ */ __name(() => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }, "gauss");
  const modules = {};
  for (const L of rt.plan.layers) {
    for (const name of ALL_PROJ) {
      if (!targets.includes(name)) continue;
      const part = L[name];
      const q4 = rt.q4[part.weight];
      const K = q4.K, N = q4.N;
      const Aarr = new Float32Array(rank * K);
      for (let i = 0; i < Aarr.length; i++) Aarr[i] = gauss() * stddev;
      const Barr = new Float32Array(rank * N);
      const A = rt.dev.createBuffer({ size: Aarr.byteLength, usage });
      const B = rt.dev.createBuffer({ size: Barr.byteLength, usage });
      rt.dev.queue.writeBuffer(A, 0, Aarr);
      rt.dev.queue.writeBuffer(B, 0, Barr);
      modules[part.loraKey] = { A, B, rank, scale, inDim: K, outDim: N };
    }
  }
  return { name: opts.name || "trainable", modules };
}
__name(createTrainableAdapter, "createTrainableAdapter");
var QwenLoraTrainer = class {
  static {
    __name(this, "QwenLoraTrainer");
  }
  // rt: a built QwenWGPU. opts: see _normalizeOpts.
  constructor(rt, opts = {}) {
    this.rt = rt;
    this.dev = rt.dev;
    this.cfg = rt.cfg;
    this.opts = this._normalizeOpts(opts);
    this.step = 0;
    this._microInWindow = 0;
    this.scratchT = 0;
    this._buildPipes();
  }
  _normalizeOpts(o) {
    return {
      lr: o.lr ?? 1e-4,
      beta1: o.beta1 ?? 0.9,
      beta2: o.beta2 ?? 0.999,
      eps: o.eps ?? 1e-8,
      weightDecay: o.weightDecay ?? 0,
      maxGradNorm: o.maxGradNorm ?? 1,
      gradAccumSteps: Math.max(1, Math.floor(o.gradAccumSteps ?? 1)),
      lmHeadBlock: Math.max(1, Math.floor(o.lmHeadBlock ?? 128)),
      maxTrainSeq: Math.max(1, Math.floor(o.maxTrainSeq ?? 512)),
      warmupSteps: Math.max(0, Math.floor(o.warmupSteps ?? 0)),
      totalSteps: o.totalSteps ?? 0,
      // for cosine decay; 0 disables decay
      minLrRatio: o.minLrRatio ?? 0.1,
      targetModules: o.targetModules ?? ALL_PROJ
    };
  }
  _buildPipes() {
    const rt = this.rt;
    this.p = {
      dx4: rt._pipe(GEMM_DX_INT4, "bwd_dx4"),
      dd: rt._pipe(LORA_DD, "bwd_lora_dd"),
      gradA: rt._pipe(LORA_GRAD_A, "bwd_lora_dA"),
      gradB: rt._pipe(LORA_GRAD_B, "bwd_lora_dB"),
      dxAdd: rt._pipe(LORA_DX_ADD, "bwd_lora_dx"),
      rmsBwd: rt._pipe(RMSNORM_BWD_T, "bwd_rms"),
      swiglu: rt._pipe(SWIGLU_BWD, "bwd_swiglu"),
      ropeBwd: rt._pipe(ROPE_BWD_T, "bwd_rope"),
      attnStats: rt._pipe(ATTN_BWD_STATS, "bwd_attn_stats"),
      attnDq: rt._pipe(ATTN_BWD_DQ, "bwd_attn_dq"),
      attnDkv: rt._pipe(ATTN_BWD_DKV, "bwd_attn_dkv"),
      logits: rt._pipe(LOGITS_GEMM_I8, "bwd_logits"),
      ceGrad: rt._pipe(CE_SOFTMAX_GRAD, "bwd_ce"),
      dHidden: rt._pipe(DHIDDEN_FROM_DLOGITS_I8, "bwd_dhidden"),
      adamw: rt._pipe(ADAMW_STEP, "adamw"),
      sumsq: rt._pipe(SUMSQ, "sumsq")
    };
  }
  // ---- adapter attach: build per-module grad + Adam moment state ----
  // The adapter must already be uploaded (loadLoraAdapterGPU) and set on rt.
  attach(adapter) {
    if (!adapter || !adapter.modules) throw new Error("trainer.attach: adapter with modules required");
    this.adapter = adapter;
    this.rt.setLora(adapter);
    const rt = this.rt;
    const byKey = /* @__PURE__ */ new Map();
    for (const L of rt.plan.layers) {
      for (const name of ALL_PROJ) {
        const part = L[name];
        byKey.set(part.loraKey, { part, kind: name, q4: rt.q4[part.weight] });
      }
    }
    this.state = {};
    let maxRank = 1;
    for (const key of Object.keys(adapter.modules)) {
      const mod = adapter.modules[key];
      const info = byKey.get(key);
      if (!info) continue;
      const kind = info.kind.replace(/_proj$/, "");
      if (!this.opts.targetModules.includes(kind)) continue;
      const K = info.q4.K, N = info.q4.N, rank = mod.rank;
      maxRank = Math.max(maxRank, rank);
      this.state[key] = {
        mod,
        q4: info.q4,
        K,
        N,
        rank,
        scale: mod.scale,
        dA: rt._buf(rank * K * 4),
        dB: rt._buf(rank * N * 4),
        mA: rt._buf(rank * K * 4),
        vA: rt._buf(rank * K * 4),
        mB: rt._buf(rank * N * 4),
        vB: rt._buf(rank * N * 4)
      };
    }
    this.maxRank = maxRank;
    this.trainedKeys = Object.keys(this.state);
    if (!this.trainedKeys.length) throw new Error("trainer.attach: no trainable modules matched targetModules");
    this._zeroAdamMoments();
    this.zeroGrads();
    return this;
  }
  _zeroAdamMoments() {
    const enc = this.dev.createCommandEncoder();
    for (const k of this.trainedKeys) {
      const st = this.state[k];
      enc.clearBuffer(st.mA);
      enc.clearBuffer(st.vA);
      enc.clearBuffer(st.mB);
      enc.clearBuffer(st.vB);
    }
    this.dev.queue.submit([enc.finish()]);
  }
  zeroGrads() {
    const enc = this.dev.createCommandEncoder();
    for (const k of this.trainedKeys) {
      enc.clearBuffer(this.state[k].dA);
      enc.clearBuffer(this.state[k].dB);
    }
    this.dev.queue.submit([enc.finish()]);
    this._microInWindow = 0;
  }
  // ---- activation/gradient scratch sized to the sequence ----
  _ensureScratch(T) {
    if (this.scratchT >= T && this.s) return;
    if (this.s) for (const k in this.s) this.s[k].destroy?.();
    if (this.ckpt) for (const c2 of this.ckpt) c2.destroy?.();
    this.lossRead?.destroy?.();
    this.normRead?.destroy?.();
    const c = this.cfg;
    const H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize, nH = c.numHeads, R = this.maxRank, lmB = this.opts.lmHeadBlock, V = c.vocabSize;
    const b = /* @__PURE__ */ __name((n) => this.rt._buf(n * 4), "b");
    this.ckpt = [];
    for (let i = 0; i <= c.numLayers; i++) this.ckpt.push(b(T * H));
    this.s = {
      hid: b(T * H),
      normed1: b(T * H),
      normed2: b(T * H),
      normedF: b(T * H),
      q: b(T * qd),
      k: b(T * kvd),
      v: b(T * kvd),
      attn: b(T * qd),
      hmid: b(T * H),
      gate: b(T * I),
      up: b(T * I),
      swig: b(T * I),
      dHidden: b(T * H),
      dnorm: b(T * H),
      dtmp: b(T * H),
      dhmid: b(T * H),
      dq: b(T * qd),
      dk: b(T * kvd),
      dv: b(T * kvd),
      dob: b(T * qd),
      dgate: b(T * I),
      dup: b(T * I),
      dswig: b(T * I),
      dD: b(T * R),
      Dmat: b(T * R),
      lse: b(nH * T),
      delta: b(nH * T),
      logits: b(lmB * V),
      loss: b(T),
      targets: this.rt._buf(T * 4),
      mask: b(T),
      normBuf: b(1)
    };
    this.lossRead = this.rt._buf(T * 4, READBACK);
    this.normRead = this.rt._buf(4, READBACK);
    this.scratchT = T;
  }
  // ---- small dispatch helpers ----
  _grid1d(n) {
    return Math.min(Math.ceil(n / 256), 65535);
  }
  _disp(enc, pipe, buffers, gx, gy, imm, cat) {
    const bg = this.rt._bg(pipe, buffers);
    this.rt._dispatch(enc, pipe, bg, gx, gy, cat || "train", imm);
  }
  _u32(arr) {
    return new Uint32Array(arr);
  }
  _meta(u32parts, f32parts = {}) {
    const buf = new ArrayBuffer(48);
    const dv = new DataView(buf);
    for (const [i, v] of u32parts) dv.setUint32(i * 4, v >>> 0, true);
    for (const [i, v] of Object.entries(f32parts)) dv.setFloat32(Number(i) * 4, v, true);
    return new Uint8Array(buf);
  }
  // ---- forward with checkpoints (LoRA-modified, f32) ----
  _layerForward(enc, L, hid, T) {
    const rt = this.rt, c = this.cfg, s = this.s;
    const H = c.hiddenSize;
    rt.rmsT(enc, hid, rt.bufs[L.inputNorm], s.normed1, T, H);
    rt.gemm4(enc, s.normed1, rt.q4[L.q.weight], s.q, T, rt.bufs[L.q.bias], L.q.loraKey);
    rt.gemm4(enc, s.normed1, rt.q4[L.k.weight], s.k, T, rt.bufs[L.k.bias], L.k.loraKey);
    rt.gemm4(enc, s.normed1, rt.q4[L.v.weight], s.v, T, rt.bufs[L.v.bias], L.v.loraKey);
    rt.ropeT(enc, s.q, T, c.numHeads);
    rt.ropeT(enc, s.k, T, c.numKVHeads);
    rt.attnPrefill(enc, s.q, s.k, s.v, s.attn, T, 0, T);
    rt.gemm4AddT(enc, s.attn, rt.q4[L.o.weight], hid, T, null, L.o.loraKey);
    rt.rmsT(enc, hid, rt.bufs[L.postAttentionNorm], s.normed2, T, H);
    rt.gemm4(enc, s.normed2, rt.q4[L.gate.weight], s.gate, T, null, L.gate.loraKey);
    rt.gemm4(enc, s.normed2, rt.q4[L.up.weight], s.up, T, null, L.up.loraKey);
    enc.copyBufferToBuffer(s.gate, 0, s.swig, 0, T * c.intermediateSize * 4);
    rt._siluMul(enc, s.swig, s.up, T * c.intermediateSize);
    rt.gemm4AddT(enc, s.swig, rt.q4[L.down.weight], hid, T, null, L.down.loraKey);
  }
  _forward(enc, ids, T) {
    const rt = this.rt, c = this.cfg, s = this.s, H = c.hiddenSize;
    rt._ensurePrefillScratch(T, this.maxRank);
    rt._resetUni();
    const e = rt.q[rt.plan.embed.name];
    this.dev.queue.writeBuffer(rt.sT.ids, 0, new Uint32Array(ids));
    rt._dispatch(
      enc,
      rt.pipes.embedT,
      rt._bg(rt.pipes.embedT, [e.w, e.scale, this.ckpt[0], rt.sT.ids]),
      Math.min(Math.ceil(T * H / 256), 65535),
      1,
      "embedT",
      this._u32([T, H, 0, 0])
    );
    enc.copyBufferToBuffer(this.ckpt[0], 0, s.hid, 0, T * H * 4);
    for (let i = 0; i < c.numLayers; i++) {
      this._layerForward(enc, rt.plan.layers[i], s.hid, T);
      enc.copyBufferToBuffer(s.hid, 0, this.ckpt[i + 1], 0, T * H * 4);
    }
  }
  // recompute one layer's forward internals (from its checkpoint) into scratch, also
  // producing hmid (= ckpt + attnProj) which the backward needs as the post-attn input.
  _recomputeLayer(enc, L, T) {
    const rt = this.rt, c = this.cfg, s = this.s, H = c.hiddenSize, idx = L.index;
    rt.rmsT(enc, this.ckpt[idx], rt.bufs[L.inputNorm], s.normed1, T, H);
    rt.gemm4(enc, s.normed1, rt.q4[L.q.weight], s.q, T, rt.bufs[L.q.bias], L.q.loraKey);
    rt.gemm4(enc, s.normed1, rt.q4[L.k.weight], s.k, T, rt.bufs[L.k.bias], L.k.loraKey);
    rt.gemm4(enc, s.normed1, rt.q4[L.v.weight], s.v, T, rt.bufs[L.v.bias], L.v.loraKey);
    rt.ropeT(enc, s.q, T, c.numHeads);
    rt.ropeT(enc, s.k, T, c.numKVHeads);
    rt.attnPrefill(enc, s.q, s.k, s.v, s.attn, T, 0, T);
    enc.copyBufferToBuffer(this.ckpt[idx], 0, s.hmid, 0, T * H * 4);
    rt.gemm4AddT(enc, s.attn, rt.q4[L.o.weight], s.hmid, T, null, L.o.loraKey);
    rt.rmsT(enc, s.hmid, rt.bufs[L.postAttentionNorm], s.normed2, T, H);
    rt.gemm4(enc, s.normed2, rt.q4[L.gate.weight], s.gate, T, null, L.gate.loraKey);
    rt.gemm4(enc, s.normed2, rt.q4[L.up.weight], s.up, T, null, L.up.loraKey);
    enc.copyBufferToBuffer(s.gate, 0, s.swig, 0, T * c.intermediateSize * 4);
    rt._siluMul(enc, s.swig, s.up, T * c.intermediateSize);
  }
  // ---- LoRA + base projection backward ----
  // dY [T][N] -> accumulate into dXbuf [T][K] (base + LoRA), plus dA/dB grads.
  _projBackward(enc, key, Xbuf, dYbuf, dXbuf, T) {
    const st = this.state[key];
    if (!st) {
      this._dispatch_dx4(enc, dYbuf, st, dXbuf, T, key);
      return;
    }
    const { K, N, rank, scale, q4, dA, dB } = st;
    const s = this.s;
    this._disp(
      enc,
      this.p.dx4,
      [dYbuf, q4.w, q4.scale, dXbuf],
      this._grid1d(T * K),
      1,
      this._meta([[0, T], [1, N], [2, K], [3, q4.gpr]]),
      "dx4"
    );
    this._disp(
      enc,
      this.p.dd,
      [dYbuf, st.mod.B, s.dD],
      T * rank,
      1,
      this._meta([[0, T], [1, N], [2, rank]], { 4: scale }),
      "dd"
    );
    this._disp(
      enc,
      this.p.gradA,
      [s.dD, Xbuf, dA],
      this._grid1d(rank * K),
      1,
      this._meta([[0, T], [1, K], [2, rank]]),
      "gradA"
    );
    this._disp(
      enc,
      this.rt.pipes.loraABatch,
      [Xbuf, st.mod.A, s.Dmat],
      rank,
      T,
      this._u32([K, rank, T, 0]),
      "loraABatch"
    );
    this._disp(
      enc,
      this.p.gradB,
      [s.Dmat, dYbuf, dB],
      this._grid1d(rank * N),
      1,
      this._meta([[0, T], [1, N], [2, rank]], { 4: scale }),
      "gradB"
    );
    this._disp(
      enc,
      this.p.dxAdd,
      [s.dD, st.mod.A, dXbuf],
      this._grid1d(T * K),
      1,
      this._meta([[0, T], [1, K], [2, rank]]),
      "dxAdd"
    );
  }
  _dispatch_dx4(enc, dYbuf, st, dXbuf, T, key) {
    const info = this._infoForKey(key);
    const q4 = info.q4;
    this._disp(
      enc,
      this.p.dx4,
      [dYbuf, q4.w, q4.scale, dXbuf],
      this._grid1d(T * q4.K),
      1,
      this._meta([[0, T], [1, q4.N], [2, q4.K], [3, q4.gpr]]),
      "dx4"
    );
  }
  _infoForKey(key) {
    for (const L of this.rt.plan.layers)
      for (const name of ALL_PROJ) if (L[name].loraKey === key) return { q4: this.rt.q4[L[name].weight] };
    throw new Error(`unknown loraKey ${key}`);
  }
  _rmsBwd(enc, xBuf, gBuf, dyBuf, dxBuf, T) {
    const c = this.cfg;
    this._disp(
      enc,
      this.p.rmsBwd,
      [xBuf, gBuf, dyBuf, dxBuf],
      T,
      1,
      new Float32Array([c.hiddenSize, c.rmsNormEps]),
      "rmsBwd"
    );
  }
  // ---- full backward for one micro-batch; accumulates grads, returns nothing ----
  _backward(enc, T, numActive) {
    const rt = this.rt, c = this.cfg, s = this.s, H = c.hiddenSize, qd = c.numHeads * c.headDim, kvd = c.numKVHeads * c.headDim, I = c.intermediateSize, V = c.vocabSize;
    rt.rmsT(enc, this.ckpt[c.numLayers], rt.bufs[rt.plan.finalNorm.name], s.normedF, T, H);
    enc.clearBuffer(s.dnorm);
    const e = rt.q[rt.plan.embed.name];
    const lossScale = 1 / Math.max(1, numActive);
    const lmB = this.opts.lmHeadBlock;
    for (let off = 0; off < T; off += lmB) {
      const bt = Math.min(lmB, T - off);
      this._disp(
        enc,
        this.p.logits,
        [s.normedF, e.w, e.scale, s.logits],
        this._grid1d(bt * V),
        1,
        this._meta([[0, bt], [1, V], [2, H], [3, off]]),
        "logits"
      );
      this._disp(
        enc,
        this.p.ceGrad,
        [s.logits, s.targets, s.mask, s.loss],
        bt,
        1,
        this._meta([[0, V], [1, off]], { 2: lossScale }),
        "ce"
      );
      this._disp(
        enc,
        this.p.dHidden,
        [s.logits, e.w, e.scale, s.dnorm],
        this._grid1d(bt * H),
        1,
        this._meta([[0, bt], [1, V], [2, H], [3, off]]),
        "dHidden"
      );
    }
    this._rmsBwd(enc, this.ckpt[c.numLayers], rt.bufs[rt.plan.finalNorm.name], s.dnorm, s.dHidden, T);
    for (let i = c.numLayers - 1; i >= 0; i--) {
      const L = rt.plan.layers[i];
      this._recomputeLayer(enc, L, T);
      enc.clearBuffer(s.dswig);
      this._projBackward(enc, L.down.loraKey, s.swig, s.dHidden, s.dswig, T);
      this._disp(
        enc,
        this.p.swiglu,
        [s.gate, s.up, s.dswig, s.dgate, s.dup],
        this._grid1d(T * I),
        1,
        this._u32([T * I]),
        "swiglu"
      );
      enc.clearBuffer(s.dnorm);
      this._projBackward(enc, L.gate.loraKey, s.normed2, s.dgate, s.dnorm, T);
      this._projBackward(enc, L.up.loraKey, s.normed2, s.dup, s.dnorm, T);
      this._rmsBwd(enc, s.hmid, rt.bufs[L.postAttentionNorm], s.dnorm, s.dtmp, T);
      enc.copyBufferToBuffer(s.dHidden, 0, s.dhmid, 0, T * H * 4);
      rt._addInto(enc, s.dhmid, s.dtmp, T * H);
      enc.clearBuffer(s.dob);
      this._projBackward(enc, L.o.loraKey, s.attn, s.dhmid, s.dob, T);
      const am = this._u32([c.numHeads, c.numKVHeads, c.headDim, T]);
      this._disp(enc, this.p.attnStats, [s.q, s.k, s.attn, s.dob, s.lse, s.delta], c.numHeads, T, am, "attnStats");
      enc.clearBuffer(s.dq);
      enc.clearBuffer(s.dk);
      enc.clearBuffer(s.dv);
      this._disp(enc, this.p.attnDq, [s.q, s.k, s.v, s.dob, s.lse, s.delta, s.dq], c.numHeads, T, am, "attnDq");
      this._disp(
        enc,
        this.p.attnDkv,
        [s.q, s.k, s.v, s.dob, s.lse, s.delta, s.dk, s.dv],
        c.numKVHeads,
        T,
        am,
        "attnDkv"
      );
      this._disp(
        enc,
        this.p.ropeBwd,
        [s.dq, rt.ropeCos, rt.ropeSin],
        Math.ceil(T * c.numHeads * (c.headDim / 2) / 256),
        1,
        this._u32([c.numHeads, c.headDim, T, 0]),
        "ropeBwd"
      );
      this._disp(
        enc,
        this.p.ropeBwd,
        [s.dk, rt.ropeCos, rt.ropeSin],
        Math.ceil(T * c.numKVHeads * (c.headDim / 2) / 256),
        1,
        this._u32([c.numKVHeads, c.headDim, T, 0]),
        "ropeBwd"
      );
      enc.clearBuffer(s.dnorm);
      this._projBackward(enc, L.q.loraKey, s.normed1, s.dq, s.dnorm, T);
      this._projBackward(enc, L.k.loraKey, s.normed1, s.dk, s.dnorm, T);
      this._projBackward(enc, L.v.loraKey, s.normed1, s.dv, s.dnorm, T);
      this._rmsBwd(enc, this.ckpt[i], rt.bufs[L.inputNorm], s.dnorm, s.dtmp, T);
      enc.copyBufferToBuffer(s.dhmid, 0, s.dHidden, 0, T * H * 4);
      rt._addInto(enc, s.dHidden, s.dtmp, T * H);
    }
  }
  // shifted-label targets + mask into the scratch buffers; returns numActive.
  _writeTargets(tokens, lossMask, T) {
    const targets = new Uint32Array(T);
    const mask = new Float32Array(T);
    let numActive = 0;
    for (let t = 0; t < T - 1; t++) {
      targets[t] = tokens[t + 1] >>> 0;
      const mk = lossMask ? lossMask[t] ? 1 : 0 : 1;
      mask[t] = mk;
      numActive += mk;
    }
    targets[T - 1] = 0;
    mask[T - 1] = 0;
    this.dev.queue.writeBuffer(this.s.targets, 0, targets);
    this.dev.queue.writeBuffer(this.s.mask, 0, mask);
    return numActive;
  }
  // loss head only (final norm + streamed logits + CE), no backward sweep. Used by
  // evalLoss(). CE overwrites s.logits with dLogits but we ignore that here.
  _lossOnly(enc, T, numActive) {
    const rt = this.rt, c = this.cfg, s = this.s, H = c.hiddenSize, V = c.vocabSize;
    rt.rmsT(enc, this.ckpt[c.numLayers], rt.bufs[rt.plan.finalNorm.name], s.normedF, T, H);
    const e = rt.q[rt.plan.embed.name];
    const lossScale = 1 / Math.max(1, numActive);
    const lmB = this.opts.lmHeadBlock;
    for (let off = 0; off < T; off += lmB) {
      const bt = Math.min(lmB, T - off);
      this._disp(enc, this.p.logits, [s.normedF, e.w, e.scale, s.logits], this._grid1d(bt * V), 1, this._meta([[0, bt], [1, V], [2, H], [3, off]]), "logits");
      this._disp(enc, this.p.ceGrad, [s.logits, s.targets, s.mask, s.loss], bt, 1, this._meta([[0, V], [1, off]], { 2: lossScale }), "ce");
    }
  }
  // ---- public: forward-only mean cross-entropy (no grads). For held-out eval. ----
  async evalLoss(tokens, lossMask) {
    const T = tokens.length;
    if (T > this.opts.maxTrainSeq) throw new Error(`seq ${T} > maxTrainSeq ${this.opts.maxTrainSeq}`);
    this._ensureScratch(T);
    const wasF16 = this.rt.usingF16?.();
    this.rt.setUseF16?.(false);
    try {
      const numActive = this._writeTargets(tokens, lossMask, T);
      const enc = this.dev.createCommandEncoder();
      this._forward(enc, tokens, T);
      this._lossOnly(enc, T, numActive);
      enc.copyBufferToBuffer(this.s.loss, 0, this.lossRead, 0, T * 4);
      this.dev.queue.submit([enc.finish()]);
      await this.lossRead.mapAsync(GPUMapMode.READ);
      const arr = new Float32Array(this.lossRead.getMappedRange().slice(0));
      this.lossRead.unmap();
      let sum = 0;
      for (let t = 0; t < T; t++) sum += arr[t];
      return { loss: sum / Math.max(1, numActive), numActive };
    } finally {
      if (wasF16) this.rt.setUseF16?.(true);
    }
  }
  // ---- public: accumulate one micro-batch. tokens: Int array, lossMask: 0/1 array. ----
  // lossMask[t]==1 means "train the prediction of tokens[t+1] from position t".
  async microStep(tokens, lossMask) {
    const c = this.cfg;
    const T = tokens.length;
    const t0 = nowMs();
    if (T > this.opts.maxTrainSeq) throw new Error(`seq ${T} > maxTrainSeq ${this.opts.maxTrainSeq}`);
    this._ensureScratch(T);
    const wasF16 = this.rt.usingF16?.();
    this.rt.setUseF16?.(false);
    try {
      const numActive = this._writeTargets(tokens, lossMask, T);
      const enc = this.dev.createCommandEncoder();
      this._forward(enc, tokens, T);
      this._backward(enc, T, numActive);
      enc.copyBufferToBuffer(this.s.loss, 0, this.lossRead, 0, T * 4);
      this.dev.queue.submit([enc.finish()]);
      await this.lossRead.mapAsync(GPUMapMode.READ);
      const lossArr = new Float32Array(this.lossRead.getMappedRange().slice(0));
      this.lossRead.unmap();
      let lossSum = 0;
      for (let t = 0; t < T; t++) lossSum += lossArr[t];
      this._microInWindow++;
      const microStepMs = nowMs() - t0;
      return {
        loss: lossSum / Math.max(1, numActive),
        numActive,
        tokens: T,
        microStepMs,
        trainTokPerSec: T / Math.max(1e-6, microStepMs / 1e3)
      };
    } finally {
      if (wasF16) this.rt.setUseF16?.(true);
    }
  }
  // ---- public: apply accumulated grads with AdamW + global-norm clip ----
  async optimizerStep() {
    const t0 = nowMs();
    const o = this.opts;
    const accum = this._microInWindow || 1;
    const encN = this.dev.createCommandEncoder();
    encN.clearBuffer(this.s.normBuf);
    for (const k of this.trainedKeys) {
      const st = this.state[k];
      this._disp(encN, this.p.sumsq, [st.dA, this.s.normBuf], 1, 1, this._u32([st.rank * st.K]), "sumsq");
      this._disp(encN, this.p.sumsq, [st.dB, this.s.normBuf], 1, 1, this._u32([st.rank * st.N]), "sumsq");
    }
    encN.copyBufferToBuffer(this.s.normBuf, 0, this.normRead, 0, 4);
    this.dev.queue.submit([encN.finish()]);
    await this.normRead.mapAsync(GPUMapMode.READ);
    const sumsq = new Float32Array(this.normRead.getMappedRange().slice(0))[0];
    this.normRead.unmap();
    const gradScale = 1 / accum;
    const gnorm = Math.sqrt(sumsq) * gradScale;
    const clip2 = o.maxGradNorm > 0 && gnorm > o.maxGradNorm ? o.maxGradNorm / (gnorm + 1e-6) : 1;
    const gScale = gradScale * clip2;
    this.step++;
    const lr = this._lrAt(this.step);
    const b1c = 1 - Math.pow(o.beta1, this.step);
    const b2c = 1 - Math.pow(o.beta2, this.step);
    const enc = this.dev.createCommandEncoder();
    for (const k of this.trainedKeys) {
      const st = this.state[k];
      const metaA = this._adamMeta(st.rank * st.K, lr, gScale, b1c, b2c);
      this._disp(enc, this.p.adamw, [st.mod.A, st.dA, st.mA, st.vA], this._grid1d(st.rank * st.K), 1, metaA, "adamw");
      const metaB = this._adamMeta(st.rank * st.N, lr, gScale, b1c, b2c);
      this._disp(enc, this.p.adamw, [st.mod.B, st.dB, st.mB, st.vB], this._grid1d(st.rank * st.N), 1, metaB, "adamw");
    }
    this.dev.queue.submit([enc.finish()]);
    this.rt.invalidateLora();
    this.zeroGrads();
    return { lr, gradNorm: gnorm, clip: clip2, optimizerStepMs: nowMs() - t0 };
  }
  _lrAt(step) {
    const o = this.opts;
    if (o.warmupSteps > 0 && step <= o.warmupSteps) return o.lr * (step / o.warmupSteps);
    if (o.totalSteps > 0 && step > o.warmupSteps) {
      const prog = (step - o.warmupSteps) / Math.max(1, o.totalSteps - o.warmupSteps);
      const cos = 0.5 * (1 + Math.cos(Math.PI * Math.min(1, prog)));
      return o.lr * (o.minLrRatio + (1 - o.minLrRatio) * cos);
    }
    return o.lr;
  }
  _adamMeta(n, lr, gScale, b1c, b2c) {
    const o = this.opts;
    const buf = new ArrayBuffer(48);
    const dv = new DataView(buf);
    dv.setUint32(0, n >>> 0, true);
    dv.setFloat32(8, lr, true);
    dv.setFloat32(12, o.beta1, true);
    dv.setFloat32(16, o.beta2, true);
    dv.setFloat32(20, o.eps, true);
    dv.setFloat32(24, o.weightDecay, true);
    dv.setFloat32(28, gScale, true);
    dv.setFloat32(32, b1c, true);
    dv.setFloat32(36, b2c, true);
    return new Uint8Array(buf);
  }
  // ---- convenience: one full optimization step over a list of micro-batches ----
  async trainStep(batches) {
    const list = Array.isArray(batches) ? batches : [batches];
    let lossSum = 0, n = 0, numActive = 0, tokens = 0, microStepMs = 0;
    for (const b of list) {
      const r = await this.microStep(b.tokens, b.lossMask);
      lossSum += r.loss;
      numActive += r.numActive || 0;
      tokens += r.tokens || b.tokens?.length || 0;
      microStepMs += r.microStepMs || 0;
      n++;
    }
    const opt = await this.optimizerStep();
    const totalStepMs = microStepMs + (opt.optimizerStepMs || 0);
    return {
      loss: lossSum / Math.max(1, n),
      microBatches: n,
      numActive,
      tokens,
      microStepMs,
      totalStepMs,
      trainTokPerSec: tokens / Math.max(1e-6, totalStepMs / 1e3),
      ...opt
    };
  }
};

// src/services/training_controller.js
var IM_END = 151645;
var TrainingController = class {
  static {
    __name(this, "TrainingController");
  }
  // session: a loaded ModelSession (rt + tokenizer). adapters: AdapterRegistry.
  constructor({ session: session2, adapters: adapters2, log: log2 = /* @__PURE__ */ __name(() => {
  }, "log"), trainerOptions = {} } = {}) {
    this.session = session2;
    this.adapters = adapters2;
    this.log = log2;
    this.trainerOptions = trainerOptions;
    this.trainer = null;
    this.adapter = null;
  }
  get rt() {
    return this.session.rt;
  }
  get tokenizer() {
    return this.session.tokenizer;
  }
  // Create + register a fresh trainable adapter and attach the trainer to it.
  initAdapter(name = "trainable", { rank = 16, alpha = 32, targetModules } = {}) {
    const adapter = createTrainableAdapter(this.rt, { name, rank, alpha, targetModules });
    this.adapters.adapters[name] = adapter;
    this.adapter = adapter;
    this.trainer = new QwenLoraTrainer(this.rt, this.trainerOptions);
    this.trainer.attach(adapter);
    this.log(`init adapter "${name}" rank=${rank} alpha=${alpha} modules=${Object.keys(adapter.modules).length}`);
    return adapter;
  }
  // Attach to an already-registered adapter (e.g. continue training a loaded one).
  attachAdapter(name) {
    const adapter = this.adapters.get(name);
    if (!adapter) throw new Error(`adapter "${name}" not found`);
    this.adapter = adapter;
    this.trainer = new QwenLoraTrainer(this.rt, this.trainerOptions);
    this.trainer.attach(adapter);
    return adapter;
  }
  /*
   * TECHNIQUE: Completion-only loss masking with shifted labels
   *   Tokenize prompt (with assistant generation prompt) and completion separately.
   *   mask[t]=1 trains the prediction of tokens[t+1] from position t — so we mask
   *   positions whose NEXT token is part of the completion (incl. the final EOS).
   *   Prompt tokens get mask=0, so the model is only graded on what it should write.
   */
  prepareExample({ messages, prompt, completion, trainPromptToo = false }) {
    const tk = this.tokenizer;
    let promptIds;
    if (messages) {
      promptIds = tk.encode(formatMessages(tk, messages));
    } else {
      promptIds = tk.encode(prompt);
    }
    const compIds = tk.encode(completion, { add_special_tokens: false });
    const tokens = [...promptIds, ...compIds, IM_END];
    const T = tokens.length;
    const lossMask = new Array(T).fill(0);
    const firstTrainPos = trainPromptToo ? 0 : Math.max(0, promptIds.length - 1);
    for (let t = firstTrainPos; t < T - 1; t++) lossMask[t] = 1;
    return {
      tokens,
      lossMask,
      promptLength: promptIds.length,
      completionLength: compIds.length,
      firstTrainPos
    };
  }
  inspectExample(example) {
    const prepared = this.prepareExample(example);
    const { tokens, lossMask, promptLength, completionLength, firstTrainPos } = prepared;
    const rows = tokens.map((id, index) => {
      const targetId = index + 1 < tokens.length ? tokens[index + 1] : null;
      const segment = index < promptLength ? "prompt" : index < promptLength + completionLength ? "completion" : "eos";
      return {
        index,
        id,
        text: decodeToken(this.tokenizer, id),
        segment,
        trainsNext: !!lossMask[index],
        targetId,
        targetText: targetId == null ? "" : decodeToken(this.tokenizer, targetId)
      };
    });
    return {
      ...prepared,
      trainPositions: lossMask.reduce((n, v) => n + (v ? 1 : 0), 0),
      firstTrainPos,
      rows
    };
  }
  prepareBatch(examples) {
    return examples.map((e) => this.prepareExample(e));
  }
  // One optimizer step over `microBatches` (array of {tokens, lossMask}); grads
  // accumulate across them, then a single AdamW update is applied.
  async step(microBatches) {
    if (!this.trainer) throw new Error("call initAdapter()/attachAdapter() first");
    return this.trainer.trainStep(microBatches);
  }
  // Full training run over a dataset of examples. Honors gradAccumSteps by grouping
  // examples into accumulation windows. Calls onStep({step, loss, lr, gradNorm}).
  async train(examples, { epochs = 1, onStep = /* @__PURE__ */ __name(() => {
  }, "onStep"), maxTrainSeq } = {}) {
    if (!this.trainer) this.initAdapter();
    const accum = this.trainer.opts.gradAccumSteps;
    const cap = maxTrainSeq ?? this.trainer.opts.maxTrainSeq;
    let globalStep = 0;
    for (let ep = 0; ep < epochs; ep++) {
      const order = shuffle([...Array(examples.length).keys()]);
      let window2 = [];
      for (const idx of order) {
        let mb = this.prepareExample(examples[idx]);
        if (mb.tokens.length > cap) mb = truncate(mb, cap);
        window2.push(mb);
        if (window2.length === accum) {
          const r = await this.step(window2);
          globalStep++;
          this.log(`step ${globalStep} epoch ${ep} loss=${r.loss.toFixed(4)} lr=${r.lr.toExponential(2)} |g|=${r.gradNorm.toFixed(3)}`);
          onStep({ step: globalStep, epoch: ep, ...r });
          window2 = [];
        }
      }
      if (window2.length) {
        const r = await this.step(window2);
        globalStep++;
        onStep({ step: globalStep, epoch: ep, ...r });
      }
    }
    this.adapters.applyToRuntime(this.adapter.name, this.rt);
    return { steps: globalStep, adapter: this.adapter };
  }
};
function truncate(mb, cap) {
  return {
    ...mb,
    tokens: mb.tokens.slice(0, cap),
    lossMask: mb.lossMask.slice(0, cap)
  };
}
__name(truncate, "truncate");
function decodeToken(tokenizer, id) {
  try {
    if (tokenizer?.decode) return tokenizer.decode([id], { skip_special_tokens: false });
  } catch {
  }
  return String(id);
}
__name(decodeToken, "decodeToken");
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
__name(shuffle, "shuffle");

// src/lora_export.js
async function readBufferF32(dev, src, byteLen) {
  const rb = dev.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rb, 0, byteLen);
  dev.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  rb.destroy();
  return out;
}
__name(readBufferF32, "readBufferF32");
function transpose2d(arr, rows, cols) {
  const o = new Float32Array(arr.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) o[c * rows + r] = arr[r * cols + c];
  return o;
}
__name(transpose2d, "transpose2d");
function buildSafetensors(tensors, metadata = { format: "pt" }) {
  let offset = 0;
  const header = {};
  if (metadata) header.__metadata__ = metadata;
  for (const t of tensors) {
    const bytes = t.data.byteLength;
    header[t.name] = { dtype: "F32", shape: t.shape, data_offsets: [offset, offset + bytes] };
    offset += bytes;
  }
  let headerStr = JSON.stringify(header);
  const enc = new TextEncoder();
  let headerBytes = enc.encode(headerStr);
  const pad = (8 - headerBytes.length % 8) % 8;
  if (pad) {
    headerStr += " ".repeat(pad);
    headerBytes = enc.encode(headerStr);
  }
  const total = 8 + headerBytes.length + offset;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  dv.setBigUint64(0, BigInt(headerBytes.length), true);
  new Uint8Array(buf, 8, headerBytes.length).set(headerBytes);
  let p = 8 + headerBytes.length;
  for (const t of tensors) {
    new Uint8Array(buf, p, t.data.byteLength).set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength));
    p += t.data.byteLength;
  }
  return new Uint8Array(buf);
}
__name(buildSafetensors, "buildSafetensors");
async function exportLoraAdapter(trainer, opts = {}) {
  const rt = trainer.rt;
  const dev = rt.dev;
  const tensors = [];
  const targets = /* @__PURE__ */ new Set();
  const rankByKey = {};
  const alphaByKey = {};
  for (const key of trainer.trainedKeys) {
    const st = trainer.state[key];
    const A = await readBufferF32(dev, st.mod.A, st.rank * st.K * 4);
    const B = await readBufferF32(dev, st.mod.B, st.rank * st.N * 4);
    const Bt = transpose2d(B, st.rank, st.N);
    const base = `base_model.model.model.${key}`;
    tensors.push({ name: `${base}.lora_A.weight`, shape: [st.rank, st.K], data: A });
    tensors.push({ name: `${base}.lora_B.weight`, shape: [st.N, st.rank], data: Bt });
    rankByKey[key] = st.rank;
    alphaByKey[key] = st.scale * st.rank;
    targets.add(key.split(".").pop());
  }
  const safetensors = buildSafetensors(tensors);
  const ranks = Object.values(rankByKey);
  const alphas = Object.values(alphaByKey);
  const r = opts.rank ?? mode(ranks) ?? 0;
  const alpha = opts.alpha ?? mode(alphas) ?? 0;
  const rankPattern = {};
  const alphaPattern = {};
  for (const key of Object.keys(rankByKey)) {
    if (rankByKey[key] !== r) rankPattern[key] = rankByKey[key];
    if (alphaByKey[key] !== alpha) alphaPattern[key] = alphaByKey[key];
  }
  const config = {
    peft_type: "LORA",
    auto_mapping: null,
    base_model_name_or_path: opts.baseModel || "WeiboAI/VibeThinker-3B",
    r,
    lora_alpha: alpha,
    target_modules: [...targets],
    lora_dropout: 0,
    bias: "none",
    fan_in_fan_out: false,
    inference_mode: true,
    task_type: "CAUSAL_LM",
    ...Object.keys(rankPattern).length ? { rank_pattern: rankPattern } : {},
    ...Object.keys(alphaPattern).length ? { alpha_pattern: alphaPattern } : {}
  };
  const configJson = JSON.stringify(config, null, 2);
  return { safetensors, config, configJson };
}
__name(exportLoraAdapter, "exportLoraAdapter");
function mode(arr) {
  if (!arr.length) return void 0;
  const counts = /* @__PURE__ */ new Map();
  let best = arr[0], bestN = 0;
  for (const v of arr) {
    const n = (counts.get(v) || 0) + 1;
    counts.set(v, n);
    if (n > bestN) {
      bestN = n;
      best = v;
    }
  }
  return best;
}
__name(mode, "mode");
async function downloadLoraAdapter(trainer, opts = {}) {
  const { safetensors, configJson } = await exportLoraAdapter(trainer, opts);
  const stem = opts.name || trainer.adapter?.name || "adapter";
  triggerDownload(new Blob([safetensors], { type: "application/octet-stream" }), `${stem}.safetensors`);
  triggerDownload(new Blob([configJson], { type: "application/json" }), "adapter_config.json");
}
__name(downloadLoraAdapter, "downloadLoraAdapter");
function triggerDownload(blob, filename) {
  if (typeof document === "undefined") return;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1e3);
}
__name(triggerDownload, "triggerDownload");

// src/lora_gpu.js
function parseSt(buf) {
  const dv = new DataView(buf);
  const hl = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
  return { header, dataStart: 8 + hl, u8: new Uint8Array(buf) };
}
__name(parseSt, "parseSt");
function bf16f32(u8, off, n) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset + off, n);
  const o = new Float32Array(n);
  const o32 = new Uint32Array(o.buffer);
  for (let i = 0; i < n; i++) o32[i] = u16[i] << 16;
  return o;
}
__name(bf16f32, "bf16f32");
function f32(u8, off, n) {
  return new Float32Array(u8.buffer.slice(u8.byteOffset + off, u8.byteOffset + off + n * 4));
}
__name(f32, "f32");
function readTensor(st, name) {
  const t = st.header[name];
  const n = t.shape.reduce((a, b) => a * b, 1);
  const dt = t.dtype.toUpperCase();
  const arr = dt === "BF16" ? bf16f32(st.u8, st.dataStart + t.data_offsets[0], n) : f32(st.u8, st.dataStart + t.data_offsets[0], n);
  return { arr, shape: t.shape };
}
__name(readTensor, "readTensor");
var isA = /* @__PURE__ */ __name((name) => /lora_a/i.test(name), "isA");
function transpose2d2(arr, rows, cols) {
  const o = new Float32Array(arr.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) o[c * rows + r] = arr[r * cols + c];
  return o;
}
__name(transpose2d2, "transpose2d");
async function loadLoraAdapterGPU(dev, files, cfg) {
  const stFile = files.find((f) => f.name.endsWith(".safetensors"));
  if (!stFile) throw new Error("no .safetensors in adapter files");
  const cfgFile = files.find((f) => /adapter_config\.json|config\.json/.test(f.name));
  let rankCfg = 16, scaleCfg = null;
  if (cfgFile) {
    const c = JSON.parse(await cfgFile.text());
    const lp = c.lora_parameters || {};
    rankCfg = c.r ?? c.rank ?? c.lora_rank ?? lp.rank ?? rankCfg;
    if (lp.scale != null)
      scaleCfg = lp.scale;
    else if (c.lora_alpha != null)
      scaleCfg = c.lora_alpha / rankCfg;
    else if (c.alpha != null) scaleCfg = c.alpha / rankCfg;
  }
  const st = parseSt(await stFile.arrayBuffer());
  const names = Object.keys(st.header).filter((k) => k !== "__metadata__" && /lora_[abAB]/.test(k));
  const groups = {};
  for (const nm of names) {
    const key = moduleKeyFromTensorName(nm);
    if (!key) continue;
    (groups[key] ||= {})[isA(nm) ? "A" : "B"] = readTensor(st, nm);
  }
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const mk = /* @__PURE__ */ __name((arr) => {
    const b = dev.createBuffer({ size: arr.byteLength, usage: S });
    dev.queue.writeBuffer(b, 0, arr);
    return b;
  }, "mk");
  const modules = {};
  for (const key of Object.keys(groups)) {
    const g = groups[key];
    if (!g.A || !g.B) continue;
    const r = Math.min(...g.A.shape, ...g.B.shape);
    let Aarr = g.A.arr;
    if (g.A.shape[0] !== r) Aarr = transpose2d2(g.A.arr, g.A.shape[0], g.A.shape[1]);
    let Barr = g.B.arr;
    if (g.B.shape[0] !== r) Barr = transpose2d2(g.B.arr, g.B.shape[0], g.B.shape[1]);
    const scale = scaleCfg != null ? scaleCfg : 2;
    modules[key] = { A: mk(Aarr), B: mk(Barr), rawA: Aarr, rawB: Barr, rank: r, scale };
  }
  if (!Object.keys(modules).length) throw new Error("no LoRA modules matched layers.*.{self_attn,mlp}.*_proj");
  const name = stFile.name.replace(/\.safetensors$/, "");
  return { name, modules };
}
__name(loadLoraAdapterGPU, "loadLoraAdapterGPU");

// src/services/store.js
var store_exports = {};
__export(store_exports, {
  connectDirectory: () => connectDirectory,
  deleteRun: () => deleteRun,
  ensurePermission: () => ensurePermission,
  forgetDirectory: () => forgetDirectory,
  fsSupported: () => fsSupported,
  getRun: () => getRun,
  getRunBlobs: () => getRunBlobs,
  listRuns: () => listRuns,
  loadRunFiles: () => loadRunFiles,
  newId: () => newId,
  readDirText: () => readDirText,
  saveRun: () => saveRun,
  savedDirectory: () => savedDirectory,
  writeFileToDir: () => writeFileToDir
});
var LS_KEY = "emberglass.history.v2";
var DB_NAME = "emberglass";
var DB_VERSION = 1;
var BLOB_STORE = "adapters";
var HANDLE_STORE = "handles";
var _dbp = null;
function db() {
  if (_dbp) return _dbp;
  _dbp = new Promise((resolve, reject) => {
    const r = indexedDB.open(DB_NAME, DB_VERSION);
    r.onupgradeneeded = () => {
      const d = r.result;
      if (!d.objectStoreNames.contains(BLOB_STORE)) d.createObjectStore(BLOB_STORE);
      if (!d.objectStoreNames.contains(HANDLE_STORE)) d.createObjectStore(HANDLE_STORE);
    };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
  return _dbp;
}
__name(db, "db");
async function idbPut(store, key, val) {
  const d = await db();
  return new Promise((res, rej) => {
    const tx = d.transaction(store, "readwrite");
    tx.objectStore(store).put(val, key);
    tx.oncomplete = () => res();
    tx.onerror = () => rej(tx.error);
  });
}
__name(idbPut, "idbPut");
async function idbGet(store, key) {
  const d = await db();
  return new Promise((res, rej) => {
    const tx = d.transaction(store, "readonly");
    const rq = tx.objectStore(store).get(key);
    rq.onsuccess = () => res(rq.result);
    rq.onerror = () => rej(rq.error);
  });
}
__name(idbGet, "idbGet");
async function idbDel(store, key) {
  const d = await db();
  return new Promise((res, rej) => {
    const tx = d.transaction(store, "readwrite");
    tx.objectStore(store).delete(key);
    tx.oncomplete = () => res();
    tx.onerror = () => rej(tx.error);
  });
}
__name(idbDel, "idbDel");
function listRuns() {
  try {
    const a = JSON.parse(localStorage.getItem(LS_KEY) || "[]");
    return Array.isArray(a) ? a : [];
  } catch {
    return [];
  }
}
__name(listRuns, "listRuns");
function writeIndex(arr) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(arr));
  } catch (e) {
    console.warn("[store] localStorage write failed", e);
  }
}
__name(writeIndex, "writeIndex");
function getRun(id) {
  return listRuns().find((r) => r.id === id) || null;
}
__name(getRun, "getRun");
function newId() {
  return "run_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 7);
}
__name(newId, "newId");
async function saveRun(meta, files) {
  const stBytes = files.safetensors instanceof Uint8Array ? files.safetensors : new Uint8Array(files.safetensors);
  await idbPut(BLOB_STORE, meta.id, {
    safetensors: new Blob([stBytes], { type: "application/octet-stream" }),
    configJson: files.configJson || "{}"
  });
  const idx = listRuns().filter((r) => r.id !== meta.id);
  idx.unshift(meta);
  writeIndex(idx);
  return meta;
}
__name(saveRun, "saveRun");
async function deleteRun(id) {
  writeIndex(listRuns().filter((r) => r.id !== id));
  try {
    await idbDel(BLOB_STORE, id);
  } catch {
  }
}
__name(deleteRun, "deleteRun");
async function loadRunFiles(id) {
  const rec = await idbGet(BLOB_STORE, id);
  if (!rec) throw new Error("adapter blob missing for " + id);
  const meta = getRun(id);
  const stem = (meta?.name || id).replace(/[^\w.-]+/g, "_");
  return [
    new File([rec.safetensors], `${stem}.safetensors`, { type: "application/octet-stream" }),
    new File([rec.configJson], "adapter_config.json", { type: "application/json" })
  ];
}
__name(loadRunFiles, "loadRunFiles");
async function getRunBlobs(id) {
  const rec = await idbGet(BLOB_STORE, id);
  if (!rec) throw new Error("adapter blob missing for " + id);
  return { safetensors: rec.safetensors, configJson: rec.configJson };
}
__name(getRunBlobs, "getRunBlobs");
var fsSupported = typeof window !== "undefined" && "showDirectoryPicker" in window;
async function connectDirectory() {
  if (!fsSupported) throw new Error("File System Access API not available in this browser");
  const handle = await window.showDirectoryPicker({ id: "emberglass", mode: "readwrite" });
  await idbPut(HANDLE_STORE, "dir", handle);
  return handle;
}
__name(connectDirectory, "connectDirectory");
async function savedDirectory() {
  if (!fsSupported) return null;
  try {
    return await idbGet(HANDLE_STORE, "dir") || null;
  } catch {
    return null;
  }
}
__name(savedDirectory, "savedDirectory");
async function forgetDirectory() {
  try {
    await idbDel(HANDLE_STORE, "dir");
  } catch {
  }
}
__name(forgetDirectory, "forgetDirectory");
async function ensurePermission(handle, mode2 = "readwrite") {
  if (!handle) return false;
  const opts = { mode: mode2 };
  if (await handle.queryPermission(opts) === "granted") return true;
  return await handle.requestPermission(opts) === "granted";
}
__name(ensurePermission, "ensurePermission");
async function readDirText(handle, { exts = ["txt", "md", "json", "csv"], maxChars = 2e5 } = {}) {
  let out = "";
  const names = [];
  for await (const [name, h] of handle.entries()) {
    if (h.kind !== "file") continue;
    const ext = name.split(".").pop().toLowerCase();
    if (!exts.includes(ext)) continue;
    try {
      const f = await h.getFile();
      out += `

# ${name}
` + await f.text();
      names.push(name);
      if (out.length > maxChars) break;
    } catch {
    }
  }
  return { text: out.slice(0, maxChars), names };
}
__name(readDirText, "readDirText");
async function writeFileToDir(handle, name, data) {
  const fh = await handle.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(data);
  await w.close();
}
__name(writeFileToDir, "writeFileToDir");

// src/skills/inbox-calendar/port.ts
var DOMAIN = "an Inbox & Calendar operator";
var SCOPE = "inbox or calendar";
var CONTEXT = "Assume today is Monday 2026-06-29, local time. Express every date and time as ISO 8601 (YYYY-MM-DDTHH:MM) and always set end = start + the requested duration.";
var OPS = [
  { name: "find_email", params: ["query"], ret: "thread" },
  { name: "compose_email", params: ["to", "subject", "body"] },
  { name: "reply_email", params: ["thread", "body"] },
  { name: "forward_email", params: ["thread", "to", "note"] },
  { name: "archive_email", params: ["thread"] },
  { name: "label_email", params: ["thread", "label"] },
  { name: "schedule_send", params: ["to", "subject", "body", "when"] },
  { name: "create_event", params: ["title", "start", "end", "remind_min"] },
  { name: "set_reminder", params: ["text", "when"] },
  { name: "find_slot", params: ["duration_min", "after", "before"], ret: "slot" },
  { name: "rsvp", params: ["event", "response"] }
];
var META = {
  key: "inbox-calendar",
  label: "Inbox & Calendar",
  icon: "\u2709",
  desc: "Compiles requests like \u201Cemail my mom and book a reminder to respond\u201D into a verifiable macro over a fixed set of inbox/calendar actions; bounces anything else.",
  suggest: "Email the design team this week's notes, then put a 30-minute review on my calendar for Monday morning."
};

// src/skills/inbox-calendar/contract.ts
var ISO_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/;
var lines = /* @__PURE__ */ __name((m) => String(m).split("\n"), "lines");
var CALENDAR_CONTRACT = {
  assertions: [
    {
      id: "iso-times",
      describe: "start/end/when/after/before literals are ISO 8601 (YYYY-MM-DDTHH:MM)",
      holds: /* @__PURE__ */ __name((m) => lines(m).every(
        (ln) => [...ln.matchAll(/(?:start|end|when|after|before)="([^"]+)"/g)].every((x) => ISO_RE.test(x[1]))
      ), "holds")
    }
  ],
  forbidden: [
    {
      id: "zero-duration-event",
      describe: "create_event must not have start == end",
      violatedBy: /* @__PURE__ */ __name((m) => lines(m).some((ln) => {
        const c = ln.match(/create_event\(.*start="([^"]+)".*end="([^"]+)"/);
        return !!c && c[1] === c[2];
      }), "violatedBy")
    },
    {
      id: "unordered-slot-window",
      describe: "find_slot must have after < before",
      violatedBy: /* @__PURE__ */ __name((m) => lines(m).some((ln) => {
        const f = ln.match(/find_slot\(.*after="([^"]+)".*before="([^"]+)"/);
        return !!f && !(f[1] < f[2]);
      }), "violatedBy")
    }
  ]
};

// src/skills/inbox-calendar/providers/google.ts
var GOOGLE_PROFILE = {
  provider: "google",
  label: "Google (Gmail + Calendar)",
  discovery: {
    source: [
      "https://gmail.googleapis.com/$discovery/rest?version=v1",
      "https://www.googleapis.com/discovery/v1/apis/calendar/v3/rest"
    ],
    revision: "2026-06-26-curated",
    note: "Curated subset. Times normalized to YYYY-MM-DDTHH:MM for the macro; Google uses RFC3339 (start.dateTime + timeZone). schedule_send/set_reminder have no clean public method \u2014 see opMap notes."
  },
  conventions: {
    timeFormat: "RFC3339",
    // events use start.dateTime/end.dateTime + timeZone; macro emits YYYY-MM-DDTHH:MM
    searchSyntax: "gmail-q"
    // from:, subject:, label:, after:, before:, has:
  },
  // canonical PORT op -> Google Discovery method id (write-layer target; not emitted in macros)
  opMap: {
    find_email: "gmail.users.messages.list",
    // q= search operators
    compose_email: "gmail.users.messages.send",
    reply_email: "gmail.users.messages.send",
    // threadId + In-Reply-To header
    forward_email: "gmail.users.messages.send",
    archive_email: "gmail.users.messages.modify",
    // removeLabelIds: [INBOX]
    label_email: "gmail.users.messages.modify",
    // addLabelIds: [<labelId>]
    schedule_send: "gmail.users.drafts.create",
    // no public scheduled-send method; client schedules the send
    create_event: "calendar.events.insert",
    set_reminder: "calendar.events.insert",
    // popup reminder override; Reminders API is not public
    find_slot: "calendar.freebusy.query",
    rsvp: "calendar.events.patch"
    // attendees[].responseStatus
  },
  pools: {
    people: ["mom", "Sarah", "Alex", "the design team", "my manager", "Priya", "John", "the landlord", "accounting", "Dana", "Marcus", "the recruiter"],
    topics: ["the Q3 roadmap", "the launch", "the budget", "onboarding", "the API redesign", "the offsite", "the bug report", "the contract", "the renewal", "the demo"],
    // each "when" carries the natural phrasing (for the request) + its ISO start (for the macro)
    whens: [
      { nat: "today at 5pm", iso: "2026-06-29T17:00" },
      { nat: "tomorrow at 9am", iso: "2026-06-30T09:00" },
      { nat: "Wednesday at 2pm", iso: "2026-07-01T14:00" },
      { nat: "Thursday at 4:30pm", iso: "2026-07-02T16:30" },
      { nat: "Friday at 11am", iso: "2026-07-03T11:00" },
      { nat: "next Monday at 10am", iso: "2026-07-06T10:00" },
      { nat: "tonight at 7pm", iso: "2026-06-29T19:00" }
    ],
    // search windows for find_slot — after STRICTLY before before
    windows: [
      { nat: "tomorrow afternoon", after: "2026-06-30T13:00", before: "2026-06-30T18:00" },
      { nat: "Wednesday morning", after: "2026-07-01T09:00", before: "2026-07-01T12:00" },
      { nat: "Friday afternoon", after: "2026-07-03T13:00", before: "2026-07-03T17:00" },
      { nat: "sometime Thursday", after: "2026-07-02T09:00", before: "2026-07-02T18:00" }
    ],
    labels: ["housing", "urgent", "finance", "travel", "follow-up", "receipts"],
    durations: [30, 45, 60],
    rsvps: [
      { resp: "yes", verb: "rsvp yes to" },
      { resp: "no", verb: "decline" },
      { resp: "maybe", verb: "tentatively accept" }
    ]
  }
};

// src/skills/inbox-calendar/intents.ts
var INTENTS = [
  {
    n: 8,
    draw: ["person", "topic"],
    phrasings: [
      "email ${person} about ${topic}",
      "ping ${person} about ${topic}",
      "shoot ${person} a quick note on ${topic}",
      "draft a message to ${person} re ${topic}"
    ],
    macro: 'compose_email(to="${person}", subject="${topic}", body="Quick note about ${topic} \u2014 let me know your thoughts.")'
  },
  {
    n: 7,
    draw: ["person", "topic"],
    phrasings: [
      "find the email from ${person} about ${topic}",
      "pull up ${person}'s message on ${topic}",
      "search my inbox for ${topic} from ${person}"
    ],
    macro: 'find_email(query="from:${person} ${topic}")'
  },
  {
    n: 7,
    draw: ["person", "topic", "when"],
    phrasings: [
      "reply to ${person}'s email about ${topic} that I'll review it by ${when.nat}",
      "tell ${person} in the ${topic} thread I'll get back by ${when.nat}"
    ],
    macro: 't = find_email(query="from:${person} ${topic}")\nreply_email(thread=t, body="Thanks \u2014 I\'ll review this by ${when.iso}.")'
  },
  {
    n: 6,
    draw: ["person", "topic"],
    phrasings: [
      "forward the ${topic} email to ${person}",
      "send ${person} the ${topic} thread for their records"
    ],
    macro: 't = find_email(query="${topic}")\nforward_email(thread=t, to="${person}", note="FYI \u2014 for your records.")'
  },
  {
    n: 6,
    draw: ["topic"],
    phrasings: [
      "archive the emails about ${topic}",
      "clear out the ${topic} threads",
      "archive everything about ${topic}"
    ],
    macro: 't = find_email(query="${topic}")\narchive_email(thread=t)'
  },
  {
    n: 6,
    draw: ["person", "label"],
    phrasings: [
      "label ${person}'s email as ${label}",
      "tag the message from ${person} ${label}",
      "mark ${person}'s thread ${label}"
    ],
    macro: 't = find_email(query="from:${person}")\nlabel_email(thread=t, label="${label}")'
  },
  {
    n: 6,
    draw: ["person", "topic", "when"],
    phrasings: [
      "schedule a thank-you to ${person} for ${topic}, send it ${when.nat}",
      "queue a note to ${person} about ${topic} to go out ${when.nat}"
    ],
    macro: 'schedule_send(to="${person}", subject="Thank you", body="Thanks for ${topic}.", when="${when.iso}")'
  },
  {
    n: 9,
    draw: ["person", "topic", "when", "dur"],
    phrasings: [
      "set up a ${dur}-minute meeting about ${topic} with ${person} ${when.nat}",
      "book ${dur} minutes with ${person} on ${topic} ${when.nat}",
      "put a ${dur}-min ${topic} sync with ${person} on my calendar ${when.nat}"
    ],
    macro: 'create_event(title="${topic} with ${person}", start="${when.iso}", end="${end}", remind_min=10)'
  },
  {
    n: 6,
    draw: ["topic", "when"],
    phrasings: [
      "remind me to follow up on ${topic} ${when.nat}",
      "set a reminder about ${topic} for ${when.nat}"
    ],
    macro: 'set_reminder(text="Follow up on ${topic}", when="${when.iso}")'
  },
  {
    n: 8,
    draw: ["topic", "window", "dur"],
    phrasings: [
      "find a ${dur}-minute slot ${window.nat} and book ${topic}",
      "grab ${dur} minutes ${window.nat} for ${topic}"
    ],
    macro: 's = find_slot(duration_min=${dur}, after="${window.after}", before="${window.before}")\ncreate_event(title="${topic}", start=s.start, end=s.end, remind_min=10)'
  },
  {
    n: 7,
    draw: ["topic", "rsvp"],
    phrasings: [
      "${rsvp.verb} the ${topic} invite",
      "respond ${rsvp.resp} to the ${topic} meeting invite"
    ],
    macro: 't = find_email(query="${topic} invite")\nrsvp(event=t, response="${rsvp.resp}")'
  }
];
var OOS = [
  "order me a pizza",
  "what is the capital of France?",
  "play some jazz",
  "book me a flight to Tokyo",
  "summarize my entire inbox",
  "translate this email to French",
  "unsubscribe me from all newsletters",
  "what's the weather tomorrow?"
].map((q) => [q, "OUT_OF_SCOPE"]);

// src/skills/inbox-calendar/generate.ts
function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
__name(hashStr, "hashStr");
function mulberry32(a) {
  return function() {
    a |= 0;
    a = a + 1831565813 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
__name(mulberry32, "mulberry32");
function isoAdd(iso, mins) {
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/);
  const t = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) + mins * 6e4);
  const p = /* @__PURE__ */ __name((n) => String(n).padStart(2, "0"), "p");
  return `${t.getUTCFullYear()}-${p(t.getUTCMonth() + 1)}-${p(t.getUTCDate())}T${p(t.getUTCHours())}:${p(t.getUTCMinutes())}`;
}
__name(isoAdd, "isoAdd");
function fill(tpl, ctx) {
  return tpl.replace(/\$\{([^}]+)\}/g, (_, k) => k in ctx ? ctx[k] : `\${${k}}`);
}
__name(fill, "fill");
function poolFor(slot, profile) {
  const p = profile.pools;
  switch (slot) {
    case "person":
      return p.people;
    case "topic":
      return p.topics;
    case "when":
      return p.whens;
    case "window":
      return p.windows;
    case "label":
      return p.labels;
    case "dur":
      return p.durations;
    case "rsvp":
      return p.rsvps;
  }
}
__name(poolFor, "poolFor");
function generateCorpus(seed, profile, intents, oos) {
  const rng = mulberry32(hashStr(seed));
  const pick = /* @__PURE__ */ __name((a) => a[Math.floor(rng() * a.length)], "pick");
  const makeOne = /* @__PURE__ */ __name((intent) => {
    const raw = {};
    for (const slot of intent.draw) raw[slot] = pick(poolFor(slot, profile));
    const ctx = {};
    if ("person" in raw) ctx.person = raw.person;
    if ("topic" in raw) ctx.topic = raw.topic;
    if ("label" in raw) ctx.label = raw.label;
    if ("dur" in raw) ctx.dur = String(raw.dur);
    if ("when" in raw) {
      ctx["when.nat"] = raw.when.nat;
      ctx["when.iso"] = raw.when.iso;
    }
    if ("window" in raw) {
      ctx["window.nat"] = raw.window.nat;
      ctx["window.after"] = raw.window.after;
      ctx["window.before"] = raw.window.before;
    }
    if ("rsvp" in raw) {
      ctx["rsvp.resp"] = raw.rsvp.resp;
      ctx["rsvp.verb"] = raw.rsvp.verb;
    }
    if ("when" in raw && "dur" in raw) ctx.end = isoAdd(raw.when.iso, raw.dur);
    const request = fill(pick(intent.phrasings), ctx);
    return [request, fill(intent.macro, ctx)];
  }, "makeOne");
  const seen = /* @__PURE__ */ new Set();
  const all = [];
  for (const intent of intents) {
    let made = 0, tries = 0;
    while (made < intent.n && tries < intent.n * 16) {
      tries++;
      const pair = makeOne(intent);
      if (seen.has(pair[0])) continue;
      seen.add(pair[0]);
      all.push(pair);
      made++;
    }
  }
  const examples = [], evals = [];
  all.forEach((p, i) => (i % 5 === 4 ? evals : examples).push(p));
  oos.forEach((q, i) => (i % 4 === 3 ? evals : examples).push(q));
  return { examples, eval: evals };
}
__name(generateCorpus, "generateCorpus");

// src/skills/inbox-calendar/adapters/google.ts
var SEED = "inbox-calendar:v2-iso";
function genCalendar() {
  return generateCorpus(SEED, GOOGLE_PROFILE, INTENTS, OOS);
}
__name(genCalendar, "genCalendar");

// src/skills/inbox-calendar/index.ts
var calendarDef = {
  key: META.key,
  label: META.label,
  icon: META.icon,
  domain: DOMAIN,
  scope: SCOPE,
  desc: META.desc,
  suggest: META.suggest,
  ops: OPS,
  context: CONTEXT,
  examplesFn: genCalendar,
  contract: CALENDAR_CONTRACT
};

// src/skills.js
function specSig(spec) {
  return spec.ops.map((o) => `${o.name}(${(o.params || []).join(", ")})${o.ret ? " -> " + o.ret : ""}`).join("; ");
}
__name(specSig, "specSig");
function skillSystem(domain, spec, context) {
  return `You are ${domain}. Convert the request into a macro using ONLY these operations:
` + specSig(spec) + ".\n" + (context ? context + "\n" : "") + `Output ONLY the macro, one call per line, no prose. If the request is outside ${spec.scope}, output exactly: OUT_OF_SCOPE.`;
}
__name(skillSystem, "skillSystem");
function parseMacroCalls(text) {
  const out = [];
  for (const raw of String(text).split("\n")) {
    const line = raw.trim();
    if (!line || line === "OUT_OF_SCOPE") continue;
    const m = line.match(/^(?:[A-Za-z_]\w*\s*=\s*)?([A-Za-z_]\w*)\s*\((.*)\)\s*;?\s*$/);
    if (!m) continue;
    const keys = [...m[2].matchAll(/(?:^|,)\s*([A-Za-z_]\w*)\s*=/g)].map((k) => k[1]);
    out.push({ op: m[1], keys });
  }
  return out;
}
__name(parseMacroCalls, "parseMacroCalls");
function verifyMacro(text, spec) {
  const t = String(text);
  const calls = parseMacroCalls(t);
  const bounced = /(^|\n)\s*OUT_OF_SCOPE\s*($|\n)/.test(t) && calls.length === 0;
  if (bounced) return { status: "oos", calls: [], issues: [], n: 0 };
  if (!calls.length) return { status: "empty", calls: [], issues: [], n: 0 };
  const byName = new Map(spec.ops.map((o) => [o.name, o]));
  const issues = [];
  const detail = [];
  for (const c of calls) {
    const op = byName.get(c.op);
    if (!op) {
      issues.push(`unknown op: ${c.op}`);
      detail.push({ op: c.op, ok: false });
      continue;
    }
    const allowed = new Set(op.params || []);
    const bad = c.keys.filter((k) => !allowed.has(k));
    if (bad.length) {
      issues.push(`${c.op}: unexpected arg ${bad.join(", ")}`);
      detail.push({ op: c.op, ok: false });
    } else detail.push({ op: c.op, ok: true });
  }
  return { status: issues.length ? "bad" : "ok", calls: detail, issues, n: calls.length };
}
__name(verifyMacro, "verifyMacro");
var BASE_ASSERTIONS = [{
  id: "spec-valid",
  describe: "every call uses a spec op with only that op\u2019s params, or the macro is a clean OUT_OF_SCOPE bounce",
  holds: /* @__PURE__ */ __name((m, spec) => {
    const r = verifyMacro(m, spec);
    return r.status === "ok" || r.status === "oos";
  }, "holds")
}];
function buildContract(def) {
  const extra = def.contract || {};
  return {
    block: def.key,
    assertions: [...BASE_ASSERTIONS, ...extra.assertions || []],
    forbidden: [...extra.forbidden || []]
  };
}
__name(buildContract, "buildContract");
function hashStr2(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
__name(hashStr2, "hashStr");
function mulberry322(a) {
  return function() {
    a |= 0;
    a = a + 1831565813 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
__name(mulberry322, "mulberry32");
function fill2(tpl, choice) {
  return tpl.replace(/\{(\w+)\}/g, (_, k) => k in choice ? choice[k] : "{" + k + "}");
}
__name(fill2, "fill");
function expand(def, perTemplate) {
  const rnd = mulberry322(hashStr2(def.key));
  const out = [];
  const seen = /* @__PURE__ */ new Set();
  for (const t of def.templates || []) {
    const slots = [...new Set([...t.req.matchAll(/\{(\w+)\}/g)].map((m) => m[1]))];
    let made = 0, tries = 0;
    const cap = perTemplate * 8;
    while (made < perTemplate && tries < cap) {
      tries++;
      const choice = {};
      for (const s of slots) {
        const arr = def.vocab[s] || ["x"];
        choice[s] = arr[Math.floor(rnd() * arr.length)];
      }
      const req = fill2(t.req, choice);
      if (seen.has(req)) continue;
      seen.add(req);
      out.push([req, fill2(t.macro, choice)]);
      made++;
    }
  }
  return out;
}
__name(expand, "expand");
function buildSkill(def, perTemplate = 6) {
  const spec = { scope: def.scope, ops: def.ops };
  let examples, evals = [];
  if (typeof def.examplesFn === "function") {
    const g = def.examplesFn();
    examples = g.examples;
    evals = g.eval || [];
  } else {
    examples = [
      ...def.fixed || [],
      ...expand(def, perTemplate),
      ...(def.oos || []).map((r) => [r, "OUT_OF_SCOPE"])
    ];
  }
  return {
    key: def.key,
    label: def.label,
    icon: def.icon,
    desc: def.desc,
    domain: def.domain,
    spec,
    context: def.context || "",
    system: skillSystem(def.domain, spec, def.context),
    suggest: def.suggest,
    examples,
    eval: evals,
    contract: buildContract(def)
  };
}
__name(buildSkill, "buildSkill");
var WHENS = ["today 17:00", "tomorrow 09:00", "Friday 14:00", "next Monday 10:00", "Thursday 16:30", "tonight 19:00"];
var DEFS = [
  calendarDef,
  // flagship block: src/skills/inbox-calendar/{port,contract,adapters/google,manifest}.ts
  {
    key: "music",
    label: "Music",
    icon: "\u266A",
    domain: "a music player operator",
    scope: "music playback",
    desc: "Turns \u201Cplay some lo-fi and turn it down\u201D into a macro over a music action space \u2014 find/play/queue/volume/playlist \u2014 and bounces non-music asks.",
    suggest: "Play something upbeat for cooking and add it to a new playlist called Dinner.",
    ops: [
      { name: "find_track", params: ["query"], ret: "track" },
      { name: "play_track", params: ["track"] },
      { name: "queue_track", params: ["track"] },
      { name: "pause", params: [] },
      { name: "skip", params: [] },
      { name: "previous", params: [] },
      { name: "set_volume", params: ["level"] },
      { name: "create_playlist", params: ["name"] },
      { name: "add_to_playlist", params: ["playlist", "track"] },
      { name: "shuffle", params: ["on"] },
      { name: "repeat", params: ["mode"] }
    ],
    fixed: [
      ["skip this song", "skip()"],
      ["pause the music", "pause()"],
      ["go back to the previous song", "previous()"]
    ],
    templates: [
      { req: "play some {genre}", macro: 't = find_track(query="{genre}")\nplay_track(track=t)' },
      { req: "queue up {artist} after this", macro: 't = find_track(query="{artist}")\nqueue_track(track=t)' },
      { req: "set the volume to {vol}", macro: "set_volume(level={vol})" },
      { req: "make a playlist called {name}", macro: 'create_playlist(name="{name}")' },
      { req: "add {artist} to my {name} playlist", macro: 't = find_track(query="{artist}")\nadd_to_playlist(playlist="{name}", track=t)' },
      { req: "shuffle my {name} playlist", macro: 'shuffle(on=true)\nt = find_track(query="playlist:{name}")\nplay_track(track=t)' },
      { req: "put on {artist} and turn it up", macro: 't = find_track(query="{artist}")\nplay_track(track=t)\nset_volume(level=80)' },
      { req: "repeat this {mode}", macro: 'repeat(mode="{mode}")' }
    ],
    vocab: {
      genre: ["lo-fi beats", "deep house", "classic jazz", "pop hits", "ambient", "classical", "90s hip hop", "indie rock"],
      artist: ["Taylor Swift", "The Beatles", "Daft Punk", "Miles Davis", "Radiohead", "Bad Bunny", "Fleetwood Mac"],
      name: ["Focus", "Workout", "Dinner", "Chill", "Road Trip", "Sleep"],
      vol: ["10", "25", "40", "60", "75", "90"],
      mode: ["one", "all"]
    },
    oos: ["email my boss", "what is the weather today?", "open an issue on the repo"]
  },
  {
    key: "github",
    label: "GitHub",
    icon: "\u{1F419}",
    domain: "a GitHub operator",
    scope: "GitHub repositories, issues, and pull requests",
    desc: "Compiles dev requests into a macro over issues, pull requests, and repos; bounces anything that isn\u2019t GitHub.",
    suggest: 'Open an issue on the api repo titled "fix login redirect", then assign it to Dana.',
    ops: [
      { name: "find_issue", params: ["query"], ret: "issue" },
      { name: "create_issue", params: ["repo", "title", "body"] },
      { name: "comment_issue", params: ["issue", "body"] },
      { name: "close_issue", params: ["issue"] },
      { name: "assign_issue", params: ["issue", "assignee"] },
      { name: "label_issue", params: ["issue", "label"] },
      { name: "find_pr", params: ["query"], ret: "pr" },
      { name: "open_pr", params: ["repo", "title", "branch"] },
      { name: "review_pr", params: ["pr", "verdict"] },
      { name: "merge_pr", params: ["pr"] },
      { name: "create_repo", params: ["name", "visibility"] },
      { name: "star_repo", params: ["repo"] }
    ],
    fixed: [
      [
        "open an issue on the api repo titled fix login redirect and assign it to Dana",
        'i = create_issue(repo="api", title="fix login redirect", body="The login flow redirects to the wrong page.")\nassign_issue(issue=i, assignee="Dana")'
      ]
    ],
    templates: [
      { req: "open an issue on {repo} titled {title}", macro: 'create_issue(repo="{repo}", title="{title}", body="{title}.")' },
      { req: "close the {topic} issue", macro: 'i = find_issue(query="{topic}")\nclose_issue(issue=i)' },
      { req: "comment {comment} on the {topic} issue", macro: 'i = find_issue(query="{topic}")\ncomment_issue(issue=i, body="{comment}")' },
      { req: "assign the {topic} issue to {user}", macro: 'i = find_issue(query="{topic}")\nassign_issue(issue=i, assignee="{user}")' },
      { req: "label the {topic} issue as {label}", macro: 'i = find_issue(query="{topic}")\nlabel_issue(issue=i, label="{label}")' },
      { req: "open a pull request on {repo} from {branch} titled {title}", macro: 'open_pr(repo="{repo}", title="{title}", branch="{branch}")' },
      { req: "approve the {topic} pull request", macro: 'p = find_pr(query="{topic}")\nreview_pr(pr=p, verdict="approve")' },
      { req: "merge the {topic} PR", macro: 'p = find_pr(query="{topic}")\nmerge_pr(pr=p)' },
      { req: "create a private repo called {repo}", macro: 'create_repo(name="{repo}", visibility="private")' },
      { req: "star the {repo} repo", macro: 'star_repo(repo="{repo}")' }
    ],
    vocab: {
      repo: ["api", "frontend", "docs", "infra", "mobile-app", "design-system"],
      title: ["fix login redirect", "add dark mode", "update README", "flaky test fix", "bump dependencies", "improve error logs"],
      topic: ["login", "dark mode", "flaky test", "memory leak", "rate limiting", "docs typo"],
      comment: ["looks good to me", "can you add a test?", "I will pick this up", "reproduced on main", "duplicate of #42"],
      user: ["Dana", "Alex", "Priya", "the on-call", "Sam"],
      label: ["bug", "enhancement", "good first issue", "p1", "docs", "wontfix"],
      branch: ["feature/auth", "fix/cache", "chore/deps", "feat/ui", "hotfix/crash"]
    },
    oos: ["play some music", "email my mom", "what is 2 + 2?"]
  },
  {
    key: "slack",
    label: "Slack",
    icon: "\u{1F4AC}",
    domain: "a Slack operator",
    scope: "Slack messaging",
    desc: "Compiles team-chat requests into a macro over channels, DMs, threads, and reminders; bounces non-Slack asks.",
    suggest: "Post the release notes in #launch and DM Dana to review them.",
    ops: [
      { name: "find_message", params: ["query"], ret: "message" },
      { name: "send_message", params: ["channel", "text"] },
      { name: "dm", params: ["user", "text"] },
      { name: "reply_thread", params: ["message", "text"] },
      { name: "react", params: ["message", "emoji"] },
      { name: "set_status", params: ["text", "emoji"] },
      { name: "create_channel", params: ["name"] },
      { name: "invite", params: ["user", "channel"] },
      { name: "remind", params: ["text", "when"] },
      { name: "pin", params: ["message"] }
    ],
    fixed: [
      [
        "post the release notes in #launch and dm Dana to review them",
        'send_message(channel="launch", text="Release notes are up \u2014 please review.")\ndm(user="Dana", text="Can you review the release notes I posted in #launch?")'
      ]
    ],
    templates: [
      { req: "post {text} in #{channel}", macro: 'send_message(channel="{channel}", text="{text}")' },
      { req: "dm {user} {text}", macro: 'dm(user="{user}", text="{text}")' },
      { req: "reply {text} to the {topic} thread", macro: 'm = find_message(query="{topic}")\nreply_thread(message=m, text="{text}")' },
      { req: "react {emoji} to the {topic} message", macro: 'm = find_message(query="{topic}")\nreact(message=m, emoji="{emoji}")' },
      { req: "set my status to {text}", macro: 'set_status(text="{text}", emoji="{emoji}")' },
      { req: "create a channel called {channel}", macro: 'create_channel(name="{channel}")' },
      { req: "invite {user} to #{channel}", macro: 'invite(user="{user}", channel="{channel}")' },
      { req: "remind the team to {task} {when}", macro: 'remind(text="{task}", when="{when}")' },
      { req: "pin the {topic} message", macro: 'm = find_message(query="{topic}")\npin(message=m)' }
    ],
    vocab: {
      channel: ["launch", "general", "engineering", "design", "random", "incidents"],
      user: ["Dana", "Alex", "Priya", "Sam", "the team lead"],
      text: ["standup in 5", "PR is ready for review", "deploy is green", "lunch at noon?", "great work today"],
      topic: ["deploy", "incident", "roadmap", "lunch", "release"],
      emoji: [":eyes:", ":white_check_mark:", ":tada:", ":fire:", ":+1:"],
      task: ["submit timesheets", "join the retro", "review the doc", "update the board"],
      when: WHENS
    },
    oos: ["play a song", "order groceries", "what time is it in Tokyo?"]
  },
  {
    key: "notion",
    label: "Notion",
    icon: "\u{1F4DD}",
    domain: "a Notion operator",
    scope: "Notion pages, notes, and tasks",
    desc: "Compiles note-taking requests into a macro over pages, blocks, tasks, and databases; bounces anything else.",
    suggest: 'Create a page titled "Trip plan" and add a task to book flights due Friday.',
    ops: [
      { name: "find_page", params: ["query"], ret: "page" },
      { name: "create_page", params: ["title", "body"] },
      { name: "append_block", params: ["page", "text"] },
      { name: "create_task", params: ["title", "due"] },
      { name: "complete_task", params: ["task"] },
      { name: "find_task", params: ["query"], ret: "task" },
      { name: "add_to_database", params: ["database", "name"] },
      { name: "set_property", params: ["page", "key", "value"] },
      { name: "create_database", params: ["name"] }
    ],
    fixed: [
      [
        "create a page titled Trip plan and add a task to book flights due Friday",
        'create_page(title="Trip plan", body="Planning notes.")\ncreate_task(title="Book flights", due="Friday")'
      ]
    ],
    templates: [
      { req: "create a page titled {title}", macro: 'create_page(title="{title}", body="{title} \u2014 notes.")' },
      { req: "add a note {text} to the {topic} page", macro: 'p = find_page(query="{topic}")\nappend_block(page=p, text="{text}")' },
      { req: "add a task to {task} due {when}", macro: 'create_task(title="{task}", due="{when}")' },
      { req: "mark the {task} task done", macro: 't = find_task(query="{task}")\ncomplete_task(task=t)' },
      { req: "add {name} to my {database} database", macro: 'add_to_database(database="{database}", name="{name}")' },
      { req: "set the status of the {topic} page to {value}", macro: 'p = find_page(query="{topic}")\nset_property(page=p, key="status", value="{value}")' },
      { req: "create a database called {database}", macro: 'create_database(name="{database}")' }
    ],
    vocab: {
      title: ["Trip plan", "Q3 goals", "Reading list", "Meeting notes", "Project brief", "Recipes"],
      text: ["remember to confirm the budget", "add the agenda", "link the spec", "note the blockers"],
      topic: ["trip", "goals", "project", "meeting", "reading"],
      task: ["book flights", "draft the brief", "email the vendor", "review the PR", "pay the invoice"],
      when: ["today", "tomorrow", "Friday", "next week", "end of month"],
      name: ["Acme Co", "Q3 launch", "Vendor X", "Idea: dark mode"],
      database: ["Projects", "CRM", "Tasks", "Reading", "Inventory"],
      value: ["in progress", "done", "blocked", "todo", "review"]
    },
    oos: ["play music", "navigate home", "send a tweet"]
  },
  {
    key: "x",
    label: "X",
    icon: "\u{1D54F}",
    domain: "an X (Twitter) operator",
    scope: "posting and engagement on X",
    desc: "Compiles social requests into a macro over posts, replies, reposts, follows, and DMs; bounces anything off-platform.",
    suggest: 'Post "shipping something fun today \u{1F680}" and schedule a follow-up for 5pm.',
    ops: [
      { name: "find_post", params: ["query"], ret: "post" },
      { name: "post", params: ["text"] },
      { name: "reply", params: ["post", "text"] },
      { name: "repost", params: ["post"] },
      { name: "like", params: ["post"] },
      { name: "follow", params: ["user"] },
      { name: "dm", params: ["user", "text"] },
      { name: "schedule_post", params: ["text", "when"] },
      { name: "bookmark", params: ["post"] }
    ],
    fixed: [
      [
        "post shipping something fun today and schedule a follow up for 5pm",
        'post(text="shipping something fun today \u{1F680}")\nschedule_post(text="more details soon \u2014 stay tuned", when="today 17:00")'
      ]
    ],
    templates: [
      { req: "post {text}", macro: 'post(text="{text}")' },
      { req: "reply {text} to the {topic} post", macro: 'p = find_post(query="{topic}")\nreply(post=p, text="{text}")' },
      { req: "repost the {topic} tweet", macro: 'p = find_post(query="{topic}")\nrepost(post=p)' },
      { req: "like the {topic} post", macro: 'p = find_post(query="{topic}")\nlike(post=p)' },
      { req: "follow {user}", macro: 'follow(user="{user}")' },
      { req: "dm {user} {text}", macro: 'dm(user="{user}", text="{text}")' },
      { req: "schedule a post {when} saying {text}", macro: 'schedule_post(text="{text}", when="{when}")' },
      { req: "bookmark the {topic} thread", macro: 'p = find_post(query="{topic}")\nbookmark(post=p)' }
    ],
    vocab: {
      text: ["gm", "big news coming", "loved this talk", "hot take: tabs > spaces", "thanks for 10k followers"],
      topic: ["the launch", "the keynote", "the meme", "the thread on AI", "the announcement"],
      user: ["@levelsio", "@naval", "@swyx", "@dhh", "@karpathy"],
      when: WHENS
    },
    oos: ["archive my inbox", "play a playlist", "open a GitHub issue"]
  },
  {
    key: "instagram",
    label: "Instagram",
    icon: "\u{1F4F7}",
    domain: "an Instagram operator",
    scope: "Instagram posts, stories, and DMs",
    desc: "Compiles requests into a macro over photo posts, stories, comments, and DMs; bounces anything off-platform.",
    suggest: 'Post a photo with caption "sunset run \u{1F305}" and share it to my story.',
    ops: [
      { name: "find_post", params: ["query"], ret: "post" },
      { name: "post_photo", params: ["caption", "media"] },
      { name: "post_story", params: ["media"] },
      { name: "reply_dm", params: ["user", "text"] },
      { name: "like_post", params: ["post"] },
      { name: "comment", params: ["post", "text"] },
      { name: "follow", params: ["user"] },
      { name: "save_post", params: ["post"] }
    ],
    fixed: [
      [
        "post a photo with caption sunset run and share it to my story",
        'post_photo(caption="sunset run \u{1F305}", media="latest")\npost_story(media="latest")'
      ]
    ],
    templates: [
      { req: "post a photo with caption {caption}", macro: 'post_photo(caption="{caption}", media="latest")' },
      { req: "share {media} to my story", macro: 'post_story(media="{media}")' },
      { req: "comment {text} on the {topic} post", macro: 'p = find_post(query="{topic}")\ncomment(post=p, text="{text}")' },
      { req: "like the {topic} post", macro: 'p = find_post(query="{topic}")\nlike_post(post=p)' },
      { req: "reply {text} to {user} in DMs", macro: 'reply_dm(user="{user}", text="{text}")' },
      { req: "follow {user}", macro: 'follow(user="{user}")' },
      { req: "save the {topic} post", macro: 'p = find_post(query="{topic}")\nsave_post(post=p)' }
    ],
    vocab: {
      caption: ["sunset run \u{1F305}", "weekend vibes", "new kicks \u{1F45F}", "homemade pasta \u{1F35D}", "trail day"],
      media: ["latest", "the beach photo", "the reel", "the carousel"],
      text: ["love this!", "where is this?", "so good \u{1F525}", "congrats!", "need the recipe"],
      topic: ["the travel", "the food", "the fit check", "the puppy", "the launch"],
      user: ["@natgeo", "@nike", "@a_friend", "@the_chef"]
    },
    oos: ["merge the pull request", "set a reminder", "navigate to work"]
  },
  {
    key: "youtube",
    label: "YouTube",
    icon: "\u25B6",
    domain: "a YouTube operator",
    scope: "YouTube playback and library",
    desc: "Compiles requests into a macro over search, playback, playlists, and subscriptions; bounces anything else.",
    suggest: "Play a 10-minute beginner yoga video and add it to my Morning playlist.",
    ops: [
      { name: "find_video", params: ["query"], ret: "video" },
      { name: "play_video", params: ["video"] },
      { name: "queue_video", params: ["video"] },
      { name: "subscribe", params: ["channel"] },
      { name: "like_video", params: ["video"] },
      { name: "add_to_playlist", params: ["playlist", "video"] },
      { name: "create_playlist", params: ["name"] },
      { name: "comment", params: ["video", "text"] }
    ],
    fixed: [
      [
        "play a beginner yoga video and add it to my Morning playlist",
        'v = find_video(query="beginner yoga 10 minutes")\nplay_video(video=v)\nadd_to_playlist(playlist="Morning", video=v)'
      ]
    ],
    templates: [
      { req: "play a video about {query}", macro: 'v = find_video(query="{query}")\nplay_video(video=v)' },
      { req: "queue a video about {query}", macro: 'v = find_video(query="{query}")\nqueue_video(video=v)' },
      { req: "subscribe to {channel}", macro: 'subscribe(channel="{channel}")' },
      { req: "like the {query} video", macro: 'v = find_video(query="{query}")\nlike_video(video=v)' },
      { req: "add a {query} video to my {name} playlist", macro: 'v = find_video(query="{query}")\nadd_to_playlist(playlist="{name}", video=v)' },
      { req: "make a playlist called {name}", macro: 'create_playlist(name="{name}")' },
      { req: "comment {text} on the {query} video", macro: 'v = find_video(query="{query}")\ncomment(video=v, text="{text}")' }
    ],
    vocab: {
      query: ["lo-fi study mix", "rust tutorial", "marathon training", "pasta recipe", "guitar lesson", "space documentary"],
      channel: ["Veritasium", "Fireship", "MKBHD", "Kurzgesagt", "NileRed"],
      name: ["Morning", "Watch Later", "Cooking", "Workouts", "Learning"],
      text: ["great explanation!", "first", "this helped a lot", "please do a part 2"]
    },
    oos: ["email the team", "open a PR", "set my Slack status"]
  },
  {
    key: "maps",
    label: "Maps",
    icon: "\u{1F4CD}",
    domain: "a Maps operator",
    scope: "navigation and places",
    desc: "Compiles requests into a macro over places, directions, and navigation; bounces anything off-map.",
    suggest: "Find the nearest coffee shop and start navigation, then share my ETA with Alex.",
    ops: [
      { name: "search_place", params: ["query"], ret: "place" },
      { name: "find_nearby", params: ["category"], ret: "place" },
      { name: "directions", params: ["to", "mode"] },
      { name: "start_navigation", params: ["place"] },
      { name: "save_place", params: ["place", "list"] },
      { name: "share_eta", params: ["place", "contact"] }
    ],
    fixed: [
      [
        "find the nearest coffee shop and start navigation then share my eta with Alex",
        'p = find_nearby(category="coffee shop")\nstart_navigation(place=p)\nshare_eta(place=p, contact="Alex")'
      ]
    ],
    templates: [
      { req: "navigate to {place}", macro: 'p = search_place(query="{place}")\nstart_navigation(place=p)' },
      { req: "directions to {place} by {mode}", macro: 'directions(to="{place}", mode="{mode}")' },
      { req: "find a {category} near me", macro: 'find_nearby(category="{category}")' },
      { req: "find the nearest {category} and navigate there", macro: 'p = find_nearby(category="{category}")\nstart_navigation(place=p)' },
      { req: "save {place} to my {list} list", macro: 'p = search_place(query="{place}")\nsave_place(place=p, list="{list}")' },
      { req: "share my ETA to {place} with {contact}", macro: 'p = search_place(query="{place}")\nshare_eta(place=p, contact="{contact}")' }
    ],
    vocab: {
      place: ["the airport", "downtown", "the office", "Central Park", "the train station", "the stadium"],
      mode: ["driving", "walking", "transit", "cycling"],
      category: ["coffee shop", "gas station", "pharmacy", "grocery store", "ATM", "parking"],
      list: ["Favorites", "Want to go", "Trip", "Restaurants"],
      contact: ["Alex", "mom", "Dana", "the group"]
    },
    oos: ["post a tweet", "play a song", "create a GitHub repo"]
  },
  {
    key: "amazon",
    label: "Shopping",
    icon: "\u{1F6D2}",
    domain: "a shopping operator",
    scope: "shopping cart and orders",
    desc: "Compiles requests into a macro over product search, cart, orders, and lists; bounces anything that isn\u2019t shopping.",
    suggest: "Add two packs of AA batteries to my cart and track my last order.",
    ops: [
      { name: "search_product", params: ["query"], ret: "product" },
      { name: "add_to_cart", params: ["product", "qty"] },
      { name: "buy_now", params: ["product"] },
      { name: "find_order", params: ["query"], ret: "order" },
      { name: "track_order", params: ["order"], ret: "status" },
      { name: "reorder", params: ["query"] },
      { name: "add_to_list", params: ["product", "list"] }
    ],
    fixed: [
      [
        "add two packs of AA batteries to my cart and track my last order",
        'p = search_product(query="AA batteries 2 pack")\nadd_to_cart(product=p, qty=2)\no = find_order(query="last order")\ntrack_order(order=o)'
      ]
    ],
    templates: [
      { req: "add {qty} {product} to my cart", macro: 'p = search_product(query="{product}")\nadd_to_cart(product=p, qty={qty})' },
      { req: "buy {product} now", macro: 'p = search_product(query="{product}")\nbuy_now(product=p)' },
      { req: "reorder {product}", macro: 'reorder(query="{product}")' },
      { req: "track my {product} order", macro: 'o = find_order(query="{product}")\ntrack_order(order=o)' },
      { req: "add {product} to my {list} list", macro: 'p = search_product(query="{product}")\nadd_to_list(product=p, list="{list}")' },
      { req: "search for {product}", macro: 'search_product(query="{product}")' }
    ],
    vocab: {
      product: ["AA batteries", "USB-C cable", "olive oil", "running shoes", "paper towels", "a coffee grinder", "phone case"],
      qty: ["1", "2", "3", "4"],
      list: ["Wishlist", "Subscribe & Save", "Home", "Gifts"]
    },
    oos: ["send an email", "play a video", "navigate to the office"]
  },
  {
    key: "reddit",
    label: "Reddit",
    icon: "\u{1F47D}",
    domain: "a Reddit operator",
    scope: "Reddit posts and comments",
    desc: "Compiles requests into a macro over submissions, comments, votes, and subscriptions; bounces anything off-platform.",
    suggest: 'Post "What mechanical keyboard should I buy?" to r/keyboards and subscribe.',
    ops: [
      { name: "find_post", params: ["query"], ret: "post" },
      { name: "submit_post", params: ["subreddit", "title", "body"] },
      { name: "comment", params: ["post", "text"] },
      { name: "upvote", params: ["post"] },
      { name: "reply_comment", params: ["comment", "text"] },
      { name: "subscribe", params: ["subreddit"] },
      { name: "save_post", params: ["post"] }
    ],
    fixed: [
      [
        "post what mechanical keyboard should I buy to r/keyboards and subscribe",
        'submit_post(subreddit="keyboards", title="What mechanical keyboard should I buy?", body="Budget is flexible \u2014 looking for recommendations.")\nsubscribe(subreddit="keyboards")'
      ]
    ],
    templates: [
      { req: "post {title} to r/{subreddit}", macro: 'submit_post(subreddit="{subreddit}", title="{title}", body="{title}")' },
      { req: "comment {text} on the {topic} post", macro: 'p = find_post(query="{topic}")\ncomment(post=p, text="{text}")' },
      { req: "upvote the {topic} post", macro: 'p = find_post(query="{topic}")\nupvote(post=p)' },
      { req: "subscribe to r/{subreddit}", macro: 'subscribe(subreddit="{subreddit}")' },
      { req: "save the {topic} post", macro: 'p = find_post(query="{topic}")\nsave_post(post=p)' }
    ],
    vocab: {
      subreddit: ["keyboards", "programming", "AskReddit", "buildapc", "cooking", "fitness"],
      title: ["What keyboard should I buy?", "Best beginner setup?", "How do I start running?", "Favorite pasta recipe?"],
      text: ["this is the way", "underrated take", "source?", "thanks for sharing", "happy cake day"],
      topic: ["the keyboard", "the build", "the recipe", "the AMA", "the discussion"]
    },
    oos: ["email my mom", "play a song", "navigate home"]
  },
  {
    key: "linkedin",
    label: "LinkedIn",
    icon: "\u{1F4BC}",
    domain: "a LinkedIn operator",
    scope: "LinkedIn networking and posts",
    desc: "Compiles requests into a macro over posts, connections, messages, and endorsements; bounces anything off-platform.",
    suggest: "Connect with Priya with a note, then endorse her for product management.",
    ops: [
      { name: "find_person", params: ["query"], ret: "person" },
      { name: "post_update", params: ["text"] },
      { name: "connect", params: ["user", "note"] },
      { name: "message", params: ["user", "text"] },
      { name: "endorse", params: ["person", "skill"] },
      { name: "find_post", params: ["query"], ret: "post" },
      { name: "comment", params: ["post", "text"] }
    ],
    fixed: [
      [
        "connect with Priya with a note then endorse her for product management",
        'connect(user="Priya", note="Great working with you \u2014 let us stay in touch!")\np = find_person(query="Priya")\nendorse(person=p, skill="product management")'
      ]
    ],
    templates: [
      { req: "post an update saying {text}", macro: 'post_update(text="{text}")' },
      { req: "connect with {user} and add a note {note}", macro: 'connect(user="{user}", note="{note}")' },
      { req: "message {user} {text}", macro: 'message(user="{user}", text="{text}")' },
      { req: "endorse {user} for {skill}", macro: 'p = find_person(query="{user}")\nendorse(person=p, skill="{skill}")' },
      { req: "comment {text} on the {topic} post", macro: 'p = find_post(query="{topic}")\ncomment(post=p, text="{text}")' }
    ],
    vocab: {
      text: ["excited to share I started a new role", "we are hiring engineers", "grateful for a great quarter", "thoughts on remote work"],
      user: ["Priya", "Alex", "a recruiter", "Dana", "my former manager"],
      note: ["Great working with you!", "Loved your talk", "Let us connect", "Fellow alum here"],
      skill: ["product management", "leadership", "TypeScript", "design", "data science"],
      topic: ["the hiring", "the milestone", "the article", "the announcement"]
    },
    oos: ["play music", "open a github issue", "navigate to the airport"]
  }
];
var SKILLS = DEFS.map((d) => buildSkill(d, 6));
var CALENDAR_SVG = '<svg viewBox="0 0 24 24" width="100%" height="100%" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><rect x="2.5" y="4" width="19" height="17" rx="3.4" fill="#ffffff"/><path d="M2.5 7.4a3.4 3.4 0 0 1 3.4-3.4h12.2a3.4 3.4 0 0 1 3.4 3.4v2.1H2.5z" fill="#ea4d3d"/><rect x="6" y="2.1" width="2.5" height="4.4" rx="1.25" fill="#b23528"/><rect x="15.5" y="2.1" width="2.5" height="4.4" rx="1.25" fill="#b23528"/><g fill="#cfd4dc"><rect x="5" y="11.6" width="3" height="2.7" rx=".7"/><rect x="10.5" y="11.6" width="3" height="2.7" rx=".7"/><rect x="16" y="11.6" width="3" height="2.7" rx=".7"/><rect x="5" y="15.7" width="3" height="2.7" rx=".7"/><rect x="16" y="15.7" width="3" height="2.7" rx=".7"/></g><rect x="10.5" y="15.7" width="3" height="2.7" rx=".7" fill="#2f72c4"/><rect x="2.5" y="4" width="19" height="17" rx="3.4" fill="none" stroke="#0000001f"/></svg>';
var POPULAR_2026 = [
  { key: "inbox-calendar", name: "Inbox & Calendar", skill: "inbox-calendar", cat: "productivity", logo: "google-calendar", bg: "linear-gradient(#fdfaf2,#efe7d4)", svg: CALENDAR_SVG, glyph: "\u2709", fs: 22 },
  { key: "music", name: "Music", skill: "music", cat: "media", logo: "spotify", bg: "#1db954", glyph: "\u266A", fs: 24 },
  { key: "github", name: "GitHub", skill: "github", cat: "developer", logo: "github", logoBg: "#f7f3e7", bg: "#181717", glyph: "GH", fs: 15 },
  { key: "youtube", name: "YouTube", skill: "youtube", cat: "media", logo: "youtube", bg: "#FF0000", glyph: "\u25B6", fs: 18 },
  { key: "instagram", name: "Instagram", skill: "instagram", cat: "social", logo: "instagram", bg: "linear-gradient(135deg,#feda75,#d62976 48%,#4f5bd5)", glyph: "\u{1F4F7}", fs: 20 },
  { key: "x", name: "X", skill: "x", cat: "social", logo: "twitter", bg: "#000000", glyph: "\u{1D54F}", fs: 23 },
  { key: "slack", name: "Slack", skill: "slack", cat: "work", logo: "slack", bg: "#4A154B", glyph: "S", fs: 24 },
  { key: "notion", name: "Notion", skill: "notion", cat: "productivity", logo: "notion", logoBg: "#f7f3e7", bg: "#0f0f0f", glyph: "N", fs: 24 },
  { key: "maps", name: "Maps", skill: "maps", cat: "navigation", logo: "google-maps", bg: "#34A853", glyph: "\u{1F4CD}", fs: 20 },
  { key: "amazon", name: "Amazon", skill: "amazon", cat: "shopping", bg: "#FF9900", fg: "#232F3E", glyph: "a", fs: 27 },
  { key: "reddit", name: "Reddit", skill: "reddit", cat: "social", logo: "reddit", bg: "#FF4500", glyph: "\u{1F47D}", fs: 20 },
  { key: "linkedin", name: "LinkedIn", skill: "linkedin", cat: "work", logo: "linkedin", bg: "#0A66C2", glyph: "in", fs: 17 },
  // ── the broader armory (coming soon) ──
  { key: "google", name: "Google", cat: "productivity", logo: "google", bg: "#4285F4", glyph: "G", fs: 25 },
  { key: "whatsapp", name: "WhatsApp", cat: "social", logo: "whatsapp", bg: "#25D366", glyph: "\u2706", fs: 22 },
  { key: "tiktok", name: "TikTok", cat: "social", logo: "tiktok", bg: "#010101", glyph: "\u266B", fs: 22 },
  { key: "facebook", name: "Facebook", cat: "social", logo: "facebook", bg: "#1877F2", glyph: "f", fs: 27 },
  { key: "snapchat", name: "Snapchat", cat: "social", bg: "#FFFC00", fg: "#111", glyph: "\u{1F47B}", fs: 22 },
  { key: "messenger", name: "Messenger", cat: "social", logo: "messenger", bg: "#0084FF", glyph: "\u2726", fs: 22 },
  { key: "discord", name: "Discord", cat: "social", logo: "discord", bg: "#5865F2", glyph: "D", fs: 24 },
  { key: "telegram", name: "Telegram", cat: "social", logo: "telegram", bg: "#229ED9", glyph: "\u2708", fs: 20 },
  { key: "netflix", name: "Netflix", cat: "media", logo: "netflix", bg: "#E50914", glyph: "NF", fs: 15 },
  { key: "twitch", name: "Twitch", cat: "media", logo: "twitch", bg: "#9146FF", glyph: "tw", fs: 16 },
  { key: "spotify", name: "Spotify", cat: "media", logo: "spotify", bg: "#1DB954", glyph: "\u25C9", fs: 20 },
  { key: "pinterest", name: "Pinterest", cat: "social", logo: "pinterest", bg: "#E60023", glyph: "P", fs: 24 },
  { key: "threads", name: "Threads", cat: "social", logo: "threads", logoBg: "#f7f3e7", bg: "#000000", glyph: "@", fs: 24 },
  { key: "uber", name: "Uber", cat: "travel", bg: "#000000", glyph: "U", fs: 24 },
  { key: "doordash", name: "DoorDash", cat: "food", bg: "#FF3008", glyph: "DD", fs: 14 },
  { key: "airbnb", name: "Airbnb", cat: "travel", logo: "airbnb", bg: "#FF5A5F", glyph: "A", fs: 24 },
  { key: "paypal", name: "PayPal", cat: "finance", logo: "paypal", bg: "#003087", glyph: "P", fs: 23 },
  { key: "venmo", name: "Venmo", cat: "finance", bg: "#3D95CE", glyph: "V", fs: 24 },
  { key: "chatgpt", name: "ChatGPT", cat: "ai", logo: "openai", bg: "#10A37F", glyph: "\u2738", fs: 20 },
  { key: "gemini", name: "Gemini", cat: "ai", logo: "google-gemini", bg: "#1C69FF", glyph: "\u2726", fs: 20 },
  { key: "perplexity", name: "Perplexity", cat: "ai", logo: "perplexity", bg: "#1FB8CD", glyph: "\u273A", fs: 20 },
  { key: "cursor", name: "Cursor", cat: "developer", bg: "#0b0b0b", glyph: "\u25AE", fs: 18 }
];

// src/icon_pipeline.js
var DEFAULT_BASE_PATHS = ["/vendor/logos", "./vendor/logos", "../vendor/logos"];
var ICON_THEME_PRESETS = {
  brand: { mode: "brand", label: "Brand", bg: null, fg: null },
  gold: { mode: "mono", label: "Gold monochrome", bg: "#2b220b", fg: "#ffd24a" },
  cyan: { mode: "mono", label: "Cyan monochrome", bg: "#082a2e", fg: "#61f2ff" },
  pixel: { mode: "pixel", label: "8-bit brand", bg: null, fg: null, pixelSize: 18 },
  pixelGold: { mode: "pixel-mono", label: "8-bit gold", bg: "#201806", fg: "#ffd24a", pixelSize: 18 },
  locked: { mode: "mono", label: "Locked", bg: "#d7d2c2", fg: "#7d7768" }
};
var LOGO_ALIASES = {
  "inbox-calendar": ["google-calendar", "google-gmail"],
  music: ["spotify"],
  github: ["github"],
  youtube: ["youtube"],
  instagram: ["instagram"],
  x: ["twitter"],
  slack: ["slack"],
  notion: ["notion"],
  maps: ["google-maps"],
  reddit: ["reddit"],
  linkedin: ["linkedin"],
  google: ["google"],
  whatsapp: ["whatsapp"],
  tiktok: ["tiktok"],
  facebook: ["facebook"],
  messenger: ["messenger"],
  discord: ["discord"],
  telegram: ["telegram"],
  netflix: ["netflix"],
  twitch: ["twitch"],
  spotify: ["spotify"],
  pinterest: ["pinterest"],
  threads: ["threads"],
  airbnb: ["airbnb"],
  paypal: ["paypal"],
  chatgpt: ["openai"],
  gemini: ["google-gemini"],
  perplexity: ["perplexity"]
};
var catalogPromise = null;
var activeTheme = safeStorageGet("eg_icon_theme") || "brand";
var paintVersions = /* @__PURE__ */ new WeakMap();
var rasterCache = /* @__PURE__ */ new Map();
function safeStorageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
__name(safeStorageGet, "safeStorageGet");
function safeStorageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
  }
}
__name(safeStorageSet, "safeStorageSet");
function slug(s) {
  return String(s || "").toLowerCase().replace(/&/g, " and ").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}
__name(slug, "slug");
function trimSlash(s) {
  return String(s || "").replace(/\/+$/, "");
}
__name(trimSlash, "trimSlash");
function cssUrl(src) {
  return `url("${String(src).replace(/"/g, '\\"')}")`;
}
__name(cssUrl, "cssUrl");
function svgDataUrl(svg) {
  return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(String(svg));
}
__name(svgDataUrl, "svgDataUrl");
function firstColor(bg) {
  if (!bg) return null;
  const m = String(bg).match(/#[0-9a-f]{3,8}/i);
  return m ? m[0] : String(bg).startsWith("#") ? bg : null;
}
__name(firstColor, "firstColor");
function iconTheme() {
  return activeTheme in ICON_THEME_PRESETS ? activeTheme : "brand";
}
__name(iconTheme, "iconTheme");
function iconThemePreset(theme = iconTheme()) {
  return ICON_THEME_PRESETS[theme] || ICON_THEME_PRESETS.brand;
}
__name(iconThemePreset, "iconThemePreset");
function setIconTheme(theme) {
  activeTheme = theme in ICON_THEME_PRESETS ? theme : "brand";
  safeStorageSet("eg_icon_theme", activeTheme);
  try {
    document.documentElement.dataset.iconTheme = activeTheme;
  } catch {
  }
  try {
    window.dispatchEvent(new CustomEvent("eg-icon-theme", { detail: { theme: activeTheme } }));
  } catch {
  }
  return activeTheme;
}
__name(setIconTheme, "setIconTheme");
function createLogoIndex(entries, basePath = "/vendor/logos") {
  const byShortname = /* @__PURE__ */ new Map();
  const byFileStem = /* @__PURE__ */ new Map();
  const byName = /* @__PURE__ */ new Map();
  for (const entry of entries || []) {
    const record = { ...entry, basePath: trimSlash(basePath) };
    byShortname.set(slug(entry.shortname), record);
    byName.set(slug(entry.name), record);
    for (const f of entry.files || []) byFileStem.set(slug(f.replace(/\.svg$/i, "")), record);
  }
  return { basePath: trimSlash(basePath), entries: entries || [], byShortname, byFileStem, byName };
}
__name(createLogoIndex, "createLogoIndex");
async function fetchCatalog(basePath) {
  const base = trimSlash(basePath);
  const resp = await fetch(`${base}/logos.json`, { cache: "force-cache" });
  if (!resp.ok) throw new Error(`logos catalog not found at ${base}`);
  const entries = await resp.json();
  return createLogoIndex(entries, base);
}
__name(fetchCatalog, "fetchCatalog");
async function loadLogoCatalog(basePaths = DEFAULT_BASE_PATHS) {
  if (!catalogPromise) {
    catalogPromise = (async () => {
      let lastErr = null;
      for (const base of basePaths) {
        try {
          return await fetchCatalog(base);
        } catch (e) {
          lastErr = e;
        }
      }
      throw lastErr || new Error("logo catalog unavailable");
    })();
  }
  return catalogPromise;
}
__name(loadLogoCatalog, "loadLogoCatalog");
function logoCandidates(tile = {}) {
  const keys = [];
  const add = /* @__PURE__ */ __name((v) => {
    if (!v) return;
    if (Array.isArray(v)) {
      for (const x of v) add(x);
      return;
    }
    const k = slug(v);
    if (k && !keys.includes(k)) keys.push(k);
  }, "add");
  add(tile.logo);
  add(LOGO_ALIASES[slug(tile.key)]);
  add(tile.key);
  add(tile.shortname);
  add(tile.name);
  add(tile.label);
  return keys;
}
__name(logoCandidates, "logoCandidates");
function chooseFile(entry, preferred) {
  const files = entry?.files || [];
  if (!files.length) return null;
  if (preferred) {
    const exact = files.find((f) => f === preferred || slug(f.replace(/\.svg$/i, "")) === slug(preferred));
    if (exact) return exact;
  }
  return files.find((f) => /-icon\.svg$/i.test(f) && !/monochrome/i.test(f)) || files.find((f) => !/monochrome/i.test(f)) || files[0];
}
__name(chooseFile, "chooseFile");
function resolveLogoFromIndex(index, tile = {}) {
  if (!index) return null;
  for (const c of logoCandidates(tile)) {
    const entry = index.byShortname.get(c) || index.byFileStem.get(c) || index.byName.get(c);
    if (!entry) continue;
    const file = chooseFile(entry, tile.logoFile);
    if (!file) continue;
    return {
      name: entry.name,
      shortname: entry.shortname,
      file,
      src: `${entry.basePath}/logos/${file}`
    };
  }
  return null;
}
__name(resolveLogoFromIndex, "resolveLogoFromIndex");
async function resolveLogo(tile) {
  if (tile?.svg) return { name: tile.name || tile.key, shortname: tile.key, file: "inline.svg", src: svgDataUrl(tile.svg), inline: tile.svg };
  const index = await loadLogoCatalog();
  return resolveLogoFromIndex(index, tile);
}
__name(resolveLogo, "resolveLogo");
function prepareTile(el, tile, preset, fallbackGlyph, fsScale, state2) {
  el.classList.remove("hasvg", "skill-icon--svg", "skill-icon--mask", "skill-icon--pixel", "skill-icon--chip", "skill-icon--locked");
  el.classList.add("skill-icon");
  el.classList.toggle("skill-icon--chip", state2 === "chip");
  el.classList.toggle("skill-icon--locked", state2 === "soon" || state2 === "locked");
  el.dataset.iconTheme = preset.mode;
  const tileBg = preset.bg || tile?.bg || "#6b6256";
  const tileFg = preset.fg || tile?.fg || "#fff";
  el.style.background = tileBg;
  el.style.color = tileFg;
  el.style.setProperty("--skill-icon-bg", tileBg);
  el.style.setProperty("--skill-icon-fg", tileFg);
  el.style.fontSize = Math.round((tile && tile.fs || 18) * fsScale) + "px";
  el.textContent = "";
  const fallback = document.createElement("span");
  fallback.className = "skill-icon__fallback";
  fallback.textContent = tile?.glyph || fallbackGlyph || "\u25C6";
  el.appendChild(fallback);
}
__name(prepareTile, "prepareTile");
function installBrand(el, logo, tile) {
  el.classList.add("hasvg", "skill-icon--svg");
  el.textContent = "";
  if (logo.inline) {
    el.innerHTML = logo.inline;
  } else {
    const img = document.createElement("img");
    img.className = "skill-icon__img";
    img.alt = "";
    img.decoding = "async";
    img.loading = "lazy";
    img.src = logo.src;
    el.appendChild(img);
  }
  el.style.background = tile?.logoBg || tile?.bg || "#fff";
}
__name(installBrand, "installBrand");
function installMask(el, logo, preset) {
  el.classList.add("hasvg", "skill-icon--mask");
  el.textContent = "";
  const mark = document.createElement("span");
  mark.className = "skill-icon__mask";
  mark.style.background = preset.fg || "#ffd24a";
  mark.style.webkitMask = `${cssUrl(logo.src)} center / contain no-repeat`;
  mark.style.mask = `${cssUrl(logo.src)} center / contain no-repeat`;
  el.appendChild(mark);
}
__name(installMask, "installMask");
async function rasterizeLogo(logo, preset) {
  const key = `${logo.src}|${preset.mode}|${preset.fg || ""}|${preset.pixelSize || 18}`;
  if (rasterCache.has(key)) return rasterCache.get(key);
  const p = new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => {
      const s = Math.max(8, Math.min(32, preset.pixelSize || 18));
      const c = document.createElement("canvas");
      c.width = s;
      c.height = s;
      const ctx = c.getContext("2d");
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, s, s);
      ctx.drawImage(img, 0, 0, s, s);
      if (preset.mode === "pixel-mono") {
        ctx.globalCompositeOperation = "source-in";
        ctx.fillStyle = preset.fg || "#ffd24a";
        ctx.fillRect(0, 0, s, s);
      }
      resolve(c.toDataURL("image/png"));
    };
    img.onerror = () => reject(new Error(`could not rasterize icon ${logo.src}`));
    img.src = logo.src;
  });
  rasterCache.set(key, p);
  return p;
}
__name(rasterizeLogo, "rasterizeLogo");
async function installPixel(el, logo, preset) {
  el.classList.add("hasvg", "skill-icon--pixel");
  el.textContent = "";
  const img = document.createElement("img");
  img.className = "skill-icon__img skill-icon__img--pixel";
  img.alt = "";
  img.decoding = "async";
  img.src = await rasterizeLogo(logo, preset);
  el.appendChild(img);
}
__name(installPixel, "installPixel");
function paintSkillIcon(el, tile = {}, options = {}) {
  if (!el) return;
  const preset = iconThemePreset(options.theme || iconTheme());
  const version = (paintVersions.get(el) || 0) + 1;
  paintVersions.set(el, version);
  prepareTile(el, tile, preset, options.fallbackGlyph, options.fsScale || 1, options.state);
  resolveLogo(tile).then(async (logo) => {
    if (!logo || paintVersions.get(el) !== version) return;
    if (preset.mode === "mono") installMask(el, logo, preset);
    else if (preset.mode === "pixel" || preset.mode === "pixel-mono") await installPixel(el, logo, preset);
    else installBrand(el, logo, tile);
  }).catch(() => {
  });
}
__name(paintSkillIcon, "paintSkillIcon");
function themedTileColor(tile, theme = iconTheme()) {
  const preset = iconThemePreset(theme);
  return preset.bg || firstColor(tile?.bg) || tile?.bg || "#6b6256";
}
__name(themedTileColor, "themedTileColor");

// src/main.js
var $ = /* @__PURE__ */ __name((id) => document.getElementById(id), "$");
var log = /* @__PURE__ */ __name((m) => {
  const s = $("railMsg");
  if (s) s.textContent = m;
  console.log("[emberglass]", m);
}, "log");
function steps(id) {
  const el = $(id), m = {};
  el.querySelectorAll(".step").forEach((s) => m[s.dataset.s] = s);
  const all = /* @__PURE__ */ __name(() => Object.values(m), "all");
  return {
    reset() {
      all().forEach((s) => s.classList.remove("active", "done", "loop"));
    },
    active(k) {
      m[k]?.classList.add("active");
    },
    activeOnly(k) {
      all().forEach((s) => s.classList.remove("active"));
      m[k]?.classList.add("active");
    },
    done(k) {
      m[k]?.classList.remove("active", "loop");
      m[k]?.classList.add("done");
    },
    loop(keys, on) {
      keys.forEach((k) => m[k]?.classList.toggle("loop", on));
    }
  };
}
__name(steps, "steps");
function startClock(id) {
  const el = $(id), t = el.querySelector(".t"), t0 = performance.now();
  let run = true;
  el.classList.add("on");
  (/* @__PURE__ */ __name((function f() {
    if (!run) return;
    t.textContent = ((performance.now() - t0) / 1e3).toFixed(1) + "s";
    requestAnimationFrame(f);
  }), "f"))();
  return () => {
    run = false;
    el.classList.remove("on");
  };
}
__name(startClock, "startClock");
var session = new ModelSession({ cfg: QWEN25_3B, log });
var adapters = new AdapterRegistry();
var state = {
  loaded: false,
  busy: false,
  err: null,
  tuned: null,
  // { name, kind:'guided'|'own', build(userText)->messages[], suggest }
  activeRunId: null,
  // history run currently applied
  dirHandle: null
  // File System Access workspace folder
};
var GEN = { maxTokens: 2048, temperature: 0.6, topP: 0.95, topK: 64 };
var skillByKey = /* @__PURE__ */ __name((key) => SKILLS.find((s) => key && (key === s.key || String(key).startsWith(s.key + " "))), "skillByKey");
var selectedSkillKey = SKILLS[0].key;
var trainLosses = [];
function sampleExamples(all, n) {
  const oos = all.filter(([, a]) => a === "OUT_OF_SCOPE");
  const inscope = all.filter(([, a]) => a !== "OUT_OF_SCOPE");
  const keep = Math.max(0, n - oos.length);
  const stride = Math.max(1, Math.floor(inscope.length / Math.max(1, keep)));
  const picked = [];
  for (let i = 0; i < inscope.length && picked.length < keep; i += stride) picked.push(inscope[i]);
  return [...picked, ...oos];
}
__name(sampleExamples, "sampleExamples");
function setBadge() {
  const rail = $("rail"), chip = $("railChip");
  if (!rail || !chip) return;
  if (state.err) {
    rail.dataset.state = "err";
    chip.textContent = "Load failed";
    return;
  }
  if (state.busy === "load") {
    rail.dataset.state = "busy";
    chip.textContent = "Loading\u2026";
    return;
  }
  if (!state.loaded) {
    rail.dataset.state = "idle";
    chip.textContent = "Model not loaded";
    return;
  }
  const sel = $("adapterSel")?.value || "none";
  if (sel === "none") {
    rail.dataset.state = "ok";
    chip.textContent = "Live \xB7 base";
  } else {
    rail.dataset.state = "tuned";
    chip.textContent = "Live \xB7 tuned: " + sel;
  }
}
__name(setBadge, "setBadge");
function lockInference(on) {
  $("inferLock").style.display = on ? "flex" : "none";
  $("run").disabled = on || !state.loaded || state.busy === "gen";
}
__name(lockInference, "lockInference");
function gateButtons() {
  const ready = state.loaded && !state.busy;
  $("run").disabled = !ready;
  $("trainGuided").disabled = !ready;
  $("trainOwn").disabled = !ready || !ownExamples().length;
  for (const id of ["load", "loadHF"]) $(id).disabled = !!state.busy;
  const ask = $("askSection");
  if (ask) ask.hidden = !state.loaded;
}
__name(gateButtons, "gateButtons");
async function loadWith(reader, label) {
  if (state.busy) return;
  state.busy = "load";
  state.err = null;
  setBadge();
  gateButtons();
  try {
    await session.loadWith(reader, label);
    state.loaded = true;
    log("Model ready. Train an account surface or equip a chain to execute writes.");
  } catch (e) {
    state.err = e.message;
    log("Load error: " + e.message);
    console.error(e);
  } finally {
    state.busy = false;
    setBadge();
    gateButtons();
  }
}
__name(loadWith, "loadWith");
function buildMessages(userText) {
  const sel = $("adapterSel")?.value || "none";
  if (sel !== "none" && state.tuned && state.tuned.name === sel) return state.tuned.build(userText);
  return [{ role: "user", content: userText }];
}
__name(buildMessages, "buildMessages");
async function runInference() {
  if (!state.loaded || state.busy) return;
  const userText = $("prompt").value.trim();
  if (!userText) {
    log("type something to ask first");
    return;
  }
  state.busy = "gen";
  gateButtons();
  const sel = $("adapterSel")?.value || "none";
  adapters.applyToRuntime(sel, session.rt);
  const out = $("out");
  out.textContent = "";
  const node = document.createTextNode("");
  out.appendChild(node);
  const st = steps("inferSteps");
  st.reset();
  const cap = $("inferCap");
  const stop = startClock("inferClock");
  $("inferProc").classList.add("on");
  setMacroCheck(null);
  st.active("tok");
  cap.textContent = "Tokenizing your prompt with the VibeThinker tokenizer\u2026";
  const t0 = performance.now();
  let n = 0, first = true, acc = "";
  try {
    const msgs = buildMessages(userText);
    st.done("tok");
    st.active("prefill");
    cap.textContent = "Reading the prompt into the KV cache (prefill)\u2026";
    for await (const d of session.generate(msgs, { maxTokens: GEN.maxTokens, temperature: GEN.temperature, topP: GEN.topP, topK: GEN.topK })) {
      if (first) {
        first = false;
        st.done("prefill");
        st.active("decode");
        cap.textContent = "Generating the answer one token at a time\u2026";
      }
      node.appendData(d);
      acc += d;
      n++;
      $("tokps").textContent = `${n} tok \xB7 ${(n / ((performance.now() - t0) / 1e3)).toFixed(1)} tok/s`;
      out.scrollTop = out.scrollHeight;
    }
    const dt = (performance.now() - t0) / 1e3;
    $("tokps").textContent = `${n} tok \xB7 ${(n / dt).toFixed(1)} tok/s \xB7 ${dt.toFixed(1)}s`;
    st.done("prefill");
    st.done("decode");
    st.done("done");
    cap.textContent = `Done \u2014 ${sel === "none" ? "base model" : 'tuned adapter "' + sel + '"'}.`;
    const skill = sel !== "none" && state.tuned && state.tuned.name === sel ? skillByKey(state.tuned.base) : null;
    if (skill) {
      const res = verifyMacro(acc, skill.spec);
      setMacroCheck(res, skill, acc);
      if (res.status === "ok") stageMsg(`Write resolved \u2014 compiled a ${res.n}-step plan on ${skill.label}.`);
      else if (res.status === "oos") stageMsg(`That request is outside the ${skill.label} surface. Try one of its writes.`);
      else stageMsg(`The plan didn't validate \u2014 adjust the request and try again.`);
      if (state.activeRunId) {
        bumpUses(state.activeRunId);
        renderDock();
      }
    }
    log(`done (${sel === "none" ? "base model" : "tuned adapter"}).`);
  } catch (e) {
    out.appendData("\n\n[error] " + e.message);
    cap.textContent = "error: " + e.message;
    console.error(e);
  } finally {
    stop();
    $("inferProc").classList.remove("on");
    state.busy = false;
    gateButtons();
  }
}
__name(runInference, "runInference");
async function runTraining({ examples, lr, epochs, accum, base, kind, system, build, suggest }) {
  if (!state.loaded) {
    log("Boot the engine first, then train a surface.");
    closeTrainer();
    return;
  }
  if (state.busy) return;
  const name = uniqueName(base);
  const runId = newId();
  state.busy = "train";
  lockInference(true);
  gateButtons();
  $("trainWidget").style.display = "";
  resetTrainTelemetry();
  const windows = Math.max(1, Math.ceil(examples.length / accum));
  const total = windows * epochs;
  let lastLoss = null;
  const ctrl = new TrainingController({
    session,
    adapters,
    log: /* @__PURE__ */ __name(() => {
    }, "log"),
    trainerOptions: { lr, maxTrainSeq: 384, lmHeadBlock: 128, maxGradNorm: 1, weightDecay: 0, warmupSteps: Math.min(4, total), totalSteps: total, gradAccumSteps: accum }
  });
  const st = steps("trainSteps");
  st.reset();
  const cap = $("trainCap");
  const stop = startClock("trainClock");
  st.active("prep");
  cap.textContent = "Building masked, shifted-label examples and tokenizing on the GPU\u2026";
  renderMaskPreview(ctrl, examples[0]);
  ctrl.initAdapter(name, { rank: 16, alpha: 32 });
  trainProgress(0, total, null, "warming up\u2026");
  const t0 = performance.now();
  try {
    st.done("prep");
    st.loop(["fwd", "bwd", "opt"], true);
    cap.textContent = "Looping forward \u2192 backward \u2192 AdamW over your examples (full-network backprop)\u2026";
    await ctrl.train(examples, {
      epochs,
      onStep: /* @__PURE__ */ __name((r) => {
        const { step, loss } = r;
        lastLoss = loss;
        updateTrainTelemetry(step, total, r);
        trainProgress(step, total, loss, `teaching \xB7 step ${step}/${total} \xB7 loss ${loss.toFixed(3)} \xB7 ${fmtNum(r.trainTokPerSec)} tok/s`);
        cap.textContent = `Step ${step}/${total} \u2014 forward ${fmtMs(r.microStepMs)} \u2192 backward \u2192 AdamW ${fmtMs(r.optimizerStepMs)} \xB7 loss ${loss.toFixed(3)}`;
      }, "onStep")
    });
    const dt = ((performance.now() - t0) / 1e3).toFixed(1);
    st.loop(["fwd", "bwd", "opt"], false);
    st.done("fwd");
    st.done("bwd");
    st.done("opt");
    st.active("swap");
    state.tuned = { name, kind, base, build, suggest, ctrl };
    state.activeRunId = runId;
    addAdapterOption(name);
    $("adapterSel").value = name;
    st.done("swap");
    trainProgress(total, total, null, `done in ${dt}s \u2014 adapter "${name}" is live`);
    cap.textContent = `Adapter "${name}" hot-swapped into inference \u2014 live. Trained in ${dt}s.`;
    $("downloadAdapter").style.display = "";
    showTryIt(suggest);
    try {
      const files = await exportLoraAdapter(ctrl.trainer, { name });
      await saveRun(
        {
          id: runId,
          name,
          base,
          kind,
          system: system || null,
          suggest: suggest || "",
          createdAt: Date.now(),
          steps: total,
          epochs,
          durationSec: +dt,
          finalLoss: lastLoss,
          rank: 16,
          alpha: 32
        },
        { safetensors: files.safetensors, configJson: files.configJson }
      );
      renderHistory();
    } catch (e) {
      console.warn("[history] save failed", e);
    }
    log(`Trained "${name}" in ${dt}s. Saved to your Atlas; equip it to try the write surface.`);
  } catch (e) {
    st.loop(["fwd", "bwd", "opt"], false);
    trainProgress(0, total, null, "training error: " + e.message);
    cap.textContent = "error: " + e.message;
    console.error(e);
  } finally {
    stop();
    state.busy = false;
    lockInference(false);
    gateButtons();
  }
}
__name(runTraining, "runTraining");
var MAX_CHARS = 12e3;
var MAX_CHUNKS = 24;
var MIN_WORDS = 12;
var HEAD_WORDS = 6;
function chunkText(text) {
  text = (text || "").replace(/\r/g, "").slice(0, MAX_CHARS);
  const paras = text.split(/\n{2,}|\.(?=\s)/).map((s) => s.trim()).filter(Boolean);
  const out = [];
  for (const p of paras) {
    const words = p.split(/\s+/).filter(Boolean);
    if (words.length < MIN_WORDS) continue;
    const head = words.slice(0, HEAD_WORDS).join(" ");
    const rest = words.slice(HEAD_WORDS).join(" ");
    out.push({ head, rest, full: p });
    if (out.length >= MAX_CHUNKS) break;
  }
  return out;
}
__name(chunkText, "chunkText");
var _ownChunks = [];
function ownExamples() {
  return _ownChunks.map((c) => ({ messages: [{ role: "user", content: c.head }], completion: " " + c.rest }));
}
__name(ownExamples, "ownExamples");
function refreshOwn() {
  const text = $("ownText").value;
  _ownChunks = chunkText(text);
  const chars = Math.min(MAX_CHARS, (text || "").length);
  $("ownStats").textContent = _ownChunks.length ? `${_ownChunks.length} snippet(s) \xB7 ${chars} chars (cap ${MAX_CHARS}) \xB7 ready to teach` : `paste/drop at least one paragraph (~${MIN_WORDS}+ words). 100% local.`;
  gateButtons();
}
__name(refreshOwn, "refreshOwn");
function openTrainer() {
  const t = $("trainer");
  if (!t) return;
  renderSkillPicker();
  selectSkill(selectedSkillKey);
  t.hidden = false;
  document.body.classList.add("modal-open");
  $("gear")?.classList.remove("on");
  $("settings") && ($("settings").hidden = true);
}
__name(openTrainer, "openTrainer");
function closeTrainer() {
  const t = $("trainer");
  if (t) t.hidden = true;
  document.body.classList.remove("modal-open");
}
__name(closeTrainer, "closeTrainer");
function switchTab(which) {
  which === "train" ? openTrainer() : closeTrainer();
}
__name(switchTab, "switchTab");
function addAdapterOption(name) {
  const sel = $("adapterSel");
  if (![...sel.options].some((o) => o.value === name)) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    sel.appendChild(o);
  }
  const wrap = $("adapterWrap");
  if (wrap) wrap.hidden = false;
}
__name(addAdapterOption, "addAdapterOption");
function trainProgress(step, total, loss, label) {
  $("trainBar").style.width = (100 * step / Math.max(1, total)).toFixed(1) + "%";
  $("trainLabel").textContent = label;
}
__name(trainProgress, "trainProgress");
function resetTrainTelemetry() {
  trainLosses = [];
  const box = $("trainMetrics");
  if (box) box.hidden = false;
  for (const [id, v] of [["tmLoss", "\u2014"], ["tmTokps", "\u2014"], ["tmActive", "\u2014"], ["tmOpt", "\u2014"]]) {
    const el = $(id);
    if (el) el.textContent = v;
  }
  const line = $("lossLine");
  if (line) line.setAttribute("points", "");
  const preview = $("maskPreview");
  if (preview) preview.hidden = true;
}
__name(resetTrainTelemetry, "resetTrainTelemetry");
function updateTrainTelemetry(step, total, r) {
  trainLosses.push(r.loss);
  $("tmLoss").textContent = r.loss.toFixed(4);
  $("tmTokps").textContent = `${fmtNum(r.trainTokPerSec)} tok/s`;
  $("tmActive").textContent = `${r.numActive || 0} / ${r.tokens || 0}`;
  $("tmOpt").textContent = fmtMs(r.optimizerStepMs);
  drawLossSpark();
}
__name(updateTrainTelemetry, "updateTrainTelemetry");
function drawLossSpark() {
  const line = $("lossLine");
  if (!line || trainLosses.length < 2) return;
  const min = Math.min(...trainLosses);
  const max = Math.max(...trainLosses);
  const span = Math.max(1e-6, max - min);
  const points = trainLosses.map((v, i) => {
    const x = i / Math.max(1, trainLosses.length - 1) * 300;
    const y = 36 - (v - min) / span * 32;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  line.setAttribute("points", points);
}
__name(drawLossSpark, "drawLossSpark");
function renderMaskPreview(ctrl, example) {
  const box = $("maskPreview");
  const rows = $("maskRows");
  if (!box || !rows || !example) return;
  try {
    const preview = ctrl.inspectExample(example);
    $("maskSummary").textContent = `${preview.tokens.length} tokens \xB7 ${preview.trainPositions} trained next-token labels`;
    const shown = preview.rows.slice(0, 96);
    rows.innerHTML = '<div class="hdr">pos</div><div class="hdr">segment</div><div class="hdr">token</div><div class="hdr target">trained target</div>' + shown.map((r) => {
      const cls = `${r.trainsNext ? "train" : ""} ${r.segment}`;
      const target = r.trainsNext ? `${r.targetId} ${clip(r.targetText, 24)}` : "";
      return `<div class="${cls}">${r.index}</div><div class="${cls}">${esc(r.segment)}</div><div class="${cls}">${r.id} ${esc(clip(r.text, 28))}</div><div class="${cls} target">${esc(target)}</div>`;
    }).join("") + (preview.rows.length > shown.length ? `<div class="prompt">\u2026</div><div class="prompt">truncated</div><div class="prompt">${preview.rows.length - shown.length} more rows</div><div class="prompt target"></div>` : "");
    box.hidden = false;
  } catch (e) {
    rows.innerHTML = `<div class="prompt">preview</div><div class="prompt">error</div><div class="prompt">${esc(e.message)}</div><div class="prompt target"></div>`;
    box.hidden = false;
  }
}
__name(renderMaskPreview, "renderMaskPreview");
function showTryIt(suggest) {
  const t = $("tryIt");
  t.style.display = "flex";
  $("tryItBtn").onclick = () => {
    switchTab("infer");
    $("adapterSel").value = state.tuned.name;
    setBadge();
    $("prompt").value = suggest;
    runInference();
  };
  renderEquipPanel();
  if (state.tuned?.name) stageMsg(`New surface trained: \u201C${state.tuned.name}\u201D \u2014 it was added to your Atlas. Equip it into a chain to act.`);
}
__name(showTryIt, "showTryIt");
function renderEquipPanel() {
  const bar = $("equipBar");
  if (!bar) return;
  const skill = state.tuned ? skillByKey(state.tuned.base) : null;
  if (!skill || !skill.spec) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;
  const set = /* @__PURE__ */ __name((id, v) => {
    const e = $(id);
    if (e) e.textContent = v;
  }, "set");
  paintIcon($("equipIcon"), dockOf(skill.key), skill.icon, 0.85);
  set("equipName", `${skill.label} surface`);
  set("equipScope", `surface: ${skill.spec.scope}`);
  const ops = $("equipOps");
  if (ops) {
    ops.innerHTML = "";
    for (const op of skill.spec.ops) {
      const c = document.createElement("span");
      c.className = "equip__op";
      c.textContent = op.name;
      c.title = `${op.name}(${(op.params || []).join(", ")})`;
      ops.appendChild(c);
    }
  }
  const host = $("equipDrills");
  if (host) {
    host.innerHTML = "";
    const inscope = skill.examples.filter(([, a]) => a !== "OUT_OF_SCOPE");
    const step = Math.max(1, Math.floor(inscope.length / 4));
    const picks = [];
    for (let i = 0; i < inscope.length && picks.length < 4; i += step) picks.push(inscope[i][0]);
    for (const q of picks) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "drill";
      b.textContent = q;
      b.title = "Fire this drill";
      b.onclick = () => {
        $("prompt").value = q;
        runInference();
      };
      host.appendChild(b);
    }
  }
}
__name(renderEquipPanel, "renderEquipPanel");
function humanizePlan(text) {
  const out = [];
  for (const raw of String(text).split("\n")) {
    const line = raw.trim();
    if (!line || line === "OUT_OF_SCOPE") continue;
    const m = line.match(/^(?:[A-Za-z_]\w*\s*=\s*)?([A-Za-z_]\w*)\s*\((.*)\)\s*;?\s*$/);
    if (!m) continue;
    const op = m[1].replace(/_/g, " ");
    const args = [...m[2].matchAll(/([A-Za-z_]\w*)\s*=\s*"([^"]*)"/g)].map((x) => x[2]).filter(Boolean);
    const summary = args.slice(0, 2).join(" \xB7 ");
    out.push(summary ? `${op} \u2014 ${summary}` : op);
  }
  return out;
}
__name(humanizePlan, "humanizePlan");
function uniqueName(base) {
  const taken = new Set(listRuns().map((r) => r.name));
  if (!taken.has(base)) return base;
  let i = 2;
  while (taken.has(`${base} #${i}`)) i++;
  return `${base} #${i}`;
}
__name(uniqueName, "uniqueName");
function buildFromMeta(meta) {
  return meta.system ? (u) => [{ role: "system", content: meta.system }, { role: "user", content: u }] : (u) => [{ role: "user", content: u }];
}
__name(buildFromMeta, "buildFromMeta");
function fmtRunMeta(m) {
  const parts = [];
  if (m.finalLoss != null) parts.push("loss " + Number(m.finalLoss).toFixed(3));
  if (m.steps) parts.push(m.steps + " steps");
  if (m.durationSec != null) parts.push(Math.round(m.durationSec) + "s");
  try {
    parts.push(new Date(m.createdAt).toLocaleDateString(void 0, { month: "short", day: "numeric" }));
  } catch {
  }
  return parts.join(" \xB7 ");
}
__name(fmtRunMeta, "fmtRunMeta");
function renderHistory() {
  const runs = listRuns();
  $("historyCount").textContent = String(runs.length);
  $("historyEmpty").style.display = runs.length ? "none" : "";
  const ul = $("historyList");
  ul.innerHTML = "";
  for (const m of runs) {
    const { lv, xp } = skillLevel(m);
    const rar = rarityOf(lv);
    const active = m.id === state.activeRunId;
    const li = document.createElement("li");
    li.className = "item" + (active ? " active" : "");
    li.dataset.id = m.id;
    li.dataset.kind = m.kind || "own";
    li.dataset.rarity = rar.key;
    li.title = `${m.name} \u2014 click to equip`;
    li.innerHTML = `<div class="item__frame"><span class="item__icon"></span><span class="item__lv">L${lv}</span></div><div class="item__body"><div class="item__name">${esc(m.name)}</div><div class="item__rar">${rar.label} \xB7 ${esc(itemTypeLabel(m))}</div><div class="item__meta">${esc(fmtRunMeta(m))}</div><div class="item__xp"><i style="width:${xp}%"></i></div></div>` + (active ? `<div class="item__tag">EQUIPPED</div>` : "") + `<div class="item__acts"><button data-act="apply" class="tiny primary">${active ? "\u2713 Equipped" : "\u25B6 Equip"}</button><button data-act="export" class="tiny secondary" title="Export adapter">\u2B07</button><button data-act="del" class="tiny danger" title="Scrap">\u2715</button></div>`;
    paintIcon(li.querySelector(".item__icon"), runTile(m), runIcon(m), 0.76);
    li.querySelector("[data-act=apply]").onclick = (e) => {
      e.stopPropagation();
      applyRun(m.id);
    };
    li.querySelector("[data-act=export]").onclick = (e) => {
      e.stopPropagation();
      exportRun(m.id);
    };
    li.querySelector("[data-act=del]").onclick = (e) => {
      e.stopPropagation();
      delRun(m.id);
    };
    li.onclick = () => applyRun(m.id);
    ul.appendChild(li);
  }
  renderDock();
  renderStage();
}
__name(renderHistory, "renderHistory");
var SKILL_ICON = { guided: "\u2694", own: "\u{1F4DC}" };
var usesByRun = /* @__PURE__ */ new Map();
function bumpUses(id) {
  usesByRun.set(id, (usesByRun.get(id) || 0) + 1);
}
__name(bumpUses, "bumpUses");
function runIcon(m) {
  const sk = skillByKey(m.base);
  return sk ? sk.icon : SKILL_ICON[m.kind] || "\u{1F5E1}";
}
__name(runIcon, "runIcon");
function runTile(m) {
  const sk = skillByKey(m.base);
  return sk ? dockOf(sk.key) : { ...BYOD_TILE, name: m.name, glyph: SKILL_ICON[m.kind] || "\u{1F5E1}" };
}
__name(runTile, "runTile");
function skillLevel(m) {
  const lv = Math.max(1, Math.min(9, Math.round((m.steps || 12) / 12)));
  const loss = m.finalLoss == null ? 1.5 : Number(m.finalLoss);
  const xp = Math.max(6, Math.min(100, Math.round(100 * (3 - loss) / 3)));
  return { lv, xp };
}
__name(skillLevel, "skillLevel");
function rarityOf(lv) {
  if (lv >= 9) return { key: "legendary", label: "Legendary" };
  if (lv >= 7) return { key: "epic", label: "Epic" };
  if (lv >= 5) return { key: "rare", label: "Rare" };
  if (lv >= 3) return { key: "uncommon", label: "Uncommon" };
  return { key: "common", label: "Common" };
}
__name(rarityOf, "rarityOf");
function itemTypeLabel(m) {
  const sk = skillByKey(m.base);
  if (sk) return sk.label;
  return m.kind === "guided" ? "Surface" : "Custom surface";
}
__name(itemTypeLabel, "itemTypeLabel");
var BYOD_TILE = { bg: "#6b6256", fg: "#fff", glyph: "\u{1F4DC}", fs: 20 };
var SERVICES = POPULAR_2026;
var dockRuns = [];
var justEquippedId = null;
var IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || "");
var SWAP_KEY = IS_MAC ? "\u2318K" : "Ctrl+K";
function renderDock() {
  const tray = $("dockSlots");
  if (!tray) return;
  const runs = listRuns();
  tray.innerHTML = "";
  dockRuns = [];
  const seen = /* @__PURE__ */ new Set();
  const addTile = /* @__PURE__ */ __name((svc, opts) => {
    const el = document.createElement("div");
    el.className = "dock__tile";
    el.tabIndex = 0;
    el.setAttribute("role", "button");
    el.dataset.state = opts.state;
    el.dataset.key = svc.key;
    if (opts.runid) el.dataset.runid = opts.runid;
    if (opts.pop) el.classList.add("dock__tile--pop");
    const g = document.createElement("span");
    g.className = "dock__glyph";
    paintIcon(g, svc, svc.glyph, 1, { state: opts.state });
    el.appendChild(g);
    if (opts.lv != null) {
      const b = document.createElement("span");
      b.className = "dock__lv";
      b.textContent = "L" + opts.lv;
      el.appendChild(b);
    }
    if (opts.keyN != null) {
      const k = document.createElement("span");
      k.className = "dock__key";
      k.textContent = opts.keyN;
      el.appendChild(k);
    }
    if (opts.forge) {
      const f = document.createElement("span");
      f.className = "dock__forge";
      f.textContent = "+";
      el.appendChild(f);
    }
    if (opts.lock) {
      const l = document.createElement("span");
      l.className = "dock__lock";
      l.textContent = "\u{1F512}";
      el.appendChild(l);
    }
    const t = document.createElement("span");
    t.className = "dock__tip";
    if (opts.tipHtml) {
      t.classList.add("dock__tip--rich");
      t.innerHTML = opts.tipHtml;
    } else t.textContent = opts.tip;
    el.appendChild(t);
    el.setAttribute("aria-label", opts.tip);
    el.onclick = opts.onClick;
    el.onmouseenter = () => sfx.hover();
    el.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        opts.onClick();
      }
    };
    tray.appendChild(el);
  }, "addTile");
  for (const svc of SERVICES) {
    if (svc.skill) {
      const run = runs.find((r) => skillByKey(r.base)?.key === svc.skill);
      if (run) {
        seen.add(run.id);
        const { lv } = skillLevel(run);
        const equipped = run.id === state.activeRunId;
        dockRuns.push(run.id);
        const keyN = dockRuns.length <= 9 ? dockRuns.length : null;
        const uses = usesByRun.get(run.id) || 0;
        const sk = skillByKey(svc.skill);
        addTile(svc, {
          state: equipped ? "equipped" : "owned",
          runid: run.id,
          lv,
          keyN,
          pop: equipped && justEquippedId === run.id,
          tip: `${svc.name} \xB7 Lv ${lv}${equipped ? " \xB7 equipped" : ""}${uses ? " \xB7 " + uses + "\xD7" : ""}${keyN ? " \xB7 [" + keyN + "]" : ""}`,
          tipHtml: dockTip(svc.name, { lv, rarity: rarityOf(lv), scope: sk?.spec?.scope, opsN: sk?.spec?.ops?.length, uses, keyN, equipped }),
          // the equipped "lead" slot opens the radial quick-swap wheel (BotW-style)
          onClick: /* @__PURE__ */ __name(() => equipped ? openWheel(false) : applyRun(run.id), "onClick")
        });
      } else {
        addTile(svc, {
          state: "forge",
          forge: true,
          tip: `${svc.name} \u2014 train this surface`,
          onClick: /* @__PURE__ */ __name(() => {
            selectSkill(svc.skill);
            openTrainer();
          }, "onClick")
        });
      }
    } else {
      addTile(svc, {
        state: "soon",
        lock: true,
        tip: `${svc.name} \u2014 planned surface`,
        onClick: /* @__PURE__ */ __name(() => stageMsg(`\u201C${svc.name}\u201D is not trainable yet \u2014 the Atlas grows as we add account surfaces.`), "onClick")
      });
    }
  }
  const extra = runs.filter((r) => !seen.has(r.id));
  if (extra.length) {
    const sep = document.createElement("div");
    sep.className = "dock__sep";
    tray.appendChild(sep);
  }
  for (const r of extra) {
    const { lv } = skillLevel(r);
    const equipped = r.id === state.activeRunId;
    dockRuns.push(r.id);
    const keyN = dockRuns.length <= 9 ? dockRuns.length : null;
    addTile({ key: "byod-" + r.id, name: r.name, ...BYOD_TILE }, {
      state: equipped ? "equipped" : "owned",
      runid: r.id,
      lv,
      keyN,
      pop: equipped && justEquippedId === r.id,
      tip: `${r.name} \xB7 Lv ${lv}${equipped ? " \xB7 equipped" : ""}${keyN ? " \xB7 [" + keyN + "]" : ""}`,
      tipHtml: dockTip(r.name, { lv, rarity: rarityOf(lv), scope: "your private notes", uses: usesByRun.get(r.id) || 0, keyN, equipped }),
      onClick: /* @__PURE__ */ __name(() => equipped ? openWheel(false) : applyRun(r.id), "onClick")
    });
  }
  justEquippedId = null;
}
__name(renderDock, "renderDock");
function dockTip(name, { lv, rarity, scope, opsN, uses, keyN, equipped } = {}) {
  const rows = [`<b class="dock__tipname">${esc(name)}</b>`];
  if (lv != null) rows.push(`<span class="dock__tiprar" data-rar="${rarity && rarity.key || "common"}">Lv ${lv} \xB7 ${esc(rarity && rarity.label || "")}</span>`);
  if (scope) rows.push(`<span class="dock__tipline">\u2694 ${esc(scope)}</span>`);
  const bits = [];
  if (opsN != null) bits.push(`${opsN} action${opsN === 1 ? "" : "s"}`);
  if (uses) bits.push(`used ${uses}\xD7`);
  if (bits.length) rows.push(`<span class="dock__tipline dim">${bits.join(" \xB7 ")}</span>`);
  rows.push(`<span class="dock__tipkey">${equipped ? `\u25C6 equipped \u2014 ${SWAP_KEY} or click to switch` : keyN ? `press [${keyN}] \xB7 ${SWAP_KEY} to switch` : "tap to equip"}</span>`);
  return rows.join("");
}
__name(dockTip, "dockTip");
var lastEquipIntent = null;
function equipByIndex(i) {
  if (i < 0 || i >= dockRuns.length) return;
  lastEquipIntent = dockRuns[i];
  applyRun(dockRuns[i]);
}
__name(equipByIndex, "equipByIndex");
var sfx = (() => {
  let ctx = null, muted = false;
  try {
    muted = localStorage.getItem("eg_mute") === "1";
  } catch {
  }
  const ac = /* @__PURE__ */ __name(() => {
    if (!ctx) {
      try {
        ctx = new (window.AudioContext || window.webkitAudioContext)();
      } catch {
      }
    }
    if (ctx && ctx.state === "suspended") ctx.resume();
    return ctx;
  }, "ac");
  const tone = /* @__PURE__ */ __name((freq, at, dur, type = "sine", gain = 0.05, slideTo = null) => {
    const c = ac();
    if (!c || muted) return;
    const t = c.currentTime + at, o = c.createOscillator(), g = c.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t);
    if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, t + dur);
    g.gain.setValueAtTime(1e-4, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.012);
    g.gain.exponentialRampToValueAtTime(1e-4, t + dur);
    o.connect(g).connect(c.destination);
    o.start(t);
    o.stop(t + dur + 0.03);
  }, "tone");
  return {
    get muted() {
      return muted;
    },
    toggle() {
      muted = !muted;
      try {
        localStorage.setItem("eg_mute", muted ? "1" : "0");
      } catch {
      }
      if (!muted) this.equip();
      return muted;
    },
    hover() {
      tone(1100, 0, 0.035, "triangle", 0.018);
    },
    open() {
      tone(360, 0, 0.14, "sawtooth", 0.035, 760);
    },
    move() {
      tone(720, 0, 0.03, "square", 0.02);
    },
    equip() {
      tone(523.25, 0, 0.08, "triangle", 0.05);
      tone(783.99, 0.06, 0.1, "triangle", 0.05);
      tone(1046.5, 0.13, 0.16, "sine", 0.045);
    },
    cancel() {
      tone(380, 0, 0.12, "sine", 0.035, 240);
    },
    error() {
      tone(170, 0, 0.18, "square", 0.045);
    }
  };
})();
var wheelOn = false;
var wheelHeld = false;
var wheelSel = 0;
var wheelNodes = [];
function ownedRunsInDockOrder() {
  return dockRuns.map((id) => getRun(id)).filter(Boolean);
}
__name(ownedRunsInDockOrder, "ownedRunsInDockOrder");
function openWheel(held) {
  if (wheelOn) {
    if (!held) closeWheel(true);
    return;
  }
  const el = $("wheel");
  if (!el) return;
  const runs = ownedRunsInDockOrder();
  if (!runs.length) {
    sfx.error();
    stageMsg("No surfaces to swap yet \u2014 train one first.");
    return;
  }
  wheelOn = true;
  wheelHeld = !!held;
  wheelNodes = [];
  const ring = $("wheelRing");
  ring.innerHTML = "";
  const N = runs.length, R = Math.min(168, 96 + N * 9);
  runs.forEach((r, i) => {
    const ang = -Math.PI / 2 + i * (2 * Math.PI / N);
    const x = Math.cos(ang) * R, y = Math.sin(ang) * R;
    const sk = skillByKey(r.base), d = dockOf(r.base) || { ...BYOD_TILE, name: r.name };
    const node = document.createElement("button");
    node.type = "button";
    node.className = "wheel__node";
    node.style.transform = `translate(-50%,-50%) translate(${x}px,${y}px)`;
    const ic = document.createElement("span");
    ic.className = "wheel__nicon";
    paintIcon(ic, d, sk?.icon, 1);
    const nm = document.createElement("span");
    nm.className = "wheel__nname";
    nm.textContent = r.name;
    const kb = document.createElement("span");
    kb.className = "wheel__nkey";
    kb.textContent = i < 9 ? i + 1 : "";
    node.append(ic, nm, kb);
    node.onmouseenter = () => setWheelSel(i, true);
    node.onclick = () => {
      setWheelSel(i);
      commitWheel();
    };
    ring.appendChild(node);
    wheelNodes.push({ el: node, run: r });
  });
  const cur = dockRuns.indexOf(state.activeRunId);
  setWheelSel(cur >= 0 ? cur : 0);
  el.hidden = false;
  el.setAttribute("aria-hidden", "false");
  document.body.classList.add("wheel-open");
  sfx.open();
}
__name(openWheel, "openWheel");
function setWheelSel(i, quiet) {
  if (!wheelNodes.length) return;
  wheelSel = (i + wheelNodes.length) % wheelNodes.length;
  wheelNodes.forEach((n, j) => n.el.classList.toggle("on", j === wheelSel));
  const hub = $("wheelHub");
  if (hub) hub.textContent = wheelNodes[wheelSel].run.name;
  if (!quiet) sfx.move();
  else sfx.hover();
}
__name(setWheelSel, "setWheelSel");
function commitWheel() {
  const sel = wheelNodes[wheelSel];
  const id = sel && sel.run.id;
  closeWheel(false);
  if (id && id !== state.activeRunId) applyRun(id);
}
__name(commitWheel, "commitWheel");
function closeWheel(silent) {
  if (!wheelOn) return;
  wheelOn = false;
  wheelHeld = false;
  const el = $("wheel");
  if (el) {
    el.hidden = true;
    el.setAttribute("aria-hidden", "true");
  }
  document.body.classList.remove("wheel-open");
  if (!silent) sfx.cancel();
}
__name(closeWheel, "closeWheel");
function wheelPointerMove(e) {
  if (!wheelOn || wheelNodes.length < 2) return;
  const el = $("wheel");
  const r = el.getBoundingClientRect();
  const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  const dx = e.clientX - cx, dy = e.clientY - cy;
  if (Math.hypot(dx, dy) < 34) return;
  const ang = Math.atan2(dy, dx), N = wheelNodes.length, step = 2 * Math.PI / N;
  let best = 0, bd = Infinity;
  for (let i = 0; i < N; i++) {
    const a = -Math.PI / 2 + i * step;
    let diff = Math.abs((ang - a + Math.PI * 3) % (2 * Math.PI) - Math.PI);
    if (diff < bd) {
      bd = diff;
      best = i;
    }
  }
  if (best !== wheelSel) setWheelSel(best, true);
}
__name(wheelPointerMove, "wheelPointerMove");
function setMacroCheck(res, skill, text) {
  const el = $("macroCheck");
  if (!el) return;
  if (!res || res.status === "empty") {
    el.hidden = true;
    el.textContent = "";
    el.removeAttribute("data-state");
    return;
  }
  el.hidden = false;
  if (res.status === "ok") {
    el.dataset.state = "ok";
    const ops = res.calls.map((c) => c.op).join(", ");
    const plan = text ? humanizePlan(text) : [];
    const planHtml = plan.length ? `<ol class="macrochk__plan">${plan.map((p) => `<li>${esc(p)}</li>`).join("")}</ol>` : "";
    el.innerHTML = `<b>\u2713 valid write plan</b> \xB7 ${res.n} call${res.n === 1 ? "" : "s"} on the ${esc(skill.label)} surface \xB7 <code>${esc(ops)}</code>${planHtml}`;
  } else if (res.status === "oos") {
    el.dataset.state = "oos";
    el.innerHTML = `<b>\u26D4 OUT_OF_SCOPE</b> \xB7 the ${esc(skill.label)} surface correctly refused \u2014 that request is outside its writes`;
  } else {
    el.dataset.state = "bad";
    el.innerHTML = `<b>\u2717 invalid macro</b> \xB7 ${esc(res.issues.slice(0, 2).join("; "))}`;
  }
}
__name(setMacroCheck, "setMacroCheck");
var RANKS = [[12, "Grandmaster"], [9, "Master"], [6, "Artisan"], [4, "Adept"], [2, "Journeyman"], [1, "Apprentice"], [0, "Initiate"]];
function paintIcon(el, d, fallbackGlyph, fsScale = 1, opts = {}) {
  if (!el) return;
  paintSkillIcon(el, d || {}, { fallbackGlyph, fsScale, state: opts.state });
}
__name(paintIcon, "paintIcon");
function stageMsg(text) {
  const e = $("stageMsg");
  if (e) e.textContent = "\xBB " + text;
}
__name(stageMsg, "stageMsg");
function renderStage() {
  const stage = $("stage");
  if (!stage) return;
  const runs = listRuns();
  const acquired = new Set(runs.map((r) => skillByKey(r.base)?.key).filter(Boolean));
  let maxLv = 0, steps2 = 0;
  for (const r of runs) {
    maxLv = Math.max(maxLv, skillLevel(r).lv);
    steps2 += r.steps || 0;
  }
  const lvl = 1 + Math.floor(steps2 / 120);
  const xpPct = Math.round(steps2 % 120 / 120 * 100);
  const rank = (RANKS.find(([t]) => runs.length >= t) || [0, "Initiate"])[1];
  const active = runs.find((r) => r.id === state.activeRunId);
  const skill = active ? skillByKey(active.base) : null;
  const d = skill ? dockOf(skill.key) : null;
  const set = /* @__PURE__ */ __name((id, v) => {
    const e = $(id);
    if (e) e.textContent = v;
  }, "set");
  set("stageScore", `${acquired.size} / ${SKILLS.length}`);
  set("stageLv", String(lvl));
  set("stageRank", rank);
  const xp = $("stageXp");
  if (xp) xp.style.width = xpPct + "%";
  const scene = $("stageScene");
  const icon = $("stageSignIcon");
  if (active) {
    set("stageSignName", active.name);
    paintIcon(icon, d, skill?.icon, 0.8);
    if (scene) scene.style.setProperty("--scene", themedTileColor(d, iconTheme()));
    stage.dataset.where = "in";
  } else {
    set("stageSignName", "Account Atlas");
    if (icon) {
      icon.classList.remove("hasvg");
      icon.textContent = "\u{1F310}";
      icon.style.background = "#13393f";
      icon.style.color = "#cdeeea";
      icon.style.fontSize = "17px";
    }
    if (scene) scene.style.setProperty("--scene", "#1d6f6a");
    stage.dataset.where = "out";
  }
}
__name(renderStage, "renderStage");
var dockOf = /* @__PURE__ */ __name((key) => POPULAR_2026.find((s) => s.key === key) || {}, "dockOf");
function renderSkillPicker() {
  const host = $("skillPicker");
  if (!host) return;
  const runs = listRuns();
  host.innerHTML = "";
  for (const sk of SKILLS) {
    const d = dockOf(sk.key);
    const run = runs.find((r) => skillByKey(r.base)?.key === sk.key);
    const lv = run ? skillLevel(run).lv : 0;
    const b = document.createElement("button");
    b.type = "button";
    b.className = "skillpick__btn" + (sk.key === selectedSkillKey ? " on" : "") + (lv ? " forged" : "");
    b.dataset.key = sk.key;
    const icon = document.createElement("span");
    icon.className = "skillpick__icon";
    paintIcon(icon, d, sk.icon, 0.78);
    const txt = document.createElement("span");
    txt.className = "skillpick__txt";
    txt.innerHTML = `<b>${esc(sk.label)}</b><i>${sk.spec.ops.length} writes \xB7 ${sk.examples.length} drills</i>`;
    b.append(icon, txt);
    if (lv) {
      const badge = document.createElement("span");
      badge.className = "skillpick__lv";
      badge.textContent = "L" + lv;
      b.appendChild(badge);
    }
    b.onclick = () => selectSkill(sk.key);
    host.appendChild(b);
  }
}
__name(renderSkillPicker, "renderSkillPicker");
function renderPairList(host, pairs, { limit = 4, compact = false } = {}) {
  if (!host) return;
  const shown = pairs.slice(0, limit);
  const more = Math.max(0, pairs.length - shown.length);
  host.innerHTML = shown.map(([q, a]) => {
    const macro = compact && a !== "OUT_OF_SCOPE" ? clip(a, 120) : a;
    return `<li><span class="skill-req">${esc(q)}</span><pre class="skill-macro">${esc(macro)}</pre></li>`;
  }).join("") + (more > 0 ? `<li class="skill-more">+ ${more} more ${compact ? "hidden" : "spec-valid"} drill${more === 1 ? "" : "s"}</li>` : "");
}
__name(renderPairList, "renderPairList");
function renderSurfacePlan(sk) {
  const d = dockOf(sk.key);
  paintIcon($("surfacePlanIcon"), d, sk.icon, 0.86);
  const guards = [...sk.examples || [], ...sk.eval || []].filter(([, a]) => a === "OUT_OF_SCOPE");
  const chips = [
    `${sk.spec.ops.length} writes`,
    `${sk.examples.length} train drills`,
    `${(sk.eval || []).length} held-out evals`,
    `${guards.length} refusal guards`,
    "rank 16 LoRA"
  ];
  const chipHost = $("surfacePlanChips");
  if (chipHost) chipHost.innerHTML = chips.map((c) => `<span class="surfacechip">${esc(c)}</span>`).join("");
  const contract = $("writeContract");
  if (contract) {
    contract.innerHTML = sk.spec.ops.map((op) => {
      const sig = `${op.name}(${(op.params || []).join(", ")})${op.ret ? " -> " + op.ret : ""}`;
      const params = (op.params || []).length ? (op.params || []).join(", ") : "no args";
      return `<div class="contractop"><code>${esc(sig)}</code><span>${esc(params)}</span></div>`;
    }).join("");
  }
  const rules = [];
  if (sk.context) rules.push(["Date anchor", sk.context]);
  rules.push(["Scope", `Only ${sk.spec.scope}; anything else must emit exactly OUT_OF_SCOPE.`]);
  for (const a of sk.contract?.assertions || []) rules.push([a.id, a.describe]);
  for (const f of sk.contract?.forbidden || []) rules.push([f.id, f.describe]);
  const ruleHost = $("surfaceRules");
  if (ruleHost) ruleHost.innerHTML = rules.map(([k, v]) => `<div class="ruleitem"><b>${esc(k)}</b>${esc(v)}</div>`).join("");
  const inscope = (sk.examples || []).filter(([, a]) => a !== "OUT_OF_SCOPE");
  renderPairList($("guidedList"), inscope, { limit: 5 });
  renderPairList($("evalList"), sk.eval || [], { limit: 4, compact: true });
  renderPairList($("guardList"), guards, { limit: 4, compact: true });
  const set = /* @__PURE__ */ __name((id, v) => {
    const e = $(id);
    if (e) e.textContent = v;
  }, "set");
  set("guidedSummary", `${inscope.length} train`);
  set("evalSummary", `${(sk.eval || []).length} held out`);
  set("guardSummary", `${guards.length} OOS`);
}
__name(renderSurfacePlan, "renderSurfacePlan");
function selectSkill(key) {
  const sk = skillByKey(key) || SKILLS[0];
  selectedSkillKey = sk.key;
  document.querySelectorAll("#skillPicker .skillpick__btn").forEach((b) => b.classList.toggle("on", b.dataset.key === sk.key));
  const title = $("skillTitle");
  if (title) title.innerHTML = `${sk.icon} ${esc(sk.label)} surface`;
  const desc = $("skillDesc");
  if (desc) desc.textContent = sk.desc;
  renderSurfacePlan(sk);
}
__name(selectSkill, "selectSkill");
async function applyRun(id) {
  const meta = getRun(id);
  if (!meta) return;
  if (!state.loaded) {
    log("Boot the engine first, then equip a surface.");
    closeTrainer();
    return;
  }
  if (state.busy) return;
  state.busy = "apply";
  gateButtons();
  try {
    log(`Applying "${meta.name}"\u2026`);
    let adapter = adapters.get(meta.name);
    if (!adapter) {
      const files = await loadRunFiles(id);
      adapter = await loadLoraAdapterGPU(session.rt.dev, files, QWEN25_3B);
      adapter.name = meta.name;
      adapters.adapters[meta.name] = adapter;
    }
    addAdapterOption(meta.name);
    state.tuned = { name: meta.name, kind: meta.kind, base: meta.base, build: buildFromMeta(meta), suggest: meta.suggest };
    state.activeRunId = id;
    justEquippedId = id;
    $("adapterSel").value = meta.name;
    setMacroCheck(null);
    sfx.equip();
    setBadge();
    renderHistory();
    renderEquipPanel();
    switchTab("infer");
    if (meta.suggest) $("prompt").value = meta.suggest;
    stageMsg(`Equipped \u201C${meta.name}\u201D. Pick a drill or write request.`);
    log(`Now serving fine-tune "${meta.name}". Ask away.`);
  } catch (e) {
    log("Could not apply: " + e.message);
    console.error(e);
  } finally {
    state.busy = false;
    gateButtons();
  }
}
__name(applyRun, "applyRun");
async function exportRun(id) {
  const meta = getRun(id);
  if (!meta) return;
  try {
    const { safetensors, configJson } = await getRunBlobs(id);
    const stem = (meta.name || "adapter").replace(/[^\w.-]+/g, "_");
    if (state.dirHandle && await ensurePermission(state.dirHandle)) {
      await writeFileToDir(state.dirHandle, stem + ".safetensors", safetensors);
      await writeFileToDir(state.dirHandle, stem + ".adapter_config.json", configJson);
      log(`Saved "${meta.name}" to your connected folder.`);
    } else {
      triggerBlob(safetensors, stem + ".safetensors");
      triggerBlob(new Blob([configJson], { type: "application/json" }), stem + ".adapter_config.json");
      log(`Exported "${meta.name}".`);
    }
  } catch (e) {
    log("Export failed: " + e.message);
  }
}
__name(exportRun, "exportRun");
async function delRun(id) {
  await deleteRun(id);
  if (state.activeRunId === id) state.activeRunId = null;
  renderHistory();
}
__name(delRun, "delRun");
function triggerBlob(data, filename) {
  const blob = data instanceof Blob ? data : new Blob([data]);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1e3);
}
__name(triggerBlob, "triggerBlob");
function fmtMs(ms) {
  return Number.isFinite(ms) ? `${ms.toFixed(ms >= 100 ? 0 : 1)}ms` : "\u2014";
}
__name(fmtMs, "fmtMs");
function fmtNum(n) {
  return Number.isFinite(n) ? n >= 100 ? n.toFixed(0) : n.toFixed(1) : "\u2014";
}
__name(fmtNum, "fmtNum");
function clip(s, n) {
  s = String(s ?? "").replace(/\s+/g, " ");
  return s.length > n ? s.slice(0, Math.max(0, n - 1)) + "\u2026" : s;
}
__name(clip, "clip");
function applyLayout() {
  const mq = /* @__PURE__ */ __name((q) => {
    try {
      return window.matchMedia(q).matches;
    } catch {
      return false;
    }
  }, "mq");
  const fold = mq("(horizontal-viewport-segments: 2)") || mq("(spanning: single-fold-vertical)");
  const mobile = mq("(max-width: 700px)");
  document.body.dataset.layout = fold ? "foldable" : mobile ? "mobile" : "desktop";
}
__name(applyLayout, "applyLayout");
function repaintIconSurfaces() {
  renderHistory();
  renderSkillPicker();
  renderEquipPanel();
  renderStage();
}
__name(repaintIconSurfaces, "repaintIconSurfaces");
function initIconTheme() {
  const sel = $("iconTheme");
  if (!sel) return;
  sel.innerHTML = Object.entries(ICON_THEME_PRESETS).filter(([k]) => k !== "locked").map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join("");
  sel.value = iconTheme();
  document.documentElement.dataset.iconTheme = iconTheme();
  sel.onchange = () => {
    setIconTheme(sel.value);
    repaintIconSurfaces();
  };
}
__name(initIconTheme, "initIconTheme");
async function initFs() {
  if (!fsSupported) {
    $("fsBlock").hidden = true;
    return;
  }
  $("fsBlock").hidden = false;
  const setDir = /* @__PURE__ */ __name((h) => {
    state.dirHandle = h;
    $("fsForget").hidden = false;
    $("ownImportDir").hidden = false;
    $("fsStatus").textContent = `connected: ${h.name || "folder"} \u2014 adapters can save here; import text below.`;
  }, "setDir");
  try {
    const saved = await savedDirectory();
    if (saved) setDir(saved);
  } catch {
  }
  $("fsConnect").onclick = async () => {
    try {
      setDir(await connectDirectory());
    } catch (e) {
      if (e.name !== "AbortError") log("folder: " + e.message);
    }
  };
  $("fsForget").onclick = async () => {
    await forgetDirectory();
    state.dirHandle = null;
    $("fsForget").hidden = true;
    $("ownImportDir").hidden = true;
    $("fsStatus").textContent = "not connected \u2014 import training text & save adapters straight to a folder you pick.";
  };
  $("ownImportDir").onclick = async () => {
    if (!state.dirHandle) return;
    if (!await ensurePermission(state.dirHandle, "read")) {
      log("permission denied for folder");
      return;
    }
    try {
      const { text, names } = await readDirText(state.dirHandle);
      if (!text.trim()) {
        $("ownStats").textContent = "no .txt/.md/.json/.csv files found in that folder";
        return;
      }
      $("ownText").value = (text + "\n" + $("ownText").value).slice(0, MAX_CHARS);
      refreshOwn();
      $("ownStats").textContent = `imported ${names.length} file(s) \xB7 ` + $("ownStats").textContent;
    } catch (e) {
      log("import failed: " + e.message);
    }
  };
}
__name(initFs, "initFs");
window.addEventListener("DOMContentLoaded", () => {
  renderSkillPicker();
  selectSkill(selectedSkillKey);
  $("learnBtn")?.addEventListener("click", () => openTrainer());
  $("learnCta")?.addEventListener("click", () => openTrainer());
  $("jobBoardBtn")?.addEventListener("click", () => stageMsg("Job Board will compare trained surfaces, evals, levels, and export status."));
  $("worldMapBtn")?.addEventListener("click", () => stageMsg("World Map will show account roots, segmented app surfaces, and workflow handoffs."));
  $("trainerClose")?.addEventListener("click", () => closeTrainer());
  $("trainer")?.addEventListener("click", (e) => {
    if (e.target.id === "trainer") closeTrainer();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeTrainer();
  });
  $("gear").onclick = () => {
    const open = $("settings").hidden;
    $("settings").hidden = !open;
    $("gear").classList.toggle("on", open);
  };
  $("adapterSel").onchange = setBadge;
  $("load").onclick = () => loadWith(urlReader($("modelUrl").value.trim()), $("modelUrl").value.trim());
  $("loadHF").onclick = () => {
    const repo = $("hfRepo").value.trim();
    const token = ($("hfToken")?.value || "").trim();
    if (!repo) return log("enter a Hugging Face repo id, e.g. WeiboAI/VibeThinker-3B");
    loadWith(hfReader(repo, token), "HF: " + repo);
  };
  $("modelFiles").onchange = (ev) => {
    const files = [...ev.target.files];
    if (!files.length) return;
    const map = {};
    for (const f of files) map[f.name] = f;
    loadWith(fileReader(map), `${files.length} local files`);
  };
  $("run").onclick = runInference;
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runInference();
  });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      wheelOn ? closeWheel(true) : openWheel(false);
      return;
    }
    if (wheelOn) {
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        setWheelSel(wheelSel + 1);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        setWheelSel(wheelSel - 1);
      } else if (e.key === "Enter") {
        e.preventDefault();
        commitWheel();
      } else if (e.key === "Escape") {
        e.preventDefault();
        closeWheel(false);
      } else if (e.key >= "1" && e.key <= "9") {
        e.preventDefault();
        setWheelSel(+e.key - 1);
        commitWheel();
      }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = e.target && e.target.tagName || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || e.target && e.target.isContentEditable) return;
    if (e.key >= "1" && e.key <= "9") equipByIndex(+e.key - 1);
  });
  const wheelEl = $("wheel");
  if (wheelEl) {
    wheelEl.addEventListener("pointermove", wheelPointerMove);
    wheelEl.addEventListener("pointerdown", (e) => {
      if (e.target === wheelEl || e.target.id === "wheelHub") closeWheel(false);
    });
  }
  const mute = $("mute");
  if (mute) {
    const paint = /* @__PURE__ */ __name(() => {
      mute.textContent = sfx.muted ? "\u{1F507}" : "\u{1F50A}";
      mute.classList.toggle("on", !sfx.muted);
      mute.setAttribute("aria-label", sfx.muted ? "Unmute sounds" : "Mute sounds");
    }, "paint");
    paint();
    mute.onclick = () => {
      sfx.toggle();
      paint();
    };
  }
  $("trainGuided").onclick = () => {
    const sk = skillByKey(selectedSkillKey) || SKILLS[0];
    const pool = sampleExamples(sk.examples, 32);
    const ex = pool.map(([q, a]) => ({ messages: [{ role: "system", content: sk.system }, { role: "user", content: q }], completion: " " + a }));
    const windows = Math.ceil(ex.length / 2);
    runTraining({
      examples: ex,
      lr: 3e-4,
      epochs: Math.max(6, Math.min(14, Math.round(280 / windows))),
      accum: 2,
      base: sk.key,
      kind: "guided",
      system: sk.system,
      build: /* @__PURE__ */ __name((u) => [{ role: "system", content: sk.system }, { role: "user", content: u }], "build"),
      suggest: sk.suggest
    });
  };
  $("ownText").addEventListener("input", refreshOwn);
  $("ownFiles").onchange = async (ev) => {
    const files = [...ev.target.files].slice(0, 5);
    let txt = "";
    for (const f of files) {
      try {
        txt += await f.text() + "\n\n";
      } catch {
      }
    }
    $("ownText").value = (txt + "\n" + $("ownText").value).slice(0, MAX_CHARS);
    refreshOwn();
  };
  $("ownFetch").onclick = async () => {
    const url = $("ownUrl").value.trim();
    if (!url) return;
    $("ownStats").textContent = "fetching readable text via reader proxy\u2026";
    try {
      const r = await fetch("https://r.jina.ai/" + url);
      if (!r.ok) throw new Error("HTTP " + r.status);
      const t = await r.text();
      $("ownText").value = t.slice(0, MAX_CHARS);
      refreshOwn();
    } catch (e) {
      $("ownStats").textContent = "could not fetch (CORS/blocked) \u2014 paste the text instead. " + e.message;
    }
  };
  $("trainOwn").onclick = () => {
    const ex = ownExamples();
    if (!ex.length) return;
    const windows = Math.ceil(ex.length / 2);
    runTraining({
      examples: ex,
      lr: 3e-4,
      accum: 2,
      epochs: Math.max(3, Math.min(8, Math.round(50 / windows))),
      base: "my-notes",
      kind: "own",
      system: null,
      build: /* @__PURE__ */ __name((u) => [{ role: "user", content: u }], "build"),
      suggest: _ownChunks[0]?.head || ""
    });
  };
  $("downloadAdapter").onclick = () => {
    if (state.tuned?.ctrl?.trainer) downloadLoraAdapter(state.tuned.ctrl.trainer, { name: state.tuned.name });
  };
  applyLayout();
  for (const q of ["(max-width: 700px)", "(horizontal-viewport-segments: 2)", "(spanning: single-fold-vertical)"]) {
    try {
      window.matchMedia(q).addEventListener("change", applyLayout);
    } catch {
    }
  }
  window.__layout = (m) => {
    document.body.dataset.layout = m;
  };
  window.__eg = {
    store: store_exports,
    renderHistory,
    renderDock,
    renderStage,
    stageMsg,
    renderEquipPanel,
    humanizePlan,
    applyRun,
    exportRun,
    delRun,
    state,
    // devtools/test surface
    openTrainer,
    closeTrainer,
    openWheel,
    closeWheel,
    commitWheel,
    setWheelSel,
    sfx,
    SKILLS,
    POPULAR_2026,
    selectSkill,
    renderSkillPicker,
    verifyMacro,
    setMacroCheck,
    equipByIndex,
    skillByKey,
    sampleExamples,
    setIconTheme: /* @__PURE__ */ __name((theme) => {
      const t = setIconTheme(theme);
      const sel = $("iconTheme");
      if (sel) sel.value = t;
      repaintIconSurfaces();
      return t;
    }, "setIconTheme"),
    get iconTheme() {
      return iconTheme();
    },
    get selectedSkillKey() {
      return selectedSkillKey;
    },
    get lastEquipIntent() {
      return lastEquipIntent;
    }
  };
  initFs();
  initIconTheme();
  renderHistory();
  switchTab("infer");
  setBadge();
  refreshOwn();
  gateButtons();
});
function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
}
__name(esc, "esc");
