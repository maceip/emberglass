// Raw WebGPU int8 GEMV POC: y[n] = scale[n] * sum_k x[k]*W[n,k] (W int8, packed 4/u32).
// Times one token's projections (7 matmuls x36 layers) to see the int8 ceiling.
window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const WGSL = `
  @group(0) @binding(0) var<storage,read> x: array<f32>;
  @group(0) @binding(1) var<storage,read> w: array<u32>;   // [K][N/4] int8 packed (4 outputs per word, column-major coalesced)
  @group(0) @binding(2) var<storage,read> scale: array<f32>;
  @group(0) @binding(3) var<storage,read_write> y: array<f32>;
  @group(0) @binding(4) var<uniform> dims: vec2<u32>;
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x; let K = dims.x; let N = dims.y;
    if (n >= N) { return; }
    let N4 = N / 4u; let lane = n & 3u; let word = n >> 2u;
    var acc = 0.0;
    for (var k = 0u; k < K; k = k + 1u) {
      let p = w[k * N4 + word];
      let b = i32((p >> (lane * 8u)) & 0xFFu);
      let sb = f32((b ^ 0x80) - 0x80);   // sign-extend byte
      acc = acc + x[k] * sb;
    }
    y[n] = acc * scale[n];
  }`;
  const mod = dev.createShaderModule({ code: WGSL });
  const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
  function mk(K, N) {
    const x = dev.createBuffer({ size: K*4, usage: GPUBufferUsage.STORAGE });
    const w = dev.createBuffer({ size: K*N, usage: GPUBufferUsage.STORAGE }); // int8 => K*N bytes
    const sc = dev.createBuffer({ size: N*4, usage: GPUBufferUsage.STORAGE });
    const y = dev.createBuffer({ size: N*4, usage: GPUBufferUsage.STORAGE });
    const u = dev.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(u, 0, new Uint32Array([K, N]));
    const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
      {binding:0,resource:{buffer:x}},{binding:1,resource:{buffer:w}},{binding:2,resource:{buffer:sc}},{binding:3,resource:{buffer:y}},{binding:4,resource:{buffer:u}}] });
    return { bg, N, y };
  }
  const H=2048,I=11008;
  const layer = [mk(H,H),mk(H,256),mk(H,256),mk(H,H),mk(H,I),mk(H,I),mk(I,H)]; // q,k,v,o,gate,up,down
  function pass(enc){ for(const m of layer){ const p=enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0,m.bg); p.dispatchWorkgroups(Math.ceil(m.N/256)); p.end(); } }
  // warm
  for(let i=0;i<3;i++){ const e=dev.createCommandEncoder(); pass(e); dev.queue.submit([e.finish()]); } await dev.queue.onSubmittedWorkDone();
  const t0=performance.now();
  for(let L=0;L<36;L++){ const e=dev.createCommandEncoder(); pass(e); dev.queue.submit([e.finish()]); }
  await dev.queue.onSubmittedWorkDone();
  const dt=performance.now()-t0;
  console.log(`Q8 36 layers x7 int8 GEMV = ${dt.toFixed(1)}ms => matmul ceiling ${(1000/dt).toFixed(1)} tok/s`);
  console.log('Q8 DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('Q8 ERROR '+e.message)));
