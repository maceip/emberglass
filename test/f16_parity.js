/*
 * Real-model F-16 vs F-32 parity check (Saturday review Later Fix).
 * Greedy-decodes ref prompt with f16 on/off; compares token ids to PyTorch ref.
 */
import { QwenWGPU } from '../src/qwgpu/runtime.js';
import { QWEN25_3B } from '../src/config.js';

const row = (data) => console.log('VWG_F16 ' + JSON.stringify(data));

async function requestDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  if (!adapter.features.has('subgroups')) throw new Error('GPU lacks subgroups');
  const dev = await adapter.requestDevice({
    requiredFeatures: ['subgroups', ...(adapter.features.has('timestamp-query') ? ['timestamp-query'] : [])],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    },
  });
  return { adapter, dev };
}

async function greedyIds(rt, ids, n = 16) {
  rt.prefillBatch(ids);
  let next = await rt.argmaxLogits();
  const got = [next];
  let pos = ids.length;
  for (let i = 0; i < n - 1; i++) {
    rt.token(next, pos);
    pos++;
    next = await rt.argmaxLogits();
    got.push(next);
  }
  return got;
}

window.run = async () => {
  const params = new URLSearchParams(location.search);
  const modelPath = params.get('model') || '/model';
  const ref = await (await fetch('./ref.json')).json();
  const { adapter, dev } = await requestDevice();
  row({
    type: 'device',
    userAgent: navigator.userAgent,
    maxBufferSize: adapter.limits.maxBufferSize,
  });

  const rt = new QwenWGPU(dev, QWEN25_3B, { decodeBatchSize: 1 });
  const t0 = performance.now();
  await rt.build(modelPath);
  row({ type: 'load', modelPath, seconds: (performance.now() - t0) / 1000 });

  const refGen = ref.gen_ids;
  const ids = ref.ids;

  rt.setUseF16(false);
  const f32Ids = await greedyIds(rt, ids, refGen.length);
  row({
    type: 'decode',
    precision: 'f32',
    usingF16: rt.usingF16(),
    tokens: f32Ids,
    matchLen: (() => { let i = 0; while (i < f32Ids.length && f32Ids[i] === refGen[i]) i++; return i; })(),
    refLen: refGen.length,
  });

  rt.setUseF16(true);
  const f16Ids = await greedyIds(rt, ids, refGen.length);
  row({
    type: 'decode',
    precision: 'f16',
    usingF16: rt.usingF16(),
    tokens: f16Ids,
    matchLen: (() => { let i = 0; while (i < f16Ids.length && f16Ids[i] === refGen[i]) i++; return i; })(),
    refLen: refGen.length,
  });

  let f16VsF32 = 0;
  for (let i = 0; i < Math.min(f16Ids.length, f32Ids.length); i++) {
    if (f16Ids[i] === f32Ids[i]) f16VsF32++;
  }
  row({
    type: 'parity',
    f16VsF32MatchLen: f16VsF32,
    f32VsRefMatchLen: f32Ids.filter((t, i) => t === refGen[i]).length === refGen.length
      ? refGen.length
      : (() => { let i = 0; while (i < f32Ids.length && f32Ids[i] === refGen[i]) i++; return i; })(),
    f16VsRefMatchLen: (() => { let i = 0; while (i < f16Ids.length && f16Ids[i] === refGen[i]) i++; return i; })(),
    refGen,
  });
  row({ type: 'done' });
};

window.addEventListener('DOMContentLoaded', () =>
  window.run().catch((e) => {
    row({ type: 'error', message: e.message, stack: (e.stack || '').slice(0, 500) });
  }),
);
