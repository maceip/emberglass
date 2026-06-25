// Lightweight WebGPU capability probe. The showcase runs its visual pipelines
// regardless, but we surface real adapter info so the page is honest about whether
// this browser could actually run the WebGPU engine.
export async function probeGPU() {
  if (!('gpu' in navigator)) {
    return { ok: false, reason: 'navigator.gpu missing (no WebGPU)', adapter: null };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: 'no GPU adapter', adapter: null };
    const info = (await adapter.requestAdapterInfo?.()) || {};
    const feats = [];
    for (const f of adapter.features || []) feats.push(f);
    const wgsl = navigator.gpu.wgslLanguageFeatures;
    return {
      ok: true,
      reason: 'WebGPU available',
      adapter: {
        vendor: info.vendor || 'unknown',
        architecture: info.architecture || '',
        description: info.description || '',
        maxBufferSize: adapter.limits?.maxBufferSize ?? 0,
        f16: feats.includes('shader-f16'),
        subgroups: feats.includes('subgroups'),
        immediates: !!(wgsl && wgsl.has && wgsl.has('immediate_address_space')),
      },
    };
  } catch (e) {
    return { ok: false, reason: String(e?.message || e), adapter: null };
  }
}

export function fmtBytes(n) {
  if (!n) return '—';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 ? 2 : 0)}${u[i]}`;
}
