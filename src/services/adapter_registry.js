/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

export class AdapterRegistry {
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
}
