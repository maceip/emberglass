# WGSL Shader Size & Compile Time

Context from 2026-06-24 research + code audit.

The JS bundle size (after good esbuild settings) is **not** the dominant first-paint / "READY" cost for this engine.  
The real cost is the creation of ~50 WGSL compute shaders + pipelines during `QwenWGPU.build()`.

## Current Situation (as of the template migration)

- `src/qwgpu/templates/forward/*.wgsl.jinja` and `src/qwgpu/templates/backward/*.wgsl.jinja` are now the source of truth.
- `src/qwgpu/kernels.js` and `src/qwgpu/backward_kernels.js` are generated files. Edit templates, then run `npm run kernels:generate`.
- `npm run kernels:check` verifies the generated modules are not stale, and `esbuild.config.mjs` regenerates before browser bundles.
- The browser bundle still ships rendered WGSL strings because WebGPU wants shader source at runtime.
- `runtime.js` calls `_pipe(...)` ~53 times in a normal `build()`:
  - Every call does synchronous `device.createShaderModule(...)` + `device.createComputePipeline(...)`.
  - Many kernels have parallel copies:
    - F32 + F16 variants (rms, rope*, attnP/C, add, silu, rmsT, ...)
    - Decode vs prefill variants
    - Paged vs non-paged
    - W4A8 (DP4a) variants
    - Several fused kernels
- `_pipe` is completely synchronous and blocks the main thread.
- Very little deduplication or caching of shader modules by content.
- No dedicated WGSL minification yet (comments, whitespace, blank lines all go to the browser's WGSL parser).
- Autotune can create even more temporary pipelines.

Result: long "build" phase that can make the page appear unresponsive (exactly what was seen in the f16 harness runs).

## High-Leverage Things We Can Do

### 1. Make pipeline creation async and parallel (biggest single win)
- Replace `createComputePipeline` with `createComputePipelineAsync` everywhere.
- Collect promises and `await Promise.all(...)` (or small batches) instead of creating serially.
- This moves the CPU cost of shader compilation off the main thread and lets the browser schedule it better.
- Measure: wall time from start of `build()` until "ready", plus a dedicated `shaderCompileMs`.

### 2. WGSL source minification / stripping at build time
- Rendered WGSL text is still shipped inside the JS bundle.
- A small processor can:
  - Strip `//` comments
  - Collapse repeated whitespace
  - Remove blank lines
  - (Optional) minify identifiers in a safe way later
- This shrinks the JS payload the browser has to parse and gives the WGSL compiler less source to chew on.
- Can be done in `scripts/generate_kernel_modules.mjs` before it writes the generated modules.

### 3. Consolidate near-duplicate F16 templates
- We still have pairs like:
  - `RMSNORM` / `RMSNORM_F16`
  - `ROPE*` / `ROPE*_F16`
  - `ATTN_PARTIAL` / `ATTN_PARTIAL_F16`, etc.
- Most differences are just `enable f16;` + `f16` vs `f32` + a few casts.
- Instead:
  - Keep one canonical source per kernel.
  - At generation time produce the f16 version from Jinja conditionals.
- Benefit: roughly halves the WGSL source we maintain for the f16 paths.

### 4. Content-based caching of ShaderModule + ComputePipeline
- Key by (normalized WGSL source + overrides object).
- The autotune path and any re-entrant builds currently risk recompiling identical code.
- Even in normal flow, some kernels are very similar.

### 5. Lazy / feature-gated creation
- Do not create pipelines for features that are off:
  - Paged attention pipelines only if `pagedAttention: true`
  - W4A8 variants only if using 4-bit path
  - Certain prefill modes only when selected
- Many of the 53 pipelines are created unconditionally today.

### 6. Use `override` + specialization constants more (already started)
- Good precedent with `WG` workgroup size.
- Can sometimes avoid whole separate shader variants if a boolean or small integer can be an `override`.

### 7. Measure shader compile cost explicitly
Add to the benchmark / harness output:

```js
const t0 = performance.now();
await rt.build(...);
row({ type: 'build', shaderCompileMs: performance.now() - t0, ... });
```

Also expose `rt.getPipelineStats()` or similar (number of modules created, which features pulled in which kernels).

## Interaction with esbuild / bundling

- Running `build:prod` (minify) helps a little because it minifies the generated JavaScript that contains the rendered WGSL strings.
- It does **not** minify inside the WGSL strings themselves.
- A dedicated WGSL strip step in the template generator is needed for the big wins.

## When to do this work

Per the overall rules on this project:

- Keep the template generator and `kernels:check` green before touching generated kernel modules.
- Instrument remaining compile cost (so we can see before/after on real hardware).
- Keep larger shader compile changes behind repeatable browser benchmarks (`bench:wgpu` + f16 harness + build timing).

The same "run on real Canary + record numbers" gate that applies to the V8/JIT ideas should apply to major shader compile changes.

## Quick Wins That Are Relatively Safe to Prototype Early

- Switch the main build path to `createComputePipelineAsync` + parallel await (behind a flag or just do it).
- Add a trivial WGSL stripper that only removes `//` comments and collapses whitespace inside `scripts/generate_kernel_modules.mjs`.
- Add timing + logging of shader creation count and time during `build()`.

These are high signal and can be measured immediately on the device the user is using for the harness runs.
