# WGSL Shader Size & Compile Time

Context from 2026-06-24 research + code audit.

The JS bundle size (after good esbuild settings) is **not** the dominant first-paint / "READY" cost for this engine.  
The real cost is the creation of ~50 WGSL compute shaders + pipelines during `QwenWGPU.build()`.

## Current Situation (as of latest kernels + runtime)

- `src/qwgpu/kernels.js`: 2163 lines, ~95 KB of WGSL source (as JS template literals).
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
- No WGSL minification (comments, whitespace, blank lines all go to the browser's WGSL parser).
- Autotune can create even more temporary pipelines.

Result: long "build" phase that can make the page appear unresponsive (exactly what was seen in the f16 harness runs).

## High-Leverage Things We Can Do

### 1. Make pipeline creation async and parallel (biggest single win)
- Replace `createComputePipeline` with `createComputePipelineAsync` everywhere.
- Collect promises and `await Promise.all(...)` (or small batches) instead of creating serially.
- This moves the CPU cost of shader compilation off the main thread and lets the browser schedule it better.
- Measure: wall time from start of `build()` until "ready", plus a dedicated `shaderCompileMs`.

### 2. WGSL source minification / stripping at build time
- The 95 KB of WGSL text is shipped inside the JS bundle.
- A small processor can:
  - Strip `//` comments
  - Collapse repeated whitespace
  - Remove blank lines
  - (Optional) minify identifiers in a safe way later
- This shrinks the JS payload the browser has to parse and gives the WGSL compiler less source to chew on.
- Can be done as a pre-step that writes a `kernels.min.js` or via an esbuild plugin that processes the exported template literals.

### 3. Stop maintaining near-duplicate F16 sources
- Currently we have pairs like:
  - `RMSNORM` / `RMSNORM_F16`
  - `ROPE*` / `ROPE*_F16`
  - `ATTN_PARTIAL` / `ATTN_PARTIAL_F16`, etc.
- Most differences are just `enable f16;` + `f16` vs `f32` + a few casts.
- Instead:
  - Keep one canonical source per kernel.
  - At runtime (or build time) produce the f16 version by controlled string transformation + insertion of `enable f16;`.
- Benefit: roughly halves the WGSL source we maintain and ship for the f16 paths.

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

- Running `build:prod` (minify) helps a little because it minifies the JavaScript that contains the huge template literals.
- It does **not** minify inside the WGSL strings themselves.
- A dedicated WGSL strip step (run before or during the esbuild of kernels) is needed for the big wins.

## When to do this work

Per the overall rules on this project:

- Document the plan here and in `OPTIMIZATION_PLAN.md`.
- Instrument the cost (so we can see before/after on real hardware).
- Do not land large refactors of the kernel set or dispatch until we have a repeatable browser benchmark (the existing `bench:wgpu` + f16 harness + a "build time" measurement) that shows the impact of recent phases (immediates, f16 coverage, GPU sampling, etc.).

The same "run on real Canary + record numbers" gate that applies to the V8/JIT ideas should apply to major shader compile changes.

## Quick Wins That Are Relatively Safe to Prototype Early

- Switch the main build path to `createComputePipelineAsync` + parallel await (behind a flag or just do it).
- Add a trivial WGSL stripper that only removes `//` comments and collapses ws, wired into the esbuild config or a `npm run build:shaders`.
- Add timing + logging of shader creation count and time during `build()`.

These are high signal and can be measured immediately on the device the user is using for the harness runs.
