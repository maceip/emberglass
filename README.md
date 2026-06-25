<h1 align="center">🜂 EMBERGLASS</h1>
<p align="center"><em>Optimized WebGPU inference for VibeThinker-3B — in your browser tab. No server, no upload.</em></p>

<p align="center">
<b>≥20 tok/s decode floor · live LoRA hot-swap · bit-exact reference checks · 100% client-side WebGPU</b>
</p>

<p align="center"><a href="https://maceip.github.io/qwen-webgpu-lora/"><b>▶ Live demo</b></a> · <a href="https://github.com/maceip/emberglass-tune">Training docs</a> · <a href="https://github.com/maceip/vibebounty">VibeBounty demo</a></p>

---

## Three repos

| Repo | Role | Train? | Run inference? |
|---|---|---|---|
| **[emberglass](https://github.com/maceip/qwen-webgpu-lora)** (this) | Custom **WebGPU** runtime — int4, fused kernels, LoRA hot-swap | **No** | **Yes** (browser) |
| **[emberglass-tune](https://github.com/maceip/emberglass-tune)** | LoRA **training** — MLX + CUDA, Anthropic trace pipeline | **Yes** | No |
| **[vibebounty](https://github.com/maceip/vibebounty)** | Bug-bounty **demo** — tuned adapter, HackerOne UI, CPU/GPU serve | Uses emberglass-tune | Yes (server) |

**How the weights are made:** labeled reports → Anthropic teacher traces → LoRA SFT → `adapter_model.safetensors`. Full pipeline: **[emberglass-tune README](https://github.com/maceip/emberglass-tune)**.

**How to run them here:** load base weights + optional adapter into WebGPU; forward pass only. No backward pass, no optimizer, no dataset code in this repo.

---

## What this repo is

Emberglass is an **inference-only** engine for Qwen2.5-class models (VibeThinker-3B):

- Custom **WGSL kernels** (GEMV/GEMM, attention, RoPE, sampling)
- **int4** layer weights on GPU, GPU-resident KV cache
- **Runtime LoRA hot-swap** — load PEFT/MLX `adapter_model.safetensors` without re-quantizing base (`src/lora_gpu.js`)
- Playwright correctness and throughput harnesses (`npm run test:*`)

| In emberglass | Elsewhere |
|---|---|
| WebGPU forward pass | Training → **emberglass-tune** |
| LoRA apply / swap / clear | Data + Anthropic traces → **emberglass-tune** + **vibebounty** |
| int4 load from `./model` or HF | HackerOne demo UI → **vibebounty** |
| | CPU/GPU serve for demos → **vibebounty** |

---

## Run it

```bash
cd ~/emberglass
npm install
npm run build
npm run serve    # http://localhost:8013
```

Open in Chrome/Edge with **WebGPU + `subgroups`**. Load base weights from `./model`, Hugging Face, or a directory picker. Optional LoRA adapter URL for hot-swap.

**Base model:** [WeiboAI/VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B)  
**Example adapter:** [macmacmacmac/vibebounty](https://huggingface.co/macmacmacmac/vibebounty) (train with emberglass-tune)

---

## Using a trained adapter

1. Train (or download) a PEFT adapter — see [emberglass-tune](https://github.com/maceip/emberglass-tune).
2. Serve adapter files same-origin (e.g. under `/adapters/my-run/`).
3. Load in the Emberglass UI or via VibeBounty's Emberglass bridge.

Tests: `npm run test:lora`, `npm run test:lora-path`.

---

## Verification

```bash
npm run test:correctness   # argmax / generation vs reference
npm run test:lora          # adapter parse, hot-swap, restore
npm run test:app           # full streaming UI path
npm run bench:wgpu         # structured throughput JSON
```

Requires port **8013**, WebGPU **`subgroups`**, and weights in `./model` (not bundled in repo).

---

## Performance

Throughput is hardware-dependent. Target: **≥20 tok/s** greedy decode on Intel Arc class; **~35 tok/s** on Apple M5 Max (Metal).

| Platform | Greedy decode (typical) |
|---|---:|
| Apple M5 Max + Metal | ~33–35 tok/s @ long ctx |
| Intel Arc 140V + D3D11 | ~22–24 tok/s @ short ctx |
| LoRA active (180 modules) | ~23 tok/s (M5 reference) |

Fused decode path: `fuseQKV` / `fuseRoPE` / `fuseMLP` / `fuseResidual`.

---

## Requirements

- Browser WebGPU with **`subgroups`** (no fallback kernel set)
- GPU memory for chosen context window
- Bring your own weights — repo does not ship model files

---

## Layout

```
src/qwgpu/           WGSL kernels, runtime, int4 quantize
src/lora_gpu.js      PEFT/MLX adapter → GPU buffers
src/services/        App session, generation, adapter registry
test/                Browser harnesses
docs/                GitHub Pages demo + architecture notes
model/               BYO base weights (gitignored)
```

---

## Related docs

- **Training (MLX, CUDA, Anthropic traces):** [emberglass-tune README](https://github.com/maceip/emberglass-tune)
- **Bug-bounty demo:** [vibebounty](https://github.com/maceip/vibebounty)
- **Architecture map:** [`docs/REPO_ARCHITECTURE.md`](docs/REPO_ARCHITECTURE.md)

---

## Performance notes (as of 2026-06-24 browser runs)

After the esbuild rework, bulk header cleanup, and continued addition of explicit optimization annotations:

- **Build speed (esbuild.config.mjs):** main + docs bundles in 11–13 ms on this machine.
- **Browser benchmark execution (Playwright + real Chromium context):** successfully loaded `test/bench_bundle.js`, ran to `"type":"done"`. Captured `profile-token`, `sampling-topk` (reported ~2.87 tok/s in severely GPU-limited headless env), and multiple VWG_BENCH records.
- **No breakage from recent formatting pass:** benchmark and f16 harness paths executed cleanly after header removal.
- **Ongoing wins from prior phases (still visible in code):**
  - Immediate push constants (`var<immediate>` + `setImmediates`) eliminating uniform bind churn on decode.
  - GPU-resident top-k + sample (chained in single CommandEncoder, only final u32 read back).
  - Bind group caching in GPUBufferPool.
  - Specialization constants for workgroup sizes.
  - Streaming Range + visit + release for model load (low peak JS memory).

Full model shader compile time remains the dominant cold-start cost (see `docs/WGSL_COMPILE_TIME.md`). Future async pipeline creation + WGSL stripping will be measured on real hardware before landing.

Numbers above captured during verification run of the benchmark inside the browser.

---

<p align="center"><sub>Built the hard way, on purpose. 🜂</sub></p>
