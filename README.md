<h1 align="center">🜂 EMBERGLASS</h1>
<p align="center"><em>Accounts as skills. Train a per-account adapter that turns your intent into verified, dry-run app-action plans — 100% in your browser tab.</em></p>

<p align="center">
<b>per-account WebGPU LoRA · constrained macro generation · contract-verified plans · dry-run only (nothing sent)</b>
</p>

<p align="center"><a href="https://maceip.github.io/emberglass/"><b>▶ Live demo</b></a> · <a href="https://github.com/maceip/vibethinker-webgpu-lora">Core WebGPU kernels</a></p>

---

## What Emberglass is

Emberglass treats every app you're logged into as a **skill** you can train. You fine-tune a small per-account LoRA adapter (in-browser, on a frozen int4 base), equip it like an RPG loadout, and from then on plain-language intent compiles into a **verified, provider-resolved action plan** for that surface (Inbox & Calendar is the flagship skill).

The model is a **planner/compiler**, never an executor: it emits a constrained macro, the macro is checked against a declarative contract, and the plan is run through a **dry-run executor only** — every receipt is `simulated`, nothing is ever sent. Provider/DOM writes remain intentionally blocked by the `executors_are_dry_run` ratchet until a separate approval-gated executor review changes that boundary.

```
intent → model → macro → verifyMacro → checkContract → compilePlan → DryRunExecutor → simulated receipts
```

## Relationship to the core kernels repo

The in-browser WebGPU runtime and LoRA kernels (`src/qwgpu/*`) originate from and are shared with the core engine repo, **[maceip/vibethinker-webgpu-lora](https://github.com/maceip/vibethinker-webgpu-lora)** (the boring, searchable home for the kernel work). Emberglass currently vendors a copy; the plan is to consume those kernels from the core repo so they aren't maintained in two places — see [docs/KERNELS_SHARING.md](docs/KERNELS_SHARING.md).

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

Throughput is hardware-dependent. No public performance table is checked in right
now. Publish numbers only from a clean browser run against real model weights
under `/model`.

## Performance features added

- Hot kernels use `var<immediate>` + `setImmediates` for per-dispatch metadata; the benchmark completed without WebGPU validation errors.
- `shader-f16` paths are active for RMS normalization, RoPE, attention partial/combine, elementwise add, and SiLU.
- GPU-resident sampling keeps top-k selection and sampling on GPU; measured `topK=40` sampling was 21.4 tok/s with one token ID read back.
- Workgroup autotuning uses `timestamp-query`; clean-run winners were `add=64`, `rms=256`, `silu=256`.
- Specialization constants (`override`) are used for workgroup sizes on key kernels and are reflected in dispatch sizing.
- High-level `generate()` can use the GPU sampler when requested.
- Benchmark harness reports prefill latency, greedy decode tok/s, GPU top-k sampling tok/s, decode sub-kernel timings, and a real backward/AdamW training step.
- Forward and backward WGSL now live in Jinja templates; `npm run kernels:check` verifies generated kernel modules are up to date.

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

<p align="center"><sub>Built the hard way, on purpose. 🜂</sub></p>
