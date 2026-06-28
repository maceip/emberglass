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

Throughput is hardware-dependent. Published numbers must come from a committed
`benchmark-artifact.json` produced by `npm run bench:wgpu` on real `/model` weights.
Validate with `npm run test:validate-bench-artifact` (see `SATURDAY_CONCISE.md`).

**Last captured run** (2026-06-27, Chrome Canary 151, MacIntel, WeiboAI/VibeThinker-3B):

| Metric | Value |
|--------|-------|
| Model load | 10.9 s |
| Prefill L=1024 | 1078 ms |
| Greedy decode @ ctx=1024 | **115.9 tok/s** |
| Greedy decode @ ctx=4096 | **100.3 tok/s** |
| Top-k sampling (k=40) | 21.5 tok/s |
| Train step (18 tok, rank 8) | 31.6 train-tok/s |

Full raw rows: see `benchmark-artifact.json` in the repo root.

## Performance features (implementation)

- Hot kernels use `var<immediate>` + `setImmediates` for per-dispatch metadata.
- `shader-f16` paths for RMS, RoPE, attention, add, and SiLU (parity: `npm run test:f16-parity`).
- GPU-resident top-k sampling; one token ID readback per sample step.
- Workgroup autotuning via `timestamp-query` when available.
- Benchmark harness reports prefill, greedy decode, sampling, profiling, and a real backward/AdamW step.

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
