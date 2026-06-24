# Training & LoRA

**Canonical documentation moved to [emberglass-tune](https://github.com/maceip/emberglass-tune).**

This repo (**emberglass**) is **inference only**. To train adapters:

```bash
cd ~/emberglass-tune
uv sync
uv run emberglass-tune --help
```

Quick answers:

| Question | Answer |
|---|---|
| How were custom weights made? | LoRA on `WeiboAI/VibeThinker-3B` via **emberglass-tune** (Anthropic traces → SFT) |
| Can I train in the browser? | **No** — use emberglass-tune (MLX or CUDA) |
| How do I run weights here? | Load `adapter_model.safetensors` + base into WebGPU (`src/lora_gpu.js`) |

See also [`REPO_ARCHITECTURE.md`](REPO_ARCHITECTURE.md).
