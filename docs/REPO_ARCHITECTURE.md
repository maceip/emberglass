# Architecture

Three sibling repos at `~/`:

| Path | GitHub | Job |
|---|---|---|
| `~/emberglass/` | [qwen-webgpu-lora](https://github.com/maceip/qwen-webgpu-lora) | WebGPU **inference** |
| `~/emberglass-tune/` | [emberglass-tune](https://github.com/maceip/emberglass-tune) | LoRA **training** |
| `~/vibebounty/` | [vibebounty](https://github.com/maceip/vibebounty) | Bug-bounty **demo** |

```mermaid
flowchart LR
  subgraph tune [emberglass-tune]
    D[Labels + traces] --> T[LoRA SFT]
    T --> A[adapter.safetensors]
  end
  subgraph infer [emberglass]
    B[Base int4] --> H[LoRA hot-swap]
    A --> H
    H --> W[Browser tokens]
  end
  subgraph product [vibebounty]
    A --> S[serve :8080]
    S --> U[HackerOne UI :8767]
  end
```

**Training details:** [emberglass-tune README](https://github.com/maceip/emberglass-tune) — MLX path, CUDA path, Anthropic teacher/judge trace pipeline.

**This repo (emberglass)** implements forward-pass WebGPU only. See root [README](../README.md).
