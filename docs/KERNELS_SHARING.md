# Kernels sharing: don't maintain the WebGPU runtime twice

## Context

The repo was split at commit `60ebf62`:

- **Core kernels harness** — [maceip/vibethinker-webgpu-lora](https://github.com/maceip/vibethinker-webgpu-lora). The searchable, "boring utility names" home for the WebGPU runtime + LoRA kernels (`src/qwgpu/*`), the inference engine, and the original in-browser guided-training demo (the fictional "Emberglass OS" Q&A dataset).
- **Emberglass (this repo)** — the accounts-as-skills app: per-account LoRA training, the skill/contract/planner stack (`src/skills/*`), the RPG UI, and the dry-run action layer (`src/skills/action/*`).

The kernels were never modified by the app work (every commit after `60ebf62` left `src/qwgpu/` untouched), so today **Emberglass vendors a copy** of `src/qwgpu/*` identical to the core repo.

## The problem

Two copies of the same WGSL kernels / runtime drift apart. A precision fix or new kernel landed in core has to be hand-ported here. We do not want that.

## Target: single source of truth in core

Emberglass should consume `src/qwgpu/*` from the core repo rather than vendoring it. Candidate mechanisms, roughly in order of preference:

1. **npm package** — core publishes `@maceip/qwgpu` (or a git-URL dependency); Emberglass imports kernels from it. Cleanest dependency boundary, versioned, but requires core to expose a stable package entry + build.
2. **git subtree** — `git subtree pull` `src/qwgpu/` from core into Emberglass on a cadence. No submodule friction for contributors; history is squashed/merged in.
3. **git submodule** — pin `src/qwgpu/` to a core commit. Exact provenance, but submodules add clone/CI friction.

## Constraints / acceptance

- The split boundary must hold: kernel changes land in **core first**, then flow into Emberglass — never the reverse.
- Whatever mechanism is chosen must keep `npm run build` and the gate suite green here with no source edits under `src/qwgpu/`.
- A CI check should fail if `src/qwgpu/*` here diverges from the pinned core source (drift guard), analogous to the existing `kernels:check`.

## Status

Not yet implemented. Tracked as a follow-up; for now the vendored copy is byte-identical to core at the split point.
