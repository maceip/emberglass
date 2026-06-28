# Saturday review.MD — Concise Active Task List

This is a minimal, prioritized extraction of the work items from `Saturday review.MD` (the control document).  
All work must obey the Recovery Contract (Tactic 1 + Tactic 2) and only use real inputs/evidence.

## Current Priority Order (from the document)

1. **Real Browser Benchmark Card** — **DONE** (2026-06-27)
   - Real `/model` weights only.
   - Real browser run (WebGPU + subgroups).
   - Committed `benchmark-artifact.json` with device, Chrome UA/version, model path, prefill, decode tok/s, training step.
   - `npm run test:validate-bench-artifact` passes.
   - README numbers sourced from artifact.

2. **Real Provider Evaluation Card** — **BLOCKED**
   - One dedicated real test account for one provider.
   - Full cycle: real auth → real read → planned write → confirmed write → read-after-write → cleanup.
   - Harness: `npm run test:real-provider-eval` (refuses without account).
   - Template: `test/provider_eval_artifact_template.json`.
   - Blocked artifact: `provider-eval-artifact.json` (`implementation_required`).

3. **Real Skill Training Card** — **IN PROGRESS** (runner ready, no passing artifact yet)
   - One loop on declared source data.
   - Before + after eval on held-out set.
   - Runner: `npm run evidence:skill-training` → `skill-training-artifact.json`.
   - Template: `test/skill_training_evidence_template.json`.
   - **Missing:** committed `skill-training-artifact.json` from a completed real run.

4. **UI Reset Card** — **PARTIAL** (wireframes + screenshots; live app not reset)
   - Three approved screens exist as wireframes: Skillbook/Home, Skill/Train Surface, Job Board.
   - Evidence: `docs/evidence/ui/` (18 screenshots, `npm run evidence:ui`).
   - Entry: `docs/product/` → wireframes.
   - **Not done:** replace/slim live `docs/index.html` (still engine harness). First-run 10s bar not validated on live app.

## Later Fixes

See `docs/saturday-later-fixes-status.md` for checkboxes. Summary:

| Fix | Status |
|-----|--------|
| Benchmark reporting + artifacts | Done |
| F-16 real-model parity | Done (`f16-parity-artifact.json`) |
| App-action layer design | Done (design only) |
| Extension architecture design | Done (design only) |
| Provider tests on real accounts | Blocked |
| Training/eval provenance | Partial |
| Full UI/UX reset | Partial |

## Non-Negotiable Rules (always)

- No browser extension code until architecture approved.
- No provider/DOM writes until action layer designed + approved.
- No public claims without real artifacts from real runs.
- No local substitutes for weights, accounts, data, or verification.
- Current main surface = engine harness only (not the product UI).
- Remove or hide complexity before adding surfaces.

Status is tracked in real code/tests, not in broad docs.  
Last updated: 2026-06-28.

## Progress Notes (real-only changes)

- **#1 DONE:** `benchmark-artifact.json` committed; Range server for multi-GB shards; validator + README cross-ref.
- **#2:** Template + harness; refuses without dedicated account; write path `implementation_required`.
- **#3:** Evidence runner + template; awaiting successful `skill-training-artifact.json`.
- **#4:** Wireframe screenshots + product entry; harness complexity reduced in `main.js`; full live reset deferred.
- **Later:** F-16 parity artifact committed; both design docs committed; contract check (`npm run test:saturday-contract`) passes.
- **Branch:** Recovery work merged to `i-dont-know-what-im-docommit` (PR #1). `main` not yet updated.
