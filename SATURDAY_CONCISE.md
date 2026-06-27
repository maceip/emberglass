# Saturday review.MD — Concise Active Task List

This is a minimal, prioritized extraction of the work items from `Saturday review.MD` (the control document).  
All work must obey the Recovery Contract (Tactic 1 + Tactic 2) and only use real inputs/evidence.

## Current Priority Order (from the document)

1. **Real Browser Benchmark Card** (active)
   - Real `/model` weights only.
   - Real browser run (WebGPU + subgroups).
   - Commit raw `benchmark-artifact.json` with: device, Chrome UA/version, model path, prefill, decode tok/s, training step, adapter state.
   - Only after that: any README numbers.

2. **Real Provider Evaluation Card**
   - One dedicated real test account for one provider.
   - Full cycle: real auth → real read → planned write → confirmed write → read-after-write → cleanup.
   - Produce raw evidence artifact (no local stores).

3. **Real Skill Training Card**
   - One loop on declared source data.
   - Before + after eval on held-out set.
   - Real adapter artifact + inputs + scores + notes proving actual behavior change (no hiding failures).

4. **UI Reset Card** (explicitly deferred from current cleanup pass)
   - Only the three approved screens: Skillbook/Home, Skill/Train Surface, Job Board.
   - Desktop + foldable + mobile variants.
   - First-run must be understandable in <10s.

## Later Fixes (do only after relevant cards above; real only)
- Real benchmark reporting + committed artifacts
- F-16 verification = real-model parity check only
- Design app-action layer (auth, idempotency, ApprovalPacket, receipts, etc.) **before** any code
  - Started: see docs/app-action-layer-design.md
- Design extension/side-panel architecture **before** any code
- Provider tests on dedicated real accounts
- Training/eval provenance (distinguish generated vs real evidence)
- Separate full UI/UX reset to the three screens

## Non-Negotiable Rules (always)
- No browser extension code until architecture approved.
- No provider/DOM writes until action layer designed + approved.
- No public claims without real artifacts from real runs.
- No local substitutes for weights, accounts, data, or verification.
- Current main surface = engine harness only (not the product UI).
- Remove or hide complexity before adding surfaces.

Status is tracked in real code/tests, not in broad docs.
Last updated: autonomous pass 2026-06-27
