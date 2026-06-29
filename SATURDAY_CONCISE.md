# Saturday review.MD — Concise Active Task List

This is a minimal, prioritized extraction of the work items from `Saturday review.MD` (the control document).  
All work must obey the Recovery Contract (Tactic 1 + Tactic 2) and only use real inputs/evidence.

## Current Priority Order (updated per user directive 2026-06-29)

- **One screen only: Skillbook / Home** (active direction)
  - Drop the three-screen product for now.
  - The single screen is "home, skillbook" — skills inventory + select + train action in place.
  - The "train thing" is folded into the skillbook: select a skill → Train is the primary verb for it.
  - No separate Skill/Train Surface screen at this time.
  - First-run must answer "what can I do now?" (Calendar skill front + center, obvious Train, equip, cast request → verified plan).
  - Keep real-only bar: real weights, real drills, no fakes. Real test account will be used when supplied.
  - Benchmarks deprioritized ("don't care about benchmarks" for now).

- **Real Provider Evaluation Card** (unblocked when test account arrives)
  - Use the offered real test account (user: "why don't you use mine?").
  - Full cycle against real provider surfaces.
  - Evidence only from real reads/writes on the dedicated account.

- **Real Skill Training Card**
  - Real training loop on declared drills from the skillbook.
  - Before/after on held-out, real adapter.

- **UI complexity reduction** (ongoing for the one screen)
  - Remove/hide engine internals, broad catalog noise, separate rails.
  - The skillbook is the whole experience for now.

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
Last updated: 2026-06-29 (one-screen skillbook pivot; benchmarks deprioritized; test account incoming)

## Progress Notes (real-only changes)
- #1: Validator added (test/validate_benchmark_artifact.mjs). Artifact emission hardened.
- #2: Full evidence template + harness that refuses without real dedicated account.
- #3: Evidence template + report skeleton using declared real source data.
- #4: Current main surface explicitly documented as "engine harness only". Dock collapsed, history hidden by default, secondary labels reduced (per symptoms + "remove or hide complexity" rule). Full three-screen reset remains deferred.
- Later Fixes: app-action-layer-design.md started with real code references. F-16 placeholder added.
- Contract enforcement: saturday_contract_check.mjs + updates.
- UI direction: one-screen skillbook (train folded in). Three-screen reset is now deferred in favor of this.
