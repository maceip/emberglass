# Saturday review.MD — Concise Active Task List

This is a minimal, prioritized extraction of the work items from `Saturday review.MD` (the control document).  
All work must obey the Recovery Contract (Tactic 1 + Tactic 2) and only use real inputs/evidence.

## Current Priority Order (from the document)

1. **Real Browser Benchmark Card** (active)
   - Real `/model` weights only.
   - Real browser run (WebGPU + subgroups).
   - Commit raw `benchmark-artifact.json` with: device, Chrome UA/version, model path, prefill, decode tok/s, training step, adapter state.
   - `test:validate-bench-artifact` enforces required fields.
   - Only after that: any README numbers.

2. **Real Provider Evaluation Card**
   - One dedicated real test account for one provider.
   - Full cycle: real auth → real read → planned write → confirmed write → read-after-write → cleanup.
   - Produce raw evidence artifact (no local stores).
   - Template: test/provider_eval_artifact_template.json (enforced shape).

3. **Real Skill Training Card**
   - One loop on declared source data.
   - Before + after eval on held-out set.
   - Real adapter artifact + inputs + scores + notes proving actual behavior change (no hiding failures).
   - Evidence template: test/skill_training_evidence_template.json.

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
Last updated: autonomous pass 2026-06-28

## Progress Notes (real-only changes)
- #1: Validator added (test/validate_benchmark_artifact.mjs). Artifact emission hardened.
- #2: Full evidence template + harness that refuses without real dedicated account.
- #3: Evidence template + report skeleton using declared real source data.
- #4: Current main surface explicitly documented as "engine harness only". Dock and history simplified/collapsed to reduce complexity (per symptoms + "remove surfaces" directive). Full three-screen reset remains deferred.
- Later Fixes: app-action-layer-design.md started with real code references. F-16 placeholder added.
- Contract enforcement: saturday_contract_check.mjs + updates.
