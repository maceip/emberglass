# Saturday review.MD — Later Fixes Status

From the "Later Fixes" section. These are to be tackled as **real capabilities**, never local substitutes.

- [x] Rebuild benchmark reporting from real browser runs against real model weights, with raw artifacts committed alongside any public number.
  - Evidence: `benchmark-artifact.json` (committed). Validator: `npm run test:validate-bench-artifact`. README numbers sourced from artifact.
- [x] Rebuild F-16 verification only as a real-model parity/performance check.
  - Evidence: `f16-parity-artifact.json` (16/16 greedy tokens match `ref.json` for f32 and f16 on real `/model`). Runner: `npm run test:f16-parity`.
- [x] Design the app-action layer before implementation, including provider auth, write idempotency, read-after-write verification, and cleanup.
  - Evidence: `docs/app-action-layer-design.md`. No write executors shipped (correct).
- [x] Design the Chrome extension / side-panel architecture before adding extension code.
  - Evidence: `docs/extension-architecture-design.md`. No `extension/` code (correct).
- [ ] Rebuild provider execution tests against dedicated real test accounts, not local stores.
  - Blocked: needs dedicated Google test account + OAuth env vars. Harness: `npm run test:real-provider-eval`. Blocked artifact: `provider-eval-artifact.json`.
- [ ] Rework training/eval provenance so user-facing claims distinguish generated teaching examples from real user/test-account eval evidence.
  - Partial: provenance headers in `pools.ts`, `state.js`; not yet reflected across all user-facing surfaces.
- [ ] Do a separate UI/UX reset against the approved skillbook, skill screen, and job-board direction.
  - Partial: wireframes + `docs/evidence/ui/` screenshots (`npm run evidence:ui`). Live `docs/index.html` remains engine harness only.

Last updated: 2026-06-28.

Work remaining items only with real evidence. See `SATURDAY_CONCISE.md` for Recovery Work Card status.
