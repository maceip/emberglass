# Emberglass Agent Rules

## Absolute Ban: Fake Or Synthetic Work

Fake or synthetic work is 100% forbidden in this repository.

Do not add, generate, keep, or present any fake or synthetic:
- product behavior
- runtime code paths
- provider/account integrations
- DOM/page fixtures
- browser-extension behavior
- model weights or mock model sources
- training data, eval data, benchmark data, or verification data
- screenshots, seeded UI state, localStorage state, demo state, or visual proof
- documentation, claims, reports, or metrics

Do not create fake Google/Calendar/Gmail/provider stores, fake OAuth/session flows,
fake write targets, fake closed-loop execution, fake browser tabs, fake Calendar DOM,
fake benchmark fixtures, fake adapter fixtures, fake model paths, fake screenshots, or
fake verification output.

If real data, real model weights, a real provider/account, a real browser tab, or a
real benchmark environment is unavailable, stop and say exactly what is missing. Do
not substitute a fake local fixture. Do not soften fake work by calling it a smoke
test, demo, placeholder, harness, mock, fixture, synthetic run, generated result, or
prototype if it can confuse someone trying to use or evaluate the app.

Public or internal claims must come only from actually executed real paths. Never
claim screenshots, benchmarks, verification, training, provider execution, or app
behavior unless that exact path was run and inspected.
