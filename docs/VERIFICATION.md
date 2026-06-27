# How Emberglass verifies a skill — and what it does *not* verify

This is the honest answer to "how do you actually know a trained skill works?"
There are four levels of verification. We do **1–3 today**. **Level 4 (executing
against a real Google Calendar and reading the change back) does not exist yet**,
because the execution/auth layer (extension + OAuth) is unbuilt — see
[`KERNELS_SHARING.md`](./KERNELS_SHARING.md) and the action-layer review. We write
that down here instead of pretending the number is something it isn't.

## The verification levels

| Level | Question it answers | Built? | Where |
|------|----------------------|--------|-------|
| **L1 — Emission** | Does the trained adapter *emit the right operation* for a natural-language prompt? | ✅ | `test/run_live_gpu.mjs` [5] |
| **L2 — Contract** | Is the emitted macro *well-formed and safe* (ISO times, non-zero duration, ordered slot windows, mapped provider method, idempotency key, resolved data-flow refs)? | ✅ | `test/_plan.mjs`, `contract.ts`, `_gold_calendar.mjs:contractHolds` |
| **L3 — Target match** | Does the intended action *match the requested target* (right duration, right weekday, right hour, mentions the right person)? | ✅ | `test/_gold_calendar.mjs` gold cases + `run_live_gpu.mjs` [5] |
| **L4 — Execution** | Was the action *actually executed on the real account* and can we read the created event back? | ❌ **unbuilt** | needs extension + OAuth (`chrome.identity`) + a real executor |

By design, the only executor today is `DryRunExecutor`: it performs **zero I/O**
(no `fetch`, no DOM, no provider SDK, no `chrome.*`) and every receipt is
`status: 'simulated'`. A ratchet invariant (`executors_are_dry_run`) trips if that
ever changes, forcing an explicit safety review. So L4 cannot silently appear.

## What "the calendar write target" is

The user's question — *"do you have a calendar target, a prompt, then verify the
action executed?"* — maps to L3, not L4. The target is checked-in, machine-readable,
and adapter-agnostic. Example (`test/_gold_calendar.mjs`):

```
prompt:  "Email the design team this week's notes, then put a 30-minute review
          on my calendar for Monday morning."
expectOps: ['create_event']
target assertions:  duration === 30 min · weekday === Monday · hour ∈ [6,12)
                    · also emits compose_email/schedule_send
```

A case **passes only if** the op is present **and** the contract holds **and** every
target assertion is true. `run_live_gpu.mjs` asks the prompt to the *real trained
adapter*, parses the emitted macro, and scores it. Result is written to
`/tmp/eg_ui_scratch/gold_results.json` with a per-check breakdown so the rate is honest.

What we do **not** do at L3: create the event on a real Google Calendar and re-read it.
That requires being logged in (the session precondition added in `state.js`/`ui.js`)
**and** a real executor with a token — neither exists yet. So "the action executed"
is verified as *"the intended action is correct and contract-clean"*, not *"the event
now exists in the account."* That distinction is the whole point of this document.

## Success rates (measured)

### L2/L3 plan gate — `npm run test:plan` (no model needed)
Compiles **every held-out calendar request × 3 providers (Google / Outlook / Zoho)**
through the planner + contract + dry-run executor:

```
plans: 45   steps: 69   receipts: 69   executors: dry-run        →  PLAN_PASS
```

**45 / 45 plans (100%)** compile contract-clean and dry-run into 69 simulated
receipts; the fail-closed guard (a zero-duration event produces 0 receipts) holds.
Note: this proves the *planner/contract* is correct **given the correct macro** — it
does not, by itself, measure whether the model emits that macro.

### L1/L3 model gate — `node test/run_live_gpu.mjs` (real WebGPU + 3B adapter)
Runs the 4 gold cases against the **actual trained LoRA adapter** in Chrome Canary.
This is the number that measures the *model*, base vs. tuned:

```
L1 (emits the op):        <pending>/4
L3 (op+contract+target):  <pending>/4   = <pending>% success rate
```

> Status: **not yet measured in this environment.** The run is real and self-verifying
> (WebGPU device initializes, tokenizer loads), but streaming + int4-quantizing the
> multi-GB VibeThinker-3B weights from HuggingFace is slow/unreliable on this machine,
> so a clean adapter-eval pass has not completed here. The canonical proof that the GPU
> train→persist→reload→re-hydrate→export path works end-to-end is the committed
> `test/e2e_roundtrip.mjs` (loads local `/model` weights). When `run_live_gpu.mjs`
> completes, paste the `gold_results.json` numbers into the block above.

## How to reproduce

```bash
npm run test:plan          # L2/L3 plan gate — fast, deterministic, no model
npm run test:skills        # skill registry + macro verifier (ok/oos/bad classes)
node test/run_live_gpu.mjs # L1/L3 against the real adapter (needs WebGPU + weights)
                           #   → writes /tmp/eg_ui_scratch/gold_results.json
node test/run_live_tour.mjs# product + login-gate UX tour (headed, client-only)
```

## The honest summary

- We verify that a skill **emits the correct, contract-clean, on-target action** (L1–L3).
- We do **not** verify that the action was **executed on a live account** (L4) — there is
  no extension, no OAuth token, and the executor does no I/O, on purpose.
- The deterministic plan gate is **45/45 (100%)**. The model-level gold success rate is
  produced by `run_live_gpu.mjs` and is currently **unmeasured here** due to the weight-
  stream constraint above — recorded as a gap rather than guessed.
