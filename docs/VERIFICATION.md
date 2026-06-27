# Verification — what "the test passed" actually means

> The honest answer to: *"Do you have a calendar write target you can track —
> train on Google Calendar, prompt, then verify the action was actually executed —
> and what is the success rate?"*
>
> **No, not at the real-provider level — by design.** Today the executor is
> dry-run only and never touches Google. So "the event was actually created on a
> real calendar" is **not yet verifiable**, and there is **no real-execution
> success rate** to report. This document writes that down, defines the levels of
> verification we *do* run, and records the numbers we *can* produce.

Last run: 2026-06-27. Re-run everything below to refresh.

---

## The verification ladder

We separate four things that are easy to conflate. A higher level does **not**
imply a lower one is the bottleneck — they fail for different reasons.

| Level | Question it answers | Touches a real account? | Status here |
|------|----------------------|--------------------------|-------------|
| **L0 — Grammar / contract** | Is the emitted macro spec-valid (allowed ops + args) or a clean `OUT_OF_SCOPE` bounce, and does it satisfy the skill contract (assertions + forbidden patterns)? | No | ✅ runs, enforced by tests |
| **L1 — Model accuracy** | Does the *fine-tuned* model emit the **correct** macro for a held-out request (vs. the golden target)? | No | ⚠️ harness exists; needs local weights (see below) |
| **L2 — Closed-loop execution** | If we *execute* the plan against a real (in-memory) calendar/inbox and **read the store back**, did the intended writes land with read→write dataflow fully resolved? | No (mock store) | ✅ **100% (17/17)** |
| **L3 — Real-provider execution** | Did the action **actually happen on real Google Calendar**, confirmed by reading it back through the API? | **Yes** | ❌ **not built** (dry-run boundary + no login broker) |

---

## L0 — Grammar / contract  (`npm run test:skills`)

`src/skills.js#verifyMacro` + `#checkContract` gate every macro against the
canonical port (`src/skills/inbox-calendar/port.ts`): only declared ops, only
that op's params, ISO times, `end = start + duration`, no forbidden patterns, or
a clean `OUT_OF_SCOPE`. This is a **pass/fail gate**, not a rate. It is the
precondition the dry-run executor enforces (`canExecute: plan.contractOk === true`).

## L2 — Closed-loop execution  (`npm run test:closedloop`)

This is the real "did the action happen?" check we can run **today, with no
account and no model**, and it is the headline number.

`test/verify_calendar_closed_loop.mjs`:
1. Takes the skill's **held-out golden eval split** (request → expected macro).
2. Compiles each macro to an `ActionPlan` (`compilePlan`, the real planner).
3. Executes it against a **genuinely separate in-memory executor that MUTATES a
   store** (calendar/inbox) — unlike the shipped `DryRunExecutor`, which mutates
   nothing. Reads (`find_slot`, `find_email`) return synthetic bindings, so any
   write that depends on a read (`find_slot → create_event`) **must resolve its
   refs to land**.
4. **Reads the store back** and asserts every intended write is present with its
   arguments fully resolved (no leftover `$refs`), and `OUT_OF_SCOPE` bounces
   write nothing.

```
eval cases   : 17 (held-out golden split)
passed       : 17
failed       : 0
SUCCESS RATE : 100.0%
by headline op: OUT_OF_SCOPE 2/2 · archive_email 1/1 · compose_email 1/1 ·
                create_event 3/3 · find_email 2/2 · forward_email 1/1 ·
                label_email 2/2 · reply_email 1/1 · rsvp 2/2 ·
                schedule_send 1/1 · set_reminder 1/1
```

**What this proves:** the macro → plan → execute → read-back pipeline is faithful,
including read→write dataflow wiring. **What it does NOT prove:** model accuracy
(it uses golden macros, not model output) or that anything happened on real Google.

## L1 — Model accuracy  (`npm run test:roundtrip` / `npm run live:gpu`)

`test/e2e_roundtrip.mjs` (real GPU, real model) loads VibeThinker-3B in WebGPU,
gets a BASE answer, runs in-browser LoRA training (real backward + AdamW),
persists + re-hydrates + exports a `.safetensors`, and asserts the tuned answer
emits the calendar/email macro ops and **differs from base**.

- ✅ It verifies the *real ML roundtrip* (load → train → persist → reload →
  re-hydrate → export) and that tuning **changes** behavior toward the macro grammar.
- ⚠️ It currently checks **macro-op presence (regex)**, not exact-match accuracy
  against the golden target. Upgrading it to compute a per-prompt accuracy over the
  held-out eval split (then feeding that output through the L2 store read-back) is
  the path to a true **model success rate** — tracked as TODO.
- 🚧 **Environment note (2026-06-27):** `e2e_roundtrip` expects **local** weights
  served at same-origin `/model`. The HF-streaming variant (`test/run_live_gpu.mjs`,
  loading `WeiboAI/VibeThinker-3B`) was run headed in Chrome Canary twice on this
  machine; WebGPU initialised fine (`maxBuffer=4.29GB`, f16, timestamp-query), the
  tokenizer loaded, but the renderer **crashed at ~4.6 min while int4-quantizing the
  streamed weights (0% progress shown)**. So L1 is currently **blocked on staging
  local weights** in this environment, not on the harness. To run it for real:
  download the model to `./model` and `BASE_URL=… npm run test:roundtrip`.

## L3 — Real-provider execution  (NOT BUILT — by design)

`src/skills/action/executor.ts` is the only executor and is a **hard dry-run
boundary**: no `fetch`, no DOM, no provider SDK, no `chrome.*`. Every receipt is
`status:'simulated'`. Adding a real executor must trip the ratchet invariant
`executors_are_dry_run` and force the action-layer review.

Therefore there is **no real Google Calendar write, no API read-back, and no
real-execution success rate.** Reporting one would be dishonest.

**To make L3 measurable** we need the two pieces the architecture review under-
specified, now scaffolded in the wireframes (`docs/wireframes/`):
1. **A signed-in surface (the login/session precondition)** — `SessionState`
   (logged_in/out/expired) + a sign-in gate, so capture/train/cast never run
   against a logged-out account. (Built in the wireframes; the mock Google
   account-chooser stands in for `chrome.identity`.)
2. **A real, approval-gated executor + a dedicated test calendar** — a throwaway
   Google Calendar as the write target. The closed-loop test then becomes:
   train → prompt the model → execute on the test calendar → **read the event
   back via the Calendar API** → assert title/start/end → repeat over N → report
   the rate. That is the number this document will carry once L3 exists.

---

## How to reproduce

```bash
npm run test:closedloop   # L2 closed-loop (execute + read back) — the headline rate
npm run test:skills       # L0 grammar/contract gate
npm run tour              # headed product+login workflow (client-only, dry-run)
npm run test:roundtrip    # L1 real GPU roundtrip — needs local ./model weights
```
