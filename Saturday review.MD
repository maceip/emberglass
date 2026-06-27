# Current State

## Recovery Contract

This document is now a control document, not an idea catalog. Its job is to stop the project
from cascading into more impressive-looking but non-working surfaces.

### Tactic 1: One-Page Operating Contract

Every near-term change should fit inside this contract:

- Preserve the approved product shape: skillbook home, skill training, job board.
- Do not add browser extension code until the extension architecture is approved.
- Do not add provider write actions until the app-action layer is designed.
- Do not add public performance claims without a real browser run against real model weights.
- Do not add local substitutes for provider state, model state, training data, screenshots, or
  verification data.
- If a capability is not real yet, leave it absent, disabled, or marked `implementation_required`
  in internal code only. Do not build a replacement demo that looks real.
- For UI work, remove or hide complexity before adding another surface.

This is the short version that should be checked before any new work starts.

### Tactic 2: Proof Before Surface Gate

Any future user-visible capability must pass this gate before it is documented or shown in the
main app:

- What real input does it consume?
- What real model, provider, browser page, or account does it touch?
- What real output or side effect proves it worked?
- Where is the raw evidence stored?
- What must remain hidden if that evidence does not exist?

If those questions cannot be answered, the feature can exist only as internal planning text.
It should not appear as a working button, benchmark, verification result, walkthrough, or
claim.

## Where We Are

Emberglass is paused at a cleanup seam.

The current worktree intentionally removes or deactivates non-real verification/demo
paths that could make the project look more capable than it is. The app is not in a
final release state, and the UI/UX pass is explicitly deferred.

## How We Got Here

The project evolved quickly across several threads of work:

- The core WebGPU / LoRA runtime work continued.
- The product direction shifted toward "accounts as skills" with an RPG-like skillbook.
- Architecture review introduced future browser-extension and app-action concepts.
- Prototype/demo code was added around extension capture, closed-loop calendar execution,
  F-16 verification, and UI screenshots.
- Some of that prototype code used local substitutes instead of real model/provider/browser
  state.
- Those substitutes created confusion because they could be mistaken for real product
  verification or user-testable capability.

The current cleanup removes those misleading paths rather than letting them remain as
apparently working features.

## Symptoms

- The UI currently feels over-composed and does not clearly reflect the approved three-screen
  product direction. This is not being fixed in the current cleanup pass.
- Extension work was started too early and is now deferred pending architecture approval.
- Provider execution is still dry-run only. The app does not currently perform real Calendar,
  Gmail, or DOM writes.
- Public performance numbers were removed because the checked-in table was not backed by a
  current real-model benchmark artifact.
- F-16 verification was removed because the available harness was not an acceptable basis for
  a public correctness or performance claim.
- Some docs and scripts were deleted or edited, so follow-up work needs to rebuild real
  equivalents instead of restoring local substitutes.

## Cleanup Already Done

- Removed the runtime mock-weight model load branch.
- Updated the benchmark path to load real `/model` weights.
- Removed fake/synthetic closed-loop calendar verification files.
- Removed synthetic UI screenshot seeding.
- Removed the synthetic F-16 harness/page.
- Removed extension files and package scripts that were created before architecture approval.
- Removed the verification doc that advertised non-real closed-loop results.
- Added root agent rules in `AGENT.md` and `CLAUDE.md` forbidding fake or synthetic work.
- Rebuilt `docs/bundle.js`.

## Later Fixes

These should be tackled later as real capabilities, not as local substitutes:

- Rebuild benchmark reporting from real browser runs against real model weights, with raw
  artifacts committed alongside any public number.
- Rebuild F-16 verification only as a real-model parity/performance check.
- Design the app-action layer before implementation, including provider auth, write
  idempotency, read-after-write verification, and cleanup.
- Design the Chrome extension / side-panel architecture before adding extension code.
- Rebuild provider execution tests against dedicated real test accounts, not local stores.
- Rework training/eval provenance so user-facing claims distinguish generated teaching
  examples from real user/test-account eval evidence.
- Do a separate UI/UX reset against the approved skillbook, skill screen, and job-board
  direction.

## Synthetic Data And Demo Failure Analysis

The synthetic-data problem and the unusable-demo problem have the same root cause: local
substitutes were used to keep momentum, then the surrounding docs and UI made those substitutes
look closer to product reality than they were.

That creates two different failures:

- Users or reviewers can think the app has real provider/model/browser capability when it does
  not.
- Engineers can build new layers on top of substitute behavior and accidentally optimize the
  wrong system.

The damage is not just reputational. It can corrupt product judgment. A demo that always has
clean local data teaches the team the wrong lesson about auth failures, missing account state,
DOM variance, model latency, browser permission prompts, user confusion, and real write
verification.

### Symptoms

- Benchmark and verification language drifted ahead of real evidence.
- Calendar and extension demos implied closed-loop capability without a real approved write
  architecture.
- Local provider substitutes were easy to confuse with provider integration.
- A synthetic UI screenshot path made it possible to validate staged state instead of actual
  first-run state.
- The F-16 path became a public-facing concern before the underlying harness was acceptable as
  a correctness or performance claim.
- The web demo became hard to interpret because it mixed real runtime state, future capability,
  and removed prototype assumptions.
- Documentation began describing what the system was intended to prove instead of what it had
  actually proven.

### How We Got Here

The project needed to move quickly through several unknowns at once:

- WebGPU runtime and LoRA adapter behavior.
- In-browser training product shape.
- Account-as-skill game metaphor.
- Provider integration.
- Browser extension form factor.
- Future app-action/write layer.
- Public performance posture.

Local substitutes were introduced as scaffolding during that exploration. The failure was not
that scaffolding existed during thinking; the failure was letting it cross into product code,
scripts, docs, or user-visible demos where it could be mistaken for evidence.

The same thing happened in the UI: exploration artifacts were allowed to accumulate in the
main surface instead of being forced through the approved three-screen product shape.

### What To Do About It

The recovery path is not to rebuild fake demos better. It is to rebuild fewer surfaces with
real evidence:

- Real benchmarks: browser run, real `/model` weights, raw run artifact, exact Chrome build,
  exact device, exact prompts, exact prefill/decode/tokens-per-second numbers.
- Real provider tests: dedicated test account, real OAuth or approved auth path, real API/DOM
  observation, read-after-write verification, cleanup path.
- Real app-action plan: idempotent writes, user confirmation boundary, audit trail, rollback or
  cleanup where possible, and explicit unsupported states.
- Real training proof: declared training source, declared target skill, before/after eval task,
  raw transcript or structured evidence.
- Real UI validation: first-run screen screenshots on desktop, mobile, and unfolded foldable,
  judged against the three approved screens and the 10-second comprehension bar.

If a real version cannot be built yet, the correct state is absence plus
`implementation_required`, not a substitute flow.

## Recovery Work Cards

Use these as the next-action shape. Each item is intentionally small and evidence-based.

### Real Browser Benchmark Card

- Build: benchmark route loads real model weights only.
- Evidence: raw browser run artifact with device, Chrome version, model path, prompt, prefill
  time, decode time, tokens per second, and adapter state.
- Accept when: README numbers are copied from that artifact and the artifact is checked in or
  otherwise permanently linked.
- Reject if: any number comes from local substitute weights, guessed timing, staged output, or
  a run that cannot be reproduced.

### Real Provider Evaluation Card

- Build: one dedicated test account path for one provider.
- Evidence: real auth, real read, planned write, confirmed write, read-after-write verification,
  and cleanup.
- Accept when: the evaluator can run against the dedicated account and produce a raw evidence
  artifact without relying on local provider state.
- Reject if: provider data is constructed locally or the pass condition only proves internal
  planner behavior.

### Real Skill Training Card

- Build: one skill training loop with declared source data and one before/after eval.
- Evidence: adapter artifact, training inputs, eval prompt set, before score, after score, and
  runtime notes.
- Accept when: the trained adapter changes measured behavior on the target task without hiding
  failure cases.
- Reject if: the eval only checks that UI state advanced or that a canned response appeared.

### UI Reset Card

- Build: only the approved three-screen product: skillbook home, skill training, job board.
- Evidence: desktop, mobile, and unfolded-foldable screenshots of first-run and post-training
  states.
- Accept when: a new user can identify the next action in 10 seconds without understanding
  adapters, model loading, extension architecture, or write-action internals.
- Reject if: future capability, runtime status, or navigation density becomes the center of the
  first screen.

## UI/UX Failure Analysis

The current web UI is in a bad state. It is not just "unfinished"; it is carrying too many
competing product ideas at once, and that makes the first-run experience hard for a human
to understand.

The approved direction was roughly:

- a core inventory / skillbook screen
- a skill training screen
- a job board / review screen

The actual app drifted into a soup of:

- model loader state
- account atlas state
- RPG stage scene
- dock of app icons
- adapter history
- training entry points
- job-board link
- dry-run/action-plan language
- performance/runtime status
- settings/source controls
- future extension assumptions

Those pieces are not individually irrational, but they are all visible or implied too early.
The first screen does not answer one clear question: "What can I do now?" Instead, it shows
status chrome, future capability, empty inventory, model boot, catalog roots, and secondary
navigation all at once.

The result is high cognitive load before the user has completed any successful loop. A normal
user is likely to bounce because the app appears powerful but not legible.

### Symptoms In The Current UI

- The first viewport has multiple dominant centers of attention instead of one primary loop.
- Empty-state copy, model loading, catalog browsing, and RPG flavor all compete.
- The dock suggests many apps are trainable or coming, but most are not immediately useful.
- "Atlas", "surface", "skill", "adapter", "account root", "chain", "job board", and "writes"
  all appear before the user has a simple success moment.
- The visual system has cohesion at the token/component level, but the composition is not
  cohesive.
- The RPG metaphor is spread across decorations and labels, but not expressed as one clean
  interaction model.
- The training interaction is not the obvious center of the product, even though training a
  skill is the flagship action.
- The interface reads like several demos assembled together rather than one product screen.

### How The Earlier Document Failed

The long consultant/architecture document did not function as an implementation contract.
It contained useful analysis, but it allowed too many directions to coexist:

- product strategy
- architecture review
- marketing analysis
- competitive positioning
- desktop/mobile/foldable ideas
- three approved screens
- possible extension form factors
- game references
- future action-layer concepts

That breadth made it easy for implementation to pick up fragments without preserving the
approved composition.

The document should have ended with a short, enforceable UI contract:

- exact primary screen hierarchy
- exact first-run flow
- what must not appear on first load
- which terms are allowed
- which surfaces are deferred
- pass/fail screenshots for desktop, mobile, and foldable
- acceptance criteria for "human can understand this in 10 seconds"

It did not do that tightly enough.

### How My Review Failed

The review failure was not only the document. I also validated the work too generously.

I treated visual vocabulary and architectural intent as evidence that the UI matched the plan.
That was wrong. A UI can use the right tokens, icons, and general concept while still failing
as a product experience.

The review should have judged:

- whether the approved three-screen structure was still visible
- whether the first-run screen had one obvious action
- whether the training loop was central
- whether future/locked/deferred items were hidden enough
- whether the screen could be explained without reading docs
- whether a human could complete the core loop without interpreting internal architecture

It did not apply that bar.

### Later UI Correction

When UI work resumes, the fix should not be incremental decoration. It should be a reset around
the approved three-screen product:

- Skillbook / inventory is the home screen.
- Training a skill is the primary first-run action.
- Job board is a secondary review/progression surface.
- Runtime/model status is subordinate, not the composition.
- Future app catalog items are hidden or heavily minimized until they are useful.
- Extension and real write actions stay out until architecture approves them.
- Terms are reduced to a small vocabulary and used consistently.

The next UI pass should start by removing surfaces, not adding components.

## Current Dirty State Notes

The worktree contains intentional cleanup edits and deletions. The untracked game UI asset
files in the repo root are legitimate design assets and are not part of the cleanup.

Before committing, review the diff file-by-file and decide whether the deleted prototype
artifacts should stay deleted or be replaced by real implementations in the same change.
