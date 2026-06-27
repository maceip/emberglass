# App-Action Layer Design (Saturday review.MD requirement)

**Status**: Design phase only. No implementation until this is reviewed and the relevant gates pass.

From Saturday review.MD:

> Design the app-action layer before implementation, including provider auth, write idempotency, read-after-write verification, and cleanup.

> Action-layer gate: raw model output never reaches executors; every write has ApprovalPacket and ExecutionReceipt; retries are idempotent or disabled; DOM actions fail closed on stale pages.

## Core Principles (enforced)

1. The model is a planner only. It emits macros over the typed port. Deterministic code does everything after that.
2. No raw model output ever goes to an executor.
3. Every write is derived from a VerifiedPlan that passed contract + scope checks.
4. User must see and approve an explicit ApprovalPacket before any mutation.
5. Every execution produces an ExecutionReceipt.
6. Idempotency is required or retries are disabled.
7. Fail closed on stale context (for DOM) or auth problems.
8. Real dedicated test accounts only for verification — no local stores.

## High-Level Pipeline (to be implemented later)

VerifiedPlan
  → resolve target account/provider (from equipped skill + current surface)
  → compile provider-specific ActionPlan (idempotencyKey, steps, permissions)
  → request any missing permissions
  → show ApprovalPacket UI (human-readable summary + risk)
  → user approves (or edits)
  → execute approved steps
  → collect ExecutionReceipt(s)
  → reconcile + audit log
  → (optional) offer undo where possible

## Key Data Shapes (initial)

See src/skills/types.ts for current VerifiedPlan / ActionPlan / ApprovalPacket / Receipt skeletons. They must be the source of truth.

## Trust Boundaries (non-negotiable)

- Model never sees tokens or session state.
- Content scripts (future) never see tokens.
- Only approved structured ActionPlan reaches any executor.
- DOM actions must re-validate origin + element identity at execution time.

## Next Real Steps (only after current Recovery Cards are solid)

- Flesh out exact ApprovalPacket UI contract.
- Define idempotency strategy per provider macro.
- Define receipt schema + audit log format.
- Design rollback/undo paths where feasible.
- Write the design review checklist before any executor code beyond dry-run.

This document is the real start of the required "design before implementation".
Do not write production write executors until this (or a successor) is accepted.
