# Architecture Review Including App Actions

This document reviews the Emberglass architecture from the lowest runtime
layers up through the future app-action layer. It is not an implementation PR.
It is a detailed architecture checkpoint so we can decide what to build next
without blending runtime, training, planning, and outbound actions into one
unsafe system.

## Core Conclusion

Emberglass should likely evolve into a Chrome extension with a side panel, but
the side panel is not the outbound app-action engine.

The system should be separated into three major layers:

```text
1. Core runtime and skill planner
   WebGPU inference, LoRA training, adapter hot-swap, skill contracts,
   macro generation, dry-run verification.

2. Surface-aware extension shell
   Side panel UI, current-tab context capture, account/surface detection,
   training context collection, prompt/dry-run experience.

3. App-action executor layer
   Future design pass. API writes, DOM writes, approvals, permissions,
   idempotency, receipts, audit log, rollback/compensation.
```

The extension direction is correct because the product is about browser/app
surfaces. It is not a license to start mutating those surfaces yet.

The model should remain a planner/compiler. It emits constrained macros over
typed ports. Deterministic code parses, verifies, resolves, approves, and only
later executes. The model should never directly execute page actions or API
writes.

## Current Boundary

The current product is:

```text
load model
  -> train skill
      -> equip skill
          -> generate typed macro / write plan
              -> verify plan against a contract
                  -> show dry-run output
```

The current product is not:

```text
execute the write
mutate user accounts
drive DOM interactions
call provider APIs on the user's behalf
store provider credentials
autonomously click or submit forms
```

That boundary is important. The hard part at this stage is proving that local
training can produce reliable, typed, verifiable plans over a real account or
surface. The outbound executor deserves a separate architecture pass because it
is where account authority, credentials, user trust, and irreversible side
effects enter.

## Existing Runtime Shape

The repo already contains useful separations that should be preserved.

### WebGPU Runtime

Relevant files:

```text
src/qwgpu/runtime.js
src/qwgpu/kernels.js
src/qwgpu/backward_kernels.js
src/qwgpu/trainer.js
src/qwgpu/buffer_pool.js
src/qwgpu/dispatch_plan.js
src/qwgpu/templates/forward/*.wgsl.jinja
src/qwgpu/templates/backward/*.wgsl.jinja
```

Responsibilities:

- Build and own the WebGPU device-facing model runtime.
- Load quantized/int4 model weights.
- Run prefill, decode, sampling, KV cache updates, and LoRA application.
- Run backward kernels and optimizer steps for in-browser LoRA training.
- Keep performance-critical buffers GPU-resident.
- Keep kernel generation deterministic through templates and checks.

Non-responsibilities:

- Browser extension permissions.
- Provider tokens.
- Page DOM inspection.
- API calls to Gmail, Calendar, CRM, GitHub, etc.
- User approval and action receipts.

This layer is the engine. It should be portable enough to run inside a normal
web page, a side panel, or another extension document. It should not import
`chrome.*`.

### Model Session

Relevant file:

```text
src/services/model_session.js
```

Responsibilities:

- Initialize WebGPU.
- Load tokenizer and weights.
- Own a `QwenWGPU` runtime instance.
- Provide generation as an async text stream.
- Apply runtime options such as decode batching and sampling top-k.

Non-responsibilities:

- Knowing what a Calendar action is.
- Parsing macro output.
- Managing account identity.
- Reading the active browser tab.
- Persisting provider credentials.

This is a good seam for an extension-hosted side panel. The side panel can
instantiate `ModelSession`, but `ModelSession` should not know it is running in
an extension.

### Training Controller

Relevant file:

```text
src/services/training_controller.js
```

Responsibilities:

- Convert examples into masked, shifted-label training batches.
- Use the same tokenizer and prompt formatter as inference.
- Initialize and attach trainable adapters.
- Drive train steps over examples.
- Apply the trained adapter to runtime after training.

The key property is completion-only training over target macro/program output.
The model learns to compile natural language into a constrained target
language, not to perform open-ended account behavior.

Non-responsibilities:

- Executing generated macros.
- Validating permissions.
- Calling provider APIs.
- Inspecting the DOM.

### Adapter Registry and Persistence

Relevant files:

```text
src/services/adapter_registry.js
src/services/store.js
src/lora_export.js
src/lora_gpu.js
```

Responsibilities:

- Register loaded or trained adapters.
- Hot-swap adapters into the runtime.
- Export adapters.
- Store adapter blobs and metadata locally.
- Keep the equipped skill separate from the base model.

Storage split:

- The web demo can keep using localStorage and IndexedDB.
- The extension should review adapter persistence separately: extension
  IndexedDB, OPFS, or `chrome.storage.local` with `unlimitedStorage`.
- Provider tokens must not live in the same persistence path as adapters.

### Skill Registry and Macro Verifier

Relevant files:

```text
src/skills.js
src/skills/types.ts
src/skills/inbox-calendar/port.ts
src/skills/inbox-calendar/contract.ts
src/skills/inbox-calendar/generate.ts
src/skills/inbox-calendar/providers/*.ts
```

Responsibilities:

- Define typed skill ports.
- Generate examples and held-out evals.
- Provide a prompt/system contract to the model.
- Parse emitted macros.
- Verify emitted macros.
- Keep provider-specific mappings out of the model target.

This is the most important product abstraction.

A skill should represent:

```text
account or app surface
  -> canonical action port
      -> training corpus
          -> verification contract
              -> provider profiles for future execution
```

The current Inbox & Calendar block is pointed in the right direction:

- `port.ts` defines provider-agnostic ops.
- `contract.ts` defines invariant checks.
- provider files map canonical ops toward Google, Microsoft, and Zoho methods.

The future action layer should consume provider profiles. The model should still
emit canonical ops, not provider API payloads.

## Bottom-to-Top Stack

The full system should be described bottom-up:

```text
Browser hardware and APIs
  WebGPU, IndexedDB, extension APIs, content scripts, OAuth flows

Emberglass engine
  tokenizer, model session, QwenWGPU runtime, LoRA trainer, adapter registry

Skill substrate
  ports, provider profiles, contracts, corpora, evals, lessons

Planner
  prompt -> model -> macro text -> parser -> contract verifier

Surface bridge
  active tab context, selected text, DOM summary, account/surface detection

Extension shell
  side panel UI, service worker broker, content script messaging

Action layer
  future executor registry, approvals, API/DOM writes, receipts, audit

Product UX
  Account Atlas, Train Surface, Job Board, World Map, dry-run action pane
```

Dependency direction should remain clean.

Bad dependency direction:

```text
QwenWGPU imports chrome.identity
TrainingController imports content script bridge
Skill contract calls Gmail API
Content script receives OAuth token
Model output directly calls executor
```

Good dependency direction:

```text
Side panel uses ModelSession
Side panel asks SurfaceBridge for read-only context
Planner verifies model output against SkillContract
Future Executor consumes VerifiedPlan after approval
ProviderExecutor requests token from AuthBroker
Content script performs DOM action only from approved ActionPlan
```

## Why a Chrome Extension Side Panel

A normal web app remains valuable for demos and training experiments. The real
product, however, is anchored on browser/app surfaces. For that, a side panel is
the right shell.

The side panel gives us:

- Persistent UI beside the active tab.
- A natural Account Atlas and skill inventory surface.
- A place for training and dry-run planning without leaving the app.
- Messaging with content scripts for current-tab context.
- A future home for permission and auth mediation.

Official Chrome references:

- Side panel API: https://developer.chrome.com/docs/extensions/reference/api/sidePanel
- Content scripts: https://developer.chrome.com/docs/extensions/develop/concepts/content-scripts
- Scripting API: https://developer.chrome.com/docs/extensions/reference/api/scripting
- Identity API: https://developer.chrome.com/docs/extensions/reference/api/identity
- Storage API: https://developer.chrome.com/docs/extensions/reference/api/storage
- Offscreen API: https://developer.chrome.com/docs/extensions/reference/api/offscreen

### Initial Side Panel Responsibilities

First side-panel responsibilities:

- Host Account Atlas / Train Surface UI.
- Show model readiness and adapter status.
- Show selected surface and equipped chain.
- Read current tab URL/title/origin through extension messaging.
- Ask a read-only content script for context.
- Feed captured context into training or prompting.
- Generate macro plans.
- Verify macro plans.
- Show dry-run write plan.

First side-panel non-responsibilities:

- Execute provider writes.
- Submit DOM forms.
- Hold OAuth refresh tokens.
- Decide permissions without user context.
- Treat generated macros as trusted code.

### Why Not Service Worker as Engine

The Manifest V3 service worker should not host the model runtime.

Reasons:

- It is lifecycle-managed by Chrome.
- It is not a persistent app runtime.
- It has no DOM access.
- It is a poor place for long WebGPU sessions and training jobs.
- Global variables should not be treated as durable state.

The service worker should be a broker:

- route messages;
- manage extension events;
- broker permissions;
- later broker auth;
- wake up content scripts;
- store small state;
- coordinate side-panel opening.

For MVP, the model should run in the side panel. If background continuity becomes
necessary later, evaluate an offscreen document or dedicated worker as a
technical spike.

## Extension Topology

Recommended extension topology:

```text
manifest.json
  permissions:
    sidePanel
    scripting
    storage
    activeTab or explicit host permissions
    identity later, only when action/auth layer begins

service_worker.js
  Extension broker
  Tab routing
  Side-panel routing
  Permission broker
  Future auth broker

sidepanel.html / sidepanel.js
  Account Atlas UI
  WebGPU model runtime
  LoRA trainer
  Adapter store UI
  Plan generation and dry-run verification

content/current_surface.js
  Read-only surface bridge
  Page metadata
  Selected text
  DOM summaries
  Surface hints

future content/action_dom.js
  DOM executor
  Only receives approved ActionPlan
  No provider tokens
  Starts with draft/fill-only actions

future providers/*.js
  API executors
  Google, Microsoft, Zoho, GitHub, CRM providers
  Token requests through AuthBroker only
```

### Message Channels

Recommended channels:

```text
side panel -> service worker
  getActiveTab
  requestSurfaceCapture
  requestHostPermission
  future requestProviderToken

service worker -> content script
  captureSurface
  future applyApprovedDomPlan

content script -> service worker -> side panel
  surfaceSnapshot
  selectedElement
  pageCapabilityHints
  future domActionReceipt

side panel -> future executor
  approved ActionPlan
```

The side panel should not assume a content script is present. It should request
capture through the service worker, which can inject or message the right script
depending on permissions and origin.

## Surface Bridge

The surface bridge is the most important extension feature before outbound
actions. It should be read-only at first.

### Surface Snapshot

First snapshot shape:

```ts
interface SurfaceSnapshot {
  version: '0.1';
  capturedAt: string;
  tab: {
    id: number;
    url: string;
    origin: string;
    title: string;
  };
  page: {
    appHint?: string;
    surfaceHint?: string;
    routeHint?: string;
    language?: string;
  };
  selection?: {
    text: string;
    length: number;
  };
  focus?: {
    tagName: string;
    role?: string;
    ariaLabel?: string;
    name?: string;
    placeholder?: string;
    valuePreview?: string;
  };
  domSummary: Array<{
    role?: string;
    tagName: string;
    textPreview?: string;
    ariaLabel?: string;
    hrefOrigin?: string;
    stableSelectorHint?: string;
  }>;
  privacy: {
    redactionsApplied: string[];
    maxChars: number;
  };
}
```

The snapshot should be intentionally lossy. We need enough context to train and
ground a skill, not a full copy of the user's page.

### Capture Policy

Capture should obey:

- User gesture first.
- Visible UI explaining what is captured.
- Origin shown before capture.
- Size limits.
- Redaction pass for obvious secrets.
- No cookies.
- No OAuth tokens.
- No localStorage/sessionStorage extraction from host pages.
- No hidden full-page scraping by default.

### Surface Detection

Detection should be heuristic first:

```text
origin
  mail.google.com -> Gmail
  calendar.google.com -> Google Calendar
  outlook.office.com -> Outlook
  github.com -> GitHub
  app.pipedrive.com -> Pipedrive

route / DOM hints
  /mail/u/.../#inbox -> inbox
  /calendar/u/... -> calendar event surface
  /issues -> issue surface
  /pull/... -> pull request surface
```

Detector output should be hints, not authority:

```ts
interface SurfaceDetection {
  accountKind: string;
  provider?: string;
  surface?: string;
  confidence: number;
  evidence: string[];
}
```

The user should be able to override the surface selection in the UI.

## Skill Blocks and Ports

The current Inbox & Calendar block should become the template for serious
skills.

### Skill Block Structure

Recommended shape:

```text
skill block
  manifest
  port
  contracts
  providers
  corpus generators
  held-out evals
  lessons
  icon metadata
```

### Canonical Port

The port is what the model targets.

For Inbox & Calendar:

```text
find_email(query) -> thread
compose_email(to, subject, body)
reply_email(thread, body)
forward_email(thread, to, note)
archive_email(thread)
label_email(thread, label)
schedule_send(to, subject, body, when)
create_event(title, start, end, remind_min)
set_reminder(text, when)
find_slot(duration_min, after, before) -> slot
rsvp(event, response)
```

This is provider-neutral. It is not Gmail JSON, Microsoft Graph JSON, or Zoho
JSON. That separation is correct.

### Contract

The contract turns model output into pass/fail criteria.

Examples already present:

- emitted calls must be known ops;
- emitted args must be valid for the op;
- calendar date/time literals must be ISO format;
- calendar events must not be zero-length;
- slot search windows must have `after < before`;
- out-of-scope requests must bounce cleanly.

Every new app surface should have family-specific contract checks.

Examples:

```text
CRM deal contract
  create_deal requires title and organization/person reference
  close_deal requires explicit status
  no money amount without currency

GitHub issue contract
  create_issue requires repo and title
  merge_pr cannot appear unless find_pr/open_pr appears first
  destructive operations require explicit user wording

Calendar contract
  end > start
  timezone must be resolved before execution
  reminder minutes must be bounded
```

The contract should be executable without the model and without the provider.

### Provider Profiles

Provider profiles translate canonical ops to provider concepts.

They should contain:

```ts
interface ProviderProfile {
  provider: string;
  label: string;
  discovery: {
    source: string[];
    revision: string;
    note: string;
  };
  conventions: {
    timeFormat: string;
    searchSyntax: string;
    recurrence?: string;
  };
  opMap: Record<string, string>;
  pools: ProviderPools;
}
```

The provider profile is not the executor. It is the bridge from skill semantics
to future executor semantics.

## Planner and Dry-Run Layer

The planner is the current action output layer, but it does not mutate anything.

### Planner Flow

```text
user request
  -> selected/equipped skill
      -> build prompt with skill system contract
          -> model generates macro text
              -> parse macro calls
                  -> verify against contract
                      -> display dry-run plan
```

### Macro Parse Output

Recommended internal shape:

```ts
interface MacroCall {
  index: number;
  assignedTo?: string;
  op: string;
  args: Record<string, MacroValue>;
  raw: string;
}

type MacroValue =
  | { kind: 'literal'; value: string | number | boolean }
  | { kind: 'ref'; symbol: string }
  | { kind: 'path'; symbol: string; path: string[] };
```

The current parser is intentionally simple. Before app actions, it should become
a real parser with:

- string literal parsing;
- number/boolean parsing;
- variable assignment;
- reference validation;
- no arbitrary expression evaluation;
- clear syntax errors.

### Verified Plan

The dry-run output should become structured:

```ts
interface VerifiedPlan {
  id: string;
  skillKey: string;
  providerHint?: string;
  surfaceHint?: string;
  macro: string;
  calls: MacroCall[];
  verification: {
    status: 'ok' | 'out_of_scope' | 'invalid';
    contractId: string;
    issues: Array<{
      severity: 'error' | 'warning';
      code: string;
      message: string;
      callIndex?: number;
    }>;
  };
  dryRun: Array<{
    callIndex: number;
    title: string;
    description: string;
    risk: 'none' | 'low' | 'medium' | 'high';
  }>;
}
```

This object is the handoff point between planner and future action layer.

Until the action layer exists, the UI should stop here.

## Future App-Action Layer

The app-action layer should be designed separately, but we can define its
expected boundary now.

The action layer consumes a verified plan. It does not consume raw model text.

### Pipeline

```text
VerifiedPlan
  -> resolve target account/provider
      -> compile provider-specific ActionPlan
          -> request missing permissions
              -> show ApprovalPacket
                  -> execute approved actions
                      -> collect receipts
                          -> reconcile final state
                              -> update audit log
```

### Action Plan

Recommended future shape:

```ts
interface ActionPlan {
  id: string;
  sourcePlanId: string;
  skillKey: string;
  provider: string;
  target: {
    accountId?: string;
    origin?: string;
    tabId?: number;
    surface?: string;
  };
  permissions: PermissionRequirement[];
  steps: ActionStep[];
  idempotencyKey: string;
  expiresAt: string;
}

interface ActionStep {
  id: string;
  callIndex: number;
  op: string;
  mode: 'api' | 'dom' | 'hybrid' | 'manual';
  preview: string;
  risk: 'low' | 'medium' | 'high';
  requiresUserConfirmation: boolean;
  compiled?: ProviderPayload | DomInstruction | ManualInstruction;
}
```

### Approval Packet

Every outbound action should have an approval packet:

```ts
interface ApprovalPacket {
  actionPlanId: string;
  title: string;
  summary: string;
  accountLabel: string;
  origin?: string;
  steps: Array<{
    id: string;
    preview: string;
    changes: string[];
    risk: string;
  }>;
  permissions: PermissionRequirement[];
  canEditBeforeRun: boolean;
  canRunPartially: boolean;
}
```

The user approves a deterministic plan, not vague natural language.

### Receipt

Execution must produce receipts:

```ts
interface ExecutionReceipt {
  id: string;
  actionPlanId: string;
  startedAt: string;
  finishedAt?: string;
  status: 'success' | 'partial' | 'failed' | 'cancelled';
  steps: Array<{
    stepId: string;
    status: 'success' | 'failed' | 'skipped';
    providerRequestId?: string;
    resourceId?: string;
    resourceUrl?: string;
    error?: string;
    undoHint?: string;
  }>;
}
```

Receipts are not optional. Without receipts, the system cannot be trusted to
mutate accounts.

### Idempotency

Action execution should be idempotent where possible.

Examples:

- Creating a calendar event should include an idempotency key in extended
  properties or a local receipt map when provider support is missing.
- Creating an issue should avoid duplicate issue creation when retrying after a
  network failure.
- Draft creation should be tracked by draft ID.
- DOM fill actions should not be retried blindly after the page changes.

The action layer should never retry a high-risk step without checking the
receipt/reconciliation state.

### Rollback and Compensation

Rollback is not universal. The architecture should distinguish:

```text
reversible
  created draft can be deleted
  event can be deleted
  label can be removed

compensatable
  sent email cannot be unsent, but follow-up correction can be drafted
  CRM note can be amended

final
  external message sent
  payment submitted
  irreversible workflow advanced
```

Approval UX should reflect this.

## API Actions vs DOM Actions

The future action layer should be API-first for structured account writes.

### API-First Surfaces

Good API-first targets:

- Calendar event creation.
- Email drafts and replies.
- GitHub issues and PR comments.
- CRM people/deals/activities.
- Notes/doc creation.
- Labels/categories.

Benefits:

- Structured validation.
- Better receipts.
- Easier idempotency.
- Less breakage from UI changes.
- Clear permission scopes.

Risks:

- OAuth complexity.
- Provider-specific semantics.
- Rate limits.
- Partial API coverage.
- Accounts with multiple tenants/workspaces.

### DOM-First Surfaces

DOM actions are useful when:

- The app has no public API.
- The needed feature is UI-only.
- The user is already on the right screen.
- We want draft/fill behavior instead of direct commit.

DOM actions should begin with low-authority modes:

```text
highlight
  Point to the element the user should use.

fill
  Fill fields but do not submit.

draft
  Prepare content in the UI for user review.

click-with-confirmation
  Only after explicit user approval and surface stability checks.

submit
  Later trust level; only with receipts or strong confirmation.
```

### DOM Executor Constraints

A DOM executor must obey:

- Never receive provider tokens.
- Never execute raw model output.
- Only receive structured approved `DomInstruction`.
- Check page origin and route before acting.
- Check target element still matches expected selector/role/text.
- Prefer user-visible draft/fill operations.
- Emit receipt with before/after evidence when safe.
- Fail closed when the page changes.

Possible `DomInstruction`:

```ts
interface DomInstruction {
  kind: 'fill' | 'click' | 'select' | 'highlight' | 'draft';
  target: {
    selector?: string;
    role?: string;
    label?: string;
    textNear?: string;
  };
  value?: string;
  preconditions: Array<{
    type: 'origin' | 'urlIncludes' | 'elementText' | 'visible';
    value: string;
  }>;
}
```

## Auth and Token Boundary

Provider tokens belong to the future action layer, not the runtime.

Rules:

- The model never sees tokens.
- Content scripts never receive provider tokens.
- Adapters and training examples should not include tokens.
- OAuth state belongs to the extension service worker/auth broker.
- Short-lived tokens should prefer in-memory or `chrome.storage.session`.
- Long-lived refresh tokens should be avoided unless the action-layer design
  explicitly justifies them.
- Account identity should be represented as a stable local account reference,
  not a raw token or secret.

### Google

Google can use `chrome.identity.getAuthToken()` where appropriate.

Considerations:

- Scope should be per capability, not blanket.
- User interaction should explain why the scope is needed.
- Calendar and Gmail scopes should be requested separately if possible.
- Token cache behavior should be treated as Chrome-owned, not app-owned.

### Non-Google Providers

Microsoft, Zoho, Pipedrive, Gong, etc. likely need OAuth via
`launchWebAuthFlow` and PKCE.

Action-layer design must specify:

- redirect URI strategy;
- token exchange location;
- whether a backend is required;
- refresh behavior;
- revocation behavior;
- tenant/workspace selection;
- how account labels are shown to the user.

## Permission Model

There are two permission families:

```text
browser permissions
  sidePanel
  scripting
  storage
  activeTab
  host_permissions
  identity

provider permissions
  Gmail read/write
  Calendar read/write
  Graph Mail/Calendar scopes
  GitHub repo scopes
  CRM scopes
```

These should not be blurred.

### Extension Permissions

For early side-panel work:

- Use minimal permissions.
- Prefer user-triggered `activeTab` for read-only capture during experiments.
- Move to explicit host permissions only for surfaces that need persistent
  capture.
- Avoid all-sites permissions for MVP.

### Provider Permissions

Provider permissions should be requested only when crossing from dry-run into
actual execution.

For example:

```text
Generating a calendar macro:
  no provider permission required

Dry-running create_event:
  no provider permission required

Checking free/busy for real:
  calendar read permission required

Creating an event:
  calendar write permission required
```

## Data Privacy Boundary

The architecture should classify data:

```text
public/static
  model config, skill metadata, icon catalog, provider profiles

local model assets
  base weights, adapter weights, tokenizer files

surface context
  page title, URL, selected text, DOM summaries, app hints

training data
  examples, generated macros, adapter artifacts

account authority
  OAuth tokens, provider account IDs, tenant IDs, scopes

execution records
  action plans, approvals, receipts, audit logs
```

Rules:

- Account authority must not mix with training data.
- Surface context should be minimized and visible to the user.
- Receipts should redact sensitive payloads by default.
- Exported adapters may encode private behavior/examples and should be treated
  as sensitive.
- A user should be able to delete local adapter runs and captured context.

## UX Architecture

The RPG/Atlas metaphor still fits, but it should map to real architecture.

### Account Atlas

Purpose:

- Show connected or detected accounts.
- Show trainable surfaces.
- Show equipped skill/chain.
- Show dry-run write capability.
- Show model and adapter status.

Architecture mapping:

```text
Account tile
  provider profile + optional account reference

Surface node
  skill block + surface detection evidence

Equipped skill
  adapter + skill key + provider hint

Command box
  prompt -> planner -> verified plan

Action trace
  dry-run plan now
  receipts later
```

### Train Surface

Purpose:

- Convert current-tab or account-surface context into training/eval examples.
- Show exact target port and examples.
- Run LoRA training.
- Validate with held-out drills.
- Equip the result.

Architecture mapping:

```text
source context
  SurfaceSnapshot

target
  SkillBlock / Port

examples
  generated or user-curated pairs

training
  TrainingController

validation
  contract + held-out eval set
```

### Dry-Run Write Pane

This is the UI that should exist before action execution.

It should show:

- raw macro;
- parsed calls;
- contract status;
- provider profile that would be used later;
- required permissions for execution;
- simulated action plan;
- why it is not executed yet.

### Future Approval Screen

When actions exist, approval should be explicit:

```text
You are about to:
  1. Create a calendar event "Q3 roadmap with Sarah"
     Monday 2026-06-29 17:00-17:30
  2. Add a 10 minute reminder

Account:
  Google Calendar - user@example.com

Permission:
  Calendar write

Run / Edit / Cancel
```

The user approves a concrete plan, not "do what I asked."

## Observability and Testing

The system needs tests at each layer.

### Runtime Tests

Already present or implied:

- kernel generation check;
- inference correctness;
- LoRA load path;
- backward/training validation;
- benchmark harness.

### Skill Tests

Needed:

- every generated training pair parses or cleanly bounces;
- held-out eval examples exist;
- contract checks catch known bad macros;
- provider profiles cover every canonical op;
- op maps are complete;
- no generated examples use provider dialect directly.

### Surface Bridge Tests

Needed for extension work:

- content script captures only allowed fields;
- redaction tests;
- origin detection tests;
- route/surface detection fixtures;
- no token/cookie/localStorage extraction;
- failure on restricted pages.

### Planner Tests

Needed:

- macro parser golden tests;
- invalid syntax tests;
- variable reference tests;
- dry-run plan rendering tests;
- out-of-scope behavior tests;
- contract issue formatting tests.

### Future Action Tests

Needed before real writes:

- provider executor unit tests against mocked APIs;
- idempotency tests;
- partial failure tests;
- approval packet snapshot tests;
- receipt generation tests;
- DOM executor tests on fixture pages;
- permission-denied tests;
- stale page precondition tests.

## Migration Path

### Phase 0: Current Browser App

Keep the existing browser app as the public demo and trainer.

Capabilities:

- load model;
- train adapter;
- equip skill;
- generate macro;
- verify macro;
- show dry-run output.

### Phase 1: Side Panel Shell

Create extension shell that hosts the same product loop.

Capabilities:

- side panel opens beside active tab;
- model can run in side panel;
- adapter storage works in extension context;
- no outbound writes.

### Phase 2: Read-Only Surface Bridge

Add content script capture.

Capabilities:

- active tab metadata;
- selected text;
- DOM summary;
- app/surface detection;
- context visible before use;
- no outbound writes.

### Phase 3: Structured Planner

Harden macro parsing and dry-run planning.

Capabilities:

- structured macro AST;
- `VerifiedPlan`;
- dry-run plan object;
- permission preview;
- provider profile preview;
- no outbound writes.

### Phase 4: Separate App-Action Architecture Approval

Before implementation, produce and approve a dedicated action-layer design.

Must decide:

- executor registry shape;
- first provider;
- OAuth flow;
- approval UX;
- receipt schema;
- idempotency mechanism;
- rollback/compensation rules;
- DOM executor limits.

### Phase 5: Controlled Draft Actions

First execution should be low-risk:

- create draft, not send;
- fill fields, not submit;
- create event only after explicit approval;
- preserve receipts;
- no broad autonomous DOM mutation.

### Phase 6: Higher-Authority Actions

Only after Phase 5 is reliable:

- sending email;
- submitting forms;
- updating CRM records;
- executing chained workflows;
- multi-app actions.

## Key Risks

### Conflating Planner and Executor

Risk:

```text
model output -> direct write
```

Mitigation:

```text
model output -> parser -> contract -> verified plan -> approval -> executor
```

### Over-Capturing User Data

Risk:

The extension captures too much DOM/account context and trains on private data
without clear user awareness.

Mitigation:

- read-only capture first;
- visible snapshot preview;
- redaction;
- size limits;
- delete controls;
- no hidden broad scraping.

### Token Leakage

Risk:

Provider tokens end up in prompts, content scripts, logs, adapters, or receipts.

Mitigation:

- token broker only;
- no tokens to model;
- no tokens to content script;
- storage separation;
- receipt redaction.

### Fragile DOM Actions

Risk:

DOM executor clicks the wrong thing after page changes.

Mitigation:

- start with highlight/fill/draft only;
- preconditions;
- route checks;
- element checks;
- user approval;
- fail closed.

### Extension Runtime Instability

Risk:

WebGPU model hosted in the wrong extension context.

Mitigation:

- MVP: side panel hosts model;
- service worker routes only;
- offscreen/worker runtime only after tested.

## Open Questions

- Should the side panel directly host the engine for the first extension MVP?
  Current recommendation: yes.
- Should adapter blobs use extension IndexedDB, OPFS, or `chrome.storage.local`
  with `unlimitedStorage`?
- What is the minimum useful `SurfaceSnapshot` for Calendar/Gmail training?
- Do we need per-surface redaction policies before training?
- Which first provider should the future action layer target?
  Calendar is likely best because event creation has clear structure and
  receipts.
- What is the first DOM action trust level?
  Recommendation: highlight/fill only.
- How do equipped chains represent account identity without storing secrets in
  the skill/adapters?
- Should provider profiles be packaged in a catalog separate from trained
  adapters?
- What is the receipt retention/deletion policy?

## Recommendation

The architecture should be:

```text
Core Runtime
  pure web runtime, no chrome APIs

Skill System
  ports, contracts, examples, provider profiles

Planner
  model output -> parsed macro -> verified plan -> dry run

Extension Shell
  side panel UI + service worker broker + read-only content script

Future Executor
  separate action layer that consumes VerifiedPlan only after approval
```

Immediate recommendation:

```text
Build toward the extension shell, but keep the first extension milestone
read-only and dry-run only.
```

Do not implement "right actions" as a side effect of extension work. Treat app
actions as their own later milestone with its own architecture review,
permission model, tests, and approval bar.
