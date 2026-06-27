# Emberglass Product, Marketing, Architecture, Evaluation, and App-Actions Review

This document consolidates the prior engineering architecture review with a
stronger product and marketing lens. It is intentionally a consulting,
criticality, and operating-plan document: it should tell the team what to do,
what to defer, what to measure, and what evidence is required before the next
phase.

Backups of the two source drafts are preserved outside the repo:

```text
/Users/mac/architecture_review_engineering_draft.md
/Users/mac/architecture_review_product_marketing_draft.md
```

## Executive Thesis

The core product is not "in-browser LoRA." The core product is:

```text
Accounts, apps, and browser tabs become trainable skills.
```

The user-facing promise should be:

```text
Train a private skill for the app you are using, equip it, and turn requests
into verified write plans.
```

The technology is interesting because it makes that promise believable:

- WebGPU makes the model local and interactive.
- LoRA makes a skill feel trainable and ownable.
- Contracts make the output trustworthy.
- The RPG/Account Atlas metaphor makes skill ownership understandable.
- A future extension side panel makes browser-tab and account context natural.

The product is interesting enough to get people to use it if the first two
minutes prove the promise. It will fail if the first experience feels like a
model-loading dashboard or a vague browser agent.

## Consultant Directives

These are the decisions this review recommends the company make now.

### Decide Now

| Decision | Directive | Rationale |
|---|---|---|
| Product wedge | Make Calendar Skill the flagship | It is universal, visually inspectable, low-risk, and contract-friendly |
| Category | Use "account skills" / "browser skills" | More memorable than local LoRA or browser agent |
| First shell | Build desktop around a Chrome side panel | Chrome side panels are designed for persistent extension UI beside browsing [S1] |
| Mobile stance | Do not promise mobile extension parity | Chrome's documented phone flow installs to desktop, and historical Chromium guidance says Chrome Android extensions are unsupported [S6][S7] |
| First extension capability | Read-only current-tab capture | Content scripts can read page DOM and pass information to the extension, but mutation should wait [S2] |
| Action layer | Defer execution into a separate milestone | API/DOM writes require approvals, auth, idempotency, receipts, and rollback policy |
| Trust model | Market dry-run as a feature | "Verified plan ready, no account changed" is safer and clearer than autonomous action |
| Evaluation | Measure base-vs-trained contract lift | Proves training matters; tok/s alone does not prove product utility |

### Build Now

Build these next:

1. Calendar first-run demo.
2. Approved product-screen wireframes:
   Home / Inventory, Skill / Train Surface, Job Board.
3. Calendar eval suite with base-vs-trained scorecard.
4. VerifiedPlan parser/schema spec.
5. Benchmark report template centered on time-to-first-valid-plan.
6. Landing page copy that leads with account skills.

### Do Not Build Yet

Do not build these yet:

1. Real API writes.
2. DOM click/submit automation.
3. OAuth scope flows beyond prototypes.
4. Multi-provider execution.
5. Broad all-skill catalog as the first screen.
6. Mobile extension-dependent UX.
7. General autonomous browser-agent positioning.
8. Form-factor-only mockups that replace the approved screen set.

### Non-Negotiable Gates

Do not promote the next phase unless these gates pass:

```text
Calendar flagship gate
  base-vs-trained contract lift is visible
  starter training completes inside the target demo window
  first valid dry-run is reachable in under two minutes
  user can explain what the skill can write

Extension shell gate
  read-only capture is explicit and user-triggered
  no provider tokens enter prompts, adapters, content scripts, or logs
  side panel works as a companion, not a detached dashboard

Action-layer gate
  raw model output never reaches executors
  every write has ApprovalPacket and ExecutionReceipt
  retries are idempotent or disabled
  DOM actions fail closed on stale pages

UI gate
  Home / Inventory exists as the primary screen
  Skill / Train Surface exists as the training screen
  Job Board exists as the comparison/evaluation screen
  desktop/foldable/mobile variants map back to these screens
```

### Operating Principle

Use this rule when making product and engineering tradeoffs:

```text
If a feature does not make the first Calendar skill feel easier, safer, or more
magical, it is probably a distraction for this phase.
```

## Prescriptive Delivery Plan

The company should run this as a staged wedge, not a broad platform launch. The
right question for each stage is not "can we build more?" The right question is:

```text
Does Calendar Skill make the account-as-skill idea feel inevitable?
```

If yes, widen. If no, fix Calendar before adding breadth.

### Phase 0: Reset Scope This Week

Directive:

```text
Freeze the public story around Calendar Skill, account skills, dry-run trust,
and local browser training.
```

Ship:

1. One canonical product sentence.
2. One Calendar first-run script.
3. One eval fixture file for Calendar.
4. One benchmark report template.
5. One product-screen map:
   Home / Inventory -> Skill / Train Surface -> Job Board.
6. One explicit non-goals page.

Cut:

1. Any homepage section that leads with "LoRA."
2. Any demo screen where the first visible state is model-loading internals.
3. Any copy implying autonomous account mutation.
4. Any mobile promise that depends on Chrome Android extension support.
5. Any broad catalog that visually competes with Calendar.

Acceptance criteria:

| Artifact | Must Prove | Reject If |
|---|---|---|
| Product sentence | A normal user understands the benefit | It requires explaining LoRA first |
| First-run script | A user reaches a verified Calendar plan quickly | The flow stalls on setup language |
| Eval fixture | Training improvement is measurable | It only stores happy-path prompts |
| Benchmark template | Product time metrics sit beside tok/s | It reports only raw kernel speed |
| Product-screen map | The three approved screens are present | It substitutes desktop/mobile/foldable as the screens |
| Non-goals page | Team stops widening scope prematurely | It reads like vague caution |

### Phase 1: Calendar Skill MVP

Directive:

```text
Make one skill feel complete before making many skills look possible.
```

Build:

1. Home / Inventory screen with Calendar selected.
2. Skill / Train Surface screen for Calendar.
3. Job Board shell with Calendar, Gmail, Notes, and locked workflow skills.
4. Calendar equipped command prompt.
5. VerifiedPlan preview.
6. Dry-run success receipt.

Do not build:

1. Live calendar write execution.
2. Generic browser automation.
3. OAuth provider switching as a primary demo.
4. CRM/email/Gong chains as interactive first-run flows.
5. Character cosmetics that distract from account-skill progression.

Acceptance criteria:

```text
Calendar Skill MVP passes when:
  user can name the current screen in under 5 seconds
  Home makes trained/equipped/locked skills visually obvious
  Skill screen lets the user train without reading docs
  starter training completes inside the demo target
  post-training eval visibly beats base
  Job Board compares reliability and suggests retraining
  dry-run plan is parseable, typed, and inspectable
  user can explain "no account changed" without prompting
```

Minimum eval gate:

| Category | Required State Before Public Demo |
|---|---|
| Date normalization | Handles relative dates and explicit dates |
| Duration | Computes end time from duration |
| Ambiguity | Asks or emits clarification, not invented values |
| Scope control | Rejects non-calendar asks |
| Provider contract | Uses allowed Calendar macros only |
| Injection pressure | Ignores page/user text that tries to alter contract |
| Base-vs-trained | Shows meaningful lift on held-out cases |

The team should set the exact numeric thresholds after one more browser run,
but the acceptance rule should be fixed now: do not show only qualitative
examples. Show the scorecard.

### Phase 2: Extension Shell

Directive:

```text
Use the extension to make account context feel native, not to execute writes.
```

Build:

1. Chrome side-panel shell using the side panel API [S1].
2. User-triggered current-tab capture through content scripts [S2].
3. Optional runtime injection path behind activeTab/scripting permissions [S3].
4. SurfaceSnapshot schema with redaction and provenance.
5. Side-panel "current tab understood" state.
6. Explicit "read-only capture" trust copy.

Do not build:

1. Background scraping.
2. Always-on full-history collection.
3. Token access in content scripts.
4. DOM submit/click automation.
5. Multi-site mutation policy.

Acceptance criteria:

```text
Extension shell passes when:
  capture requires a visible user action
  captured origin, title, selected text, and DOM summary are shown back
  sensitive fields are redacted or excluded before model use
  user can delete the captured snapshot
  no provider OAuth token reaches the content script or prompt
  panel remains useful across tab navigation
```

This phase should create a strong demo moment:

```text
Open calendar or email tab
  -> open Emberglass side panel
      -> capture visible context
          -> train or use Calendar Skill
              -> produce verified dry-run
```

### Phase 3: Three-Screen Product System

Directive:

```text
Implement the approved product screens before adding more product surfaces.
```

Build:

1. Home / Inventory as the primary operating surface.
2. Skill / Train Surface as the training and inspection surface.
3. Job Board as the comparison and retraining surface.
4. Shared skill state across all three screens.
5. Desktop, unfolded foldable, and mobile layouts for each screen.

Do not build:

1. A giant app-store grid as the first screen.
2. Skill icons without capability states.
3. Skill chains that lack typed intermediate contracts.
4. Visual RPG flourishes that do not explain training/progress/equip.
5. World Map as a fourth screen in this cycle.
6. Device-form-factor mockups that replace the approved three screens.

Acceptance criteria:

```text
Three-screen product passes when:
  user can tell which skills are trained, equipped, locked, and available
  user can move from Home to Skill / Train Surface without hunting
  user can compare skills and decide what to retrain on Job Board
  account-rooted progression is visually obvious
  Calendar remains the most complete branch
  each screen has desktop, unfolded foldable, and mobile behavior
  device variants do not become separate product screens
```

### Phase 4: Action Layer Planning

Directive:

```text
Design writes as a separate trust product.
```

Build only after the dry-run product is credible:

1. ActionPlan compiler.
2. PermissionRequirement model.
3. ApprovalPacket UI.
4. ExecutionReceipt log.
5. Provider-specific API executor for one low-risk write.
6. DOM executor only for highlight/fill/draft behavior.

Do not build:

1. Raw model-to-DOM action.
2. One-click destructive submit.
3. Silent retries.
4. Token persistence without a concrete threat model.
5. Multi-provider write automation before one provider is boringly reliable.

Acceptance criteria:

```text
Action layer passes when:
  every write is generated from a VerifiedPlan
  every write has an ApprovalPacket before execution
  every execution emits a Receipt
  every retry is idempotent or disabled
  every DOM action validates origin and element state at runtime
  every failure leaves the account unchanged or explicitly partial with receipt
```

### Operating Metrics

The team should review these weekly:

| Metric | Owner | Why It Decides Strategy |
|---|---|---|
| Time to first valid Calendar dry-run | Product + runtime | Determines whether the demo earns patience |
| Base-vs-trained contract lift | ML/runtime | Proves training is the feature |
| Prompt-to-plan latency | Runtime | Determines whether the prompt feels alive |
| Dry-run comprehension | Design | Determines whether trust copy works |
| Capture-to-plan success | Extension | Determines whether the side panel is worth it |
| OUT_OF_SCOPE pass rate | Safety/product | Keeps the skill from becoming a vague agent |
| User replay rate | Product | Best early signal that the concept has pull |

Weekly decision rule:

```text
If Calendar quality is weak, do not add skills.
If first-run is slow, do not add extension depth.
If dry-run is confusing, do not add writes.
If users do not replay, do not polish the RPG shell before fixing usefulness.
```

### Prescriptive Resourcing

For the next build cycle, allocate effort this way:

| Workstream | Share | Reason |
|---|---:|---|
| Calendar UX and eval | 35% | This is the product wedge |
| Runtime performance and training reliability | 25% | This is the credibility layer |
| Side-panel read-only extension prototype | 20% | This is the distribution/form-factor bet |
| Product copy and first-run polish | 10% | This makes the story repeatable |
| Competitive/demo benchmarking | 10% | This supports launch confidence |

Explicitly allocate 0% this cycle to live writes, broad provider automation, and
general browser-agent autonomy.

## North-Star Product Question

```text
Is this interesting enough to get people to use it?
```

Answer: yes, with a narrow first wedge.

Best first wedge:

```text
Calendar Skill
  -> train locally
      -> improve on drills
          -> emit create_event(...)
              -> contract passed
                  -> dry run only
```

The flagship moment should be:

```text
Before training:
  vague output or invalid macro

After training:
  create_event(title="Sarah sync",
               start="2026-06-30T14:00",
               end="2026-06-30T14:30",
               remind_min=10)
  contract passed
  no account changed
```

That is the shareable product loop.

## Product and Marketing Lens

### What Is The New Thing?

The new thing is an account-surface skill system.

Plain explanation:

```text
Every app has surfaces: inbox, calendar, deals, issues, notes, calls, threads.
Emberglass lets you train a small local skill for one surface, equip it, and
produce verified write plans for that surface.
```

Examples:

```text
Google Calendar -> Calendar Skill
Gmail Inbox -> Inbox Skill
Pipedrive Deals -> Deal Skill
Gong Calls -> Call Follow-up Skill
GitHub Issues -> Issue Triage Skill
Slack Threads -> Thread Reply Skill
Notion Notes -> Notes Skill
```

Internal phrase:

```text
fine-tune each browser tab
```

External phrase:

```text
train a skill for this app
train a private skill for this account
turn this tab into an equipped skill
```

### Category Positioning

Avoid positioning as:

```text
fastest browser LLM
generic local chat app
autonomous browser agent
Zapier replacement
Chrome AI replacement
```

Position as:

```text
private browser skills
trainable app surfaces
account skills
verified write plans
local skill ownership
dry-run before action
```

Best current one-liner:

```text
Train private skills for the apps you use, then turn requests into verified
write plans.
```

More playful:

```text
An RPG skillbook for your browser tabs.
```

More technical:

```text
Local WebGPU LoRA training for account-specific browser skills with
contract-verified macro output.
```

The public product should lead with the first two. The technical line is proof,
not the hook.

### Landing Page Message Order

Recommended order:

```text
1. Skill ownership
   "Train a private skill for this app."

2. Visible utility
   "Ask for a calendar change and see a verified write plan."

3. Trust boundary
   "Dry run first. Nothing runs without approval."

4. Locality
   "The model runs and learns in your browser."

5. Technical proof
   "Custom WebGPU runtime, LoRA hot-swap, contract checks."
```

Do not start with:

```text
WebGPU LoRA fine-tuning demo
```

That phrase is impressive to technical people but too abstract as a first hook.

### Suggested Landing Page Skeleton

```text
H1
  Train a private skill for this app.

Subhead
  Emberglass turns browser tabs and accounts into local skills that produce
  verified write plans before anything runs.

Primary CTA
  Train Calendar Skill

Proof strip
  Runs locally in your browser
  Trains a skill on-device
  Verifies every write plan
  Dry-run first

Demo section
  Before training: invalid/vague
  After training: create_event(...) contract passed

Account Atlas section
  Gmail -> Calendar -> Pipedrive -> Gong

Trust section
  No hidden actions
  No provider tokens in prompts
  User approves execution when action mode ships

Technical section
  Custom WebGPU runtime
  Hot-swappable adapters
  Contract-checked skill ports
  Benchmarks
```

### User Personas

#### AI Builders and Local-First Tinkerers

Why they care:

- in-browser fine-tuning is novel;
- WebGPU performance is interesting;
- local/private model work is credible;
- skill contracts make the agent story less vague.

What they need:

- clear architecture;
- benchmarks;
- GitHub demo;
- proof training changes behavior;
- inspectable macros and contracts.

#### Power Users and Productivity Operators

Why they care:

- they live in email, calendar, notes, CRM, GitHub, and Slack;
- they want workflows without fragile automation setup;
- they like tools that feel personal.

What they need:

- low setup;
- one obvious first skill;
- no fear of hidden account mutation;
- visible dry-run plans.

#### Sales, Ops, and Founder Workflows

Why they care:

- email -> CRM -> call notes -> follow-up is repetitive;
- cross-app handoffs are painful;
- account-specific context matters.

What they need:

- Gmail/Calendar first;
- Pipedrive/Gong fast-follow story;
- eventually API execution and receipts;
- strong privacy posture.

## UI/UX Review: Approved Three-Screen Product

This section replaces the prior form-factor-first wireframes. The approved
product screens are:

```text
1. Home / Inventory
2. Skill / Train Surface
3. Job Board
```

Desktop, unfolded foldable, and mobile are not additional screens. They are
implementation variants of these three screens.

The prior framing was wrong because it treated device layouts as the product
architecture. The corrected directive is:

```text
Design the three approved screens first.
Then adapt each screen to desktop, unfolded foldable, and mobile.
Do not ship a desktop/foldable/mobile gallery as a substitute for the product
screen set.
```

World Map remains a later progression idea. It should not be part of the core
three-screen MVP and should not appear as a first-class screen until Home,
Skill / Train Surface, and Job Board are coherent.

### Product Screen Model

The app should feel like an RPG system because the user is managing trained
capabilities, not because the UI is decorative.

Good metaphor:

```text
Home / Inventory
  the user's equipped skills, account roots, and current command surface

Skill / Train Surface
  the forge or spell page where one account skill is trained, tested, and
  equipped

Job Board
  the notice board that tells the user what skills need work, what failed,
  and what should be trained next
```

Weak metaphor:

```text
generic fantasy character sheet
decorative inventory with no capability information
world map before the user has trained anything
levels that do not map to evals, contracts, or allowed writes
game UI that hides the actual productivity task
```

The user should understand three sentences:

```text
Home: I can see and equip my trained account skills.
Skill: I can train Calendar and see exactly what improved.
Job Board: I can see which skills are reliable and what to fix next.
```

### Game Design Pass

Read this product like a game first and a tool second. The current direction is
strong, but the UI still risks feeling like a dashboard wearing RPG clothes.
The better target is a playable loop:

```text
notice a useful locked or weak skill
  -> train through a short challenge
      -> watch the skill visibly improve
          -> equip it
              -> try it immediately
                  -> receive a new reason to keep playing
```

The core game fantasy:

```text
I am building a spellbook of account skills.
Each skill is earned through trials.
Equipping a skill changes what I can do.
The world opens outward through useful apps and accounts.
```

This is more compelling than:

```text
I configured a local model and reviewed its metrics.
```

Player-facing screen names can carry more flavor while preserving the approved
product architecture:

```text
Home / Inventory
  player-facing feel: Skillbook
  purpose: choose what is equipped and cast a request

Skill / Train Surface
  player-facing feel: Training Page / Forge / Trial Page
  purpose: make one skill stronger through visible challenges

Job Board
  player-facing feel: Quest Board
  purpose: show what needs work and what can be unlocked next
```

Do not add a fourth screen for this flavor. The names are mood and copy; the
screen set remains Home / Inventory, Skill / Train Surface, and Job Board.

### Player Experience Directives

1. Give the player one obvious action per state.

Every screen needs one dominant verb:

```text
Home
  Equip or Cast

Skill
  Train

Job Board
  Claim next task
```

Secondary verbs can exist, but they should visually defer.

2. Replace abstract status with earned state.

Do not lead with raw percentages as the emotional layer. Lead with state labels
that feel earned, then show numbers as proof:

```text
Untrained
Learning
Reliable
Mastered
Rusty
Locked
```

Example:

```text
Calendar
  Reliable
  93% trial pass
```

This is more game-readable than:

```text
Calendar
  eval 93%
```

3. Make training feel like a short challenge, not a configuration form.

The player should see a few named trials:

```text
Trial 1: Create a simple event
Trial 2: Resolve "tomorrow afternoon"
Trial 3: Reject an email request
Trial 4: Find a free slot
```

The win state is not "training completed." The win state is:

```text
Calendar learned the trial.
Calendar can now be equipped.
```

4. Use locks as promises, not walls.

Locked skills should always explain what they need and why the player wants
them:

```text
Pipedrive Deals
  Locked
  Requires: Gmail Inbox Reliable
  Unlocks: update deal after call
```

Bad lock:

```text
Pipedrive locked
```

5. Make failure useful and named.

A failed drill should feel like a quest hook:

```text
Weakness discovered: reply/calendar confusion
Next trial: separate meeting requests from email replies
Reward: Gmail -> Calendar chain becomes available
```

This turns error recovery into progression.

6. Keep the RPG language grounded in usefulness.

Avoid decorative fantasy terms that hide the real work. Use a small vocabulary:

```text
Skill
Equip
Train
Trial
Unlock
Reliable
Cast
Plan
```

Avoid:

```text
mana
HP
damage
loot rarity
randomized stats
boss fights
```

The fantasy is mastery of accounts, not cosplay.

7. Give every completed action a visible state change.

The UI should never respond with only text. A player action should update at
least one visible object:

```text
Train
  failed trial card flips to pass
  skill badge fills from Untrained to Reliable
  locked child skill reveals prerequisite progress

Equip
  skill moves into the equipped slot
  command box changes placeholder to match that skill
  allowed actions appear as icons

Cast
  plan card receives a passed seal
  no-account-changed state remains visible
  Job Board receives a new receipt/task
```

8. Design the first two minutes like a tutorial level.

The first run should not ask the player to understand the whole system. It
should stage the loop:

```text
Start with Calendar visible and almost ready.
Show one locked future skill below it.
Ask the player to train one starter trial pack.
Show three trials passing.
Let the player equip Calendar.
Let the player cast one request.
Show the verified plan seal.
Reveal the next board task.
```

### Screen-Level Game Improvements

Home / Inventory should improve from "management panel" to "equipped loadout."

Priority changes:

```text
make the equipped slot visually heroic
show Calendar as the starting spell, not one item in a catalog
show one future unlock teaser under the core roots
make the command box feel like using the equipped skill
move raw metrics below state labels
```

Skill / Train Surface should improve from "training workbench" to "trial page."

Priority changes:

```text
replace the first impression of settings with named trials
show before/after as a reveal
show "what changed" in plain player terms
make failed drills become new tasks
make Train Starter Skill the dominant action
```

Job Board should improve from "analytics table" to "quest board."

Priority changes:

```text
pin recommended tasks as cards before showing dense comparison
make each task include reward, weakness, and next action
show locked chains as future promise
use status stamps instead of rows of similar numbers
keep the table as a secondary detail view on desktop
```

### Game Feel Acceptance Criteria

The three-screen loop should pass these tests before visual polish:

```text
the player knows what to click next within 3 seconds
the player sees one skill get visibly stronger
the player understands why a locked skill is desirable
the player can describe the next unlock in plain language
failure creates a new task instead of a dead end
equipping changes the command surface visibly
the Job Board feels like progress, not chores
the RPG layer makes the tool easier to understand
```

### Asset Footnote Convention

Wireframe markers such as `*H1`, `*S1`, and `*J1` are intentional references to
the UI source images in `docs/ui/`. Keep those markers in implementation specs,
tickets, and design QA notes so everyone knows which visual reference is driving
each surface.

Do not rename or overwrite the raw files while processing them. Treat `docs/ui/`
as provenance. Processed derivatives should be generated into stable product
paths such as:

```text
docs/ui/processed/home/
docs/ui/processed/skill/
docs/ui/processed/job-board/
docs/ui/processed/icons/
```

Expected processing:

```text
slice reusable frames, tabs, boxes, and icon cells
restore transparency where useful
normalize palette and contrast
produce monochrome and 8-bit/pixel variants for skill icons
preserve exact raw source filename in a manifest
```

### Shared State Across All Three Screens

All three screens must reuse the same underlying state. Do not create separate
mock states per screen.

Required shared model:

```ts
interface AccountSkill {
  id: string;
  accountRoot: 'email' | 'calendar' | 'notes' | 'crm' | 'calls' | 'code' | string;
  surface: string;
  displayName: string;
  provider?: string;
  iconKey?: string;
  level: number;
  status: 'locked' | 'available' | 'training' | 'trained' | 'equipped';
  evalScore?: number;
  contractPassRate?: number;
  oosPassRate?: number;
  lastTrainedAt?: string;
  allowedWrites: string[];
  recentFailures: DrillFailure[];
  prerequisites: string[];
}

interface EquippedChain {
  skills: string[];
  mode: 'single' | 'chain';
  currentSurface?: string;
}

interface DrillFailure {
  category: string;
  prompt: string;
  expected: string;
  actual: string;
}
```

This model lets the screens stay consistent:

```text
Home chooses and equips.
Skill trains and inspects.
Job Board compares and recommends.
```

### First-Run Review

Recommended first-run:

```text
1. User opens Home / Inventory.
2. Calendar is the primary visible skill.
3. Calendar shows Level 0 / untrained / can write: event, reminder, slot.
4. User selects Calendar and opens Skill / Train Surface.
5. User clicks Train Starter Skill.
6. Training runs on built-in Calendar drills.
7. Skill screen shows before/after eval and failed drill categories.
8. User equips Calendar.
9. User returns Home and types: "Schedule 30m with Sarah tomorrow afternoon."
10. Emberglass emits a verified dry-run plan.
11. Job Board shows Calendar as improved and lists next recommended drills.
```

Avoid in first-run:

- model selection;
- provider OAuth;
- broad app catalog browsing;
- World Map;
- live account mutation;
- long LoRA explanations;
- device-specific detours.

## Screen 1: Home / Inventory

### Consultant Read

Home / Inventory is the primary screen. It should not be a marketing page, a
model dashboard, or a generic app launcher. It is the user's RPG inventory for
trained account skills.

The screen must answer:

```text
What skills do I have?
Which skill or chain is equipped?
What account or browser surface am I currently on?
What can the equipped skill write?
What happens if I type a request now?
```

### Required Content

Home / Inventory must show:

1. Current surface:
   browser tab, account, provider, capture state.
2. Equipped chain:
   one to four skills, visually ordered.
3. Core account roots:
   Email, Calendar, Notes as the default roots.
4. Skill inventory:
   trained, available, locked, and recommended skills.
5. Selected skill summary:
   level, eval, allowed writes, status, next action.
6. Command and verified plan:
   prompt input, parsed macro, dry-run status.
7. Trust state:
   "dry run only" and "no account changed" should be visible.

### Game Design Improvements

Home / Inventory should feel like the player is looking at an equipped
skillbook, not a settings page.

Improve the screen this way:

```text
make the equipped skill the visual anchor
show "Reliable" or "Learning" before percentages
show one tempting next unlock below the current skill
make locked items explain their prerequisite and reward
rename the command area visually as Cast Request
turn the verified plan into a sealed result card
```

The player should feel:

```text
I have Calendar equipped.
I know what Calendar can do.
I see what training unlocks next.
I can try Calendar right now.
```

### Desktop Implementation

Desktop should use the browser app surface plus a persistent Emberglass panel.
The panel can be a web app shell now and a Chrome side panel later [S1]. The
important product point is not the exact container. It is that the user never
loses sight of the current account surface.

Desktop Home wireframe:

```text
+----------------------------- CURRENT APP / TAB -----------------------------+ +------------- HOME / INVENTORY *H1 ---------------+
| calendar.google.com                                                         | | Emberglass                                      |
| Week view, email thread, CRM record, etc.                                    | | surface: Calendar / Google                      |
|                                                                              | | dry run only                                    |
| The real app remains visible.                                                | |                                                   |
|                                                                              | | Equipped                                        |
|                                                                              | | [Calendar Lv1 Reliable]                         |
|                                                                              | |                                                   |
|                                                                              | | Core Skills *H1                                 |
|                                                                              | | [Email] [Calendar selected] [Notes]              |
|                                                                              | |                                                   |
|                                                                              | | Skill Inventory *H1                             |
|                                                                              | | trained: Calendar                               |
|                                                                              | | available: Gmail Inbox                          |
|                                                                              | | locked: Pipedrive, Gong                         |
|                                                                              | |                                                   |
|                                                                              | | Next Unlock                                     |
|                                                                              | | Gmail Inbox -> Calendar handoff                 |
|                                                                              | | requires: train Inbox starter                   |
|                                                                              | |                                                   |
|                                                                              | | Selected Skill *D1                              |
|                                                                              | | Calendar Lv1 / Reliable                         |
|                                                                              | | writes: create_event, find_slot, reminder *I1   |
|                                                                              | | eval: 78% -> 93% after starter drills           |
|                                                                              | | [Train] [Equip] [Open Skill]                    |
|                                                                              | |                                                   |
|                                                                              | | Cast Request                                    |
|                                                                              | | "Schedule 30m with Sarah tomorrow"              |
|                                                                              | |                                                   |
|                                                                              | | Plan Seal                                       |
|                                                                              | | create_event(...)                               |
|                                                                              | | contract passed / no account changed            |
+------------------------------------------------------------------------------+ +---------------------------------------------------+
```

Desktop rules:

- Keep the command visible.
- Keep dry-run status visible.
- Keep the selected skill's allowed writes visible.
- Do not show every possible app equally.
- Do not lead with a huge catalog.
- If the panel is narrow, collapse inventory details before collapsing command
  or trust state.

### Unfolded Foldable Implementation

Unfolded foldable should use a two-pane Home layout, but the screen is still
Home / Inventory. Do not call this a separate "command book" screen.

Foldable Home wireframe:

```text
+------------- HOME / INVENTORY: LEFT *H1 ---------------+ +---------- HOME / INVENTORY: RIGHT *D1 ------------------+
| Account Roots                                           | | Selected Skill                                          |
| [Email] [Calendar] [Notes]                              | | Calendar Lv1                                           |
|                                                         | | writes: event, reminder, slot *I1                      |
| Skill Inventory                                         | | state: Reliable / 93%                                  |
| [Calendar trained]                                      | | status: equipped                                       |
| [Gmail available]                                       | |                                                         |
| [Pipedrive locked]                                      | | Command                                                |
| [Gong locked]                                           | | "Find 30m tomorrow afternoon"                          |
|                                                         | |                                                         |
| Next Unlock                                             | | Plan Seal                                              |
| Gmail -> Calendar handoff                               | | find_slot(...) -> create_event(...)                     |
|                                                         | | contract passed / dry run only                         |
| Equipped Chain                                          | |                                                         |
| [Calendar]                                              | |                                                         |
+---------------------------------------------------------+ +---------------------------------------------------------+
```

Foldable Home rules:

- Left pane is inventory and account roots.
- Right pane is selected skill, command, and plan.
- Touch targets should be large enough for thumbs.
- Drag-and-drop is optional; tap-to-equip is required.
- If running as PWA split-screen with Chrome, show "captured via share /
  bookmarklet" in the current surface state.

### Mobile Implementation

Mobile Home is a compact inventory and command launcher. It cannot show the
full graph.

Mobile Home wireframe:

```text
+----------------------+
| Home / Inventory *H1 |
| Calendar Lv1 equipped|
| dry run only         |
+----------------------+
| Current capture      |
| calendar.google.com  |
| via share/bookmarklet|
+----------------------+
| Equipped             |
| [Calendar Reliable]  |
+----------------------+
| Core Skills          |
| [Email] [Calendar]   |
| [Notes]              |
+----------------------+
| Selected Skill *D1   |
| Calendar Lv1         |
| writes: event, slot  |
| icon ref *I1         |
| Reliable / 93%       |
| [Open Skill]         |
+----------------------+
| Next Unlock          |
| Gmail handoff        |
| train Inbox starter  |
+----------------------+
| Cast Request         |
| Schedule 30m...      |
| [Plan]               |
+----------------------+
| Plan Seal            |
| event at 2:00 PM     |
| contract passed      |
| dry run only         |
+----------------------+
```

Mobile Home rules:

- Show one equipped skill or chain summary.
- Show one selected skill card.
- Show one command input.
- Show one plan summary.
- Hide full inventory behind "View all skills."
- Do not hide the plan summary; if macro text is too long, summarize it.

### Home Acceptance Criteria

Home / Inventory is ready when:

```text
user can identify the equipped skill immediately
user can see what the skill can write before issuing a request
user can move to Skill / Train Surface from selected skill
user can issue a dry-run command without visiting another screen
user can understand no account changed
mobile still shows selected skill, command, and plan summary
foldable still maps to Home, not a new product screen
```

## Screen 2: Skill / Train Surface

### Consultant Read

Skill / Train Surface is the core loop. It is where the product proves that
training matters. If this screen is weak, the RPG metaphor becomes decorative.

This screen must answer:

```text
What exact account surface am I training?
What examples or drills are being used?
What writes will the trained skill be allowed to produce?
Did training improve the skill?
What should I do after training?
```

### Required Content

Skill / Train Surface must show:

1. Training target:
   account root, provider, surface, icon, capture/source.
2. Allowed write contract:
   exact macros/actions the skill may emit.
3. Training source:
   built-in drills, current tab capture, pasted examples, local examples.
4. Drill list:
   starter prompts, expected macros, pass/fail.
5. Training controls:
   starter pack, train button, stop button, and optional advanced settings.
6. Player-facing run progress:
   learned trials, progress, elapsed state, and an optional details view for
   deeper run metrics.
7. Before/after eval:
   base score, trained score, delta.
8. Completion actions:
   equip, add to chain, export adapter, open Job Board.

### Game Design Improvements

Skill / Train Surface is the most important game screen. It should feel like a
short, readable trial page where the player earns a stronger skill.

Improve the screen this way:

```text
lead with named trials, not knobs
show the reward before the player starts
make pass/fail cards flip or stamp visibly
show before/after as a reveal moment
turn each failure into a named next trial
make Equip Calendar the obvious reward button
hide raw charts behind a details drawer in the default player view
```

The player should feel:

```text
I know what Calendar is trying to learn.
I see the exact trials it must pass.
I pressed Train and watched weak trials improve.
I earned the right to equip this skill.
```

Player-facing trial examples:

```text
Simple Event
  "Schedule 30m tomorrow"
  reward: create clean event plan

Time Reading
  "tomorrow afternoon"
  reward: resolve vague time safely

Boundary Check
  "Email Sarah"
  reward: refuse non-calendar request

Open Slot
  "Find 30m Friday"
  reward: suggest available time
```

### Desktop Implementation

Desktop Skill / Train Surface should be a focused workbench. It can be opened
from Home as a route, overlay, or side-panel detail. It should not be hidden
inside a small drawer.

Desktop Skill wireframe:

```text
+--------------------------- SKILL / TRAIN SURFACE: CALENDAR *S1 -------------------------------+
| Target: Calendar / Google                         state: Learning -> Reliable                   |
| Reward: equip Calendar and unlock Gmail handoff progress                                        |
+-----------------------------+------------------------------+------------------------------------+
| Training Target             | Allowed Writes               | Before / After                     |
| Calendar                    | create_event(...)            | base contract pass: 61%            |
| provider: Google            | find_slot(...)               | trained pass:       93%            |
| surface: events             | set_reminder(...)            | OOS pass:           96%            |
|                             | rsvp(...)                    | delta:              +32 pts        |
+-----------------------------+------------------------------+------------------------------------+
| Trials *C1                                                                                 |
| [seal] Simple Event: Schedule 30m tomorrow -> learned create_event                          |
| [seal] Open Slot: Find open slot Friday -> learned find_slot                                |
| [weak] Boundary Check: "Email Sarah" -> next trial: reject non-calendar asks                |
+-------------------------------------------------------------------------------------------+
| Forge *P1                                                                                  |
| Starter Trial Pack: 18 examples / 4 trial types                                            |
| [Train Starter Skill] [Stop]                                                              |
| progress meter / learned trials / details                                                  |
+-------------------------------------------------------------------------------------------+
| Reward                                                                                    |
| [Equip Calendar] [Add After Gmail] [Claim Next Trial] [Open Job Board]                     |
+-------------------------------------------------------------------------------------------+
```

Desktop Skill rules:

- Put allowed writes above the fold.
- Put before/after eval above or beside the training controls.
- Show failed drills explicitly.
- Show OUT_OF_SCOPE behavior as a first-class quality metric.
- Do not bury training behind "advanced."
- In the default player view, show technical run details as "details," not as
  the emotional center of the page.

### Unfolded Foldable Implementation

Unfolded foldable should split Skill / Train Surface into source/contract on
the left and training/eval on the right.

Foldable Skill wireframe:

```text
+------------- SKILL: LEFT *S1 ---------------+ +------------- SKILL: RIGHT *P1 ---------------+
| Target                                    | | Trial Run                                    |
| Calendar / Google                         | | starter pack / 18 examples                  |
| reward: equip Calendar                     | | [Train Starter Skill]                       |
|                                            | |                                              |
| Allowed Writes *I1                        | | Progress                                     |
| create_event                              | | learned trials                               |
| find_slot                                 | | trial seals                                  |
| set_reminder                              | |                                              |
|                                            | | Eval                                         |
| Trials                                    | | Learning -> Reliable                         |
| Simple Event / Open Slot / Boundary       | | next trial: OOS email request                |
+--------------------------------------------+ +----------------------------------------------+
```

Foldable Skill rules:

- Keep target and allowed writes visible while training.
- Use the second pane for live run state and eval.
- Provide a persistent "Equip" action after training.
- Use large controls; training is a committed action.

### Mobile Implementation

Mobile Skill / Train Surface should become a stepper. Do not attempt to show the
whole training workbench at once.

Mobile Skill wireframe:

```text
+----------------------+
| Skill: Calendar *S1  |
| Learning -> Reliable |
+----------------------+
| Step 1: Target       |
| Google Calendar      |
| writes: event, slot  |
| icon ref *I1         |
+----------------------+
| Step 2: Trials *C1   |
| Simple Event         |
| Time Reading         |
| Boundary Check       |
| [Review Trials]      |
+----------------------+
| Step 3: Train *P1    |
| starter pack         |
| [Train Skill]        |
+----------------------+
| Step 4: Reward       |
| Reliable / 93%       |
| next: OOS email trial|
| [Equip] [Job Board]  |
+----------------------+
```

Mobile Skill rules:

- Use progressive disclosure.
- Default to starter settings.
- Show exact allowed writes before "Train Skill."
- Show a compact before/after score after training.
- Let the user equip immediately.
- Do not show raw loss charts unless expanded.

### Skill Acceptance Criteria

Skill / Train Surface is ready when:

```text
user knows what surface is being trained
user sees allowed writes before training
user can start starter training without docs
training progress is visible
before/after eval is visible
failed drill categories are visible
completion actions are obvious
mobile uses a stepper instead of a dense dashboard
foldable uses two panes but remains the same screen
```

## Screen 3: Job Board

### Consultant Read

Job Board is not a world map and not a leaderboard. It is the operational
screen that tells the user what is reliable, what failed, and what to train
next. It becomes useful once the user has more than one trained or trainable
surface.

This screen must answer:

```text
Which skills are reliable?
Which skill should I retrain?
Which failures are blocking useful chains?
Which skill should be equipped next?
What did the last run prove?
```

### Required Content

Job Board must show:

1. Skill rows:
   name, level, eval, contract pass, OOS pass, last trained.
2. Status:
   keep, train, drill, locked, needs review.
3. Recent failures:
   category and representative failure.
4. Recommended action:
   train more, view contract, equip after another skill, ignore.
5. Selected skill details:
   failed drills, allowed writes, recommended next drill pack.
6. Chain suggestions:
   safe adjacent skill handoffs.
7. Evidence:
   link back to eval samples, dry-run receipts, or training run.

### Game Design Improvements

Job Board should feel like the player's next set of meaningful tasks. The
desktop table is useful, but it should not be the first emotional read.

Improve the screen this way:

```text
show pinned task cards above dense comparison
make every task contain weakness, action, and reward
use stamps for Keep, Train, Drill, Locked
show locked chains as future promise
turn recent failures into named quests
make "Train Starter" feel like accepting a quest
```

The player should feel:

```text
I know what to improve next.
I know why it matters.
I can see what this unlocks.
My failures became useful tasks.
```

Good task-card shape:

```text
Inbox Starter Trial
  weakness: reply/calendar confusion
  action: train Gmail Inbox starter
  reward: unlock Gmail -> Calendar handoff
```

Bad task-card shape:

```text
Gmail Inbox
  eval 71%
```

### Desktop Implementation

Desktop Job Board should lead with pinned tasks and then offer dense comparison.
It should feel like a quest board inside the RPG shell, not an analytics page.

Desktop Job Board wireframe:

```text
+------------------------------------ JOB BOARD *J1 --------------------------------------+
| Pinned Tasks                                                                            |
| [Train Gmail Inbox] weakness: reply/calendar confusion -> unlock Gmail handoff          |
| [Drill Notes] weakness: title missing -> improve meeting notes                          |
+-----------------------------------------------------------------------------------------+
| Filter: [All] [Needs Training] [Equipped Chain] [Calendar] [CRM]                        |
+-----------------------------------------------------------------------------------------+
| Skill *I1        | State     | Proof | Boundary | Recent Failure | Suggested           |
+-----------------------------------------------------------------------------------------+
| Calendar         | Reliable  | 93%   | safe     | ambiguous time | keep equipped      |
| Gmail Inbox      | Learning  | 71%   | shaky    | reply scope    | train starter      |
| Notes            | Learning  | 68%   | safe     | title missing  | drill              |
| Pipedrive Deals  | Locked    | --    | --       | prereq Gmail   | unlock later       |
+-----------------------------------------------------------------------------------------+
| Selected: Gmail Inbox *D1                                                                |
| Quest: Inbox Starter Trial                                                               |
| Weakness: reply-vs-calendar confusion                                                    |
| Reward: unlock Email -> Calendar handoff                                                 |
| Allowed writes: draft_reply, label, archive, extract_meeting_request                     |
| [Train Starter] [View Contract] [Equip Before Calendar] [Open Skill]                     |
+-----------------------------------------------------------------------------------------+
```

Desktop Job Board rules:

- Use a table/list for comparison.
- Use pinned task cards to create motivation before comparison.
- Keep recommended action visible per row.
- Let the selected row open Skill / Train Surface.
- Do not gamify this as rank for rank's sake.
- Do not show fake mature scoring. If an eval is small, label it small.

### Unfolded Foldable Implementation

Unfolded foldable should show list on one pane and selected skill details on the
other.

Foldable Job Board wireframe:

```text
+------------- JOB BOARD: LIST *J1 --------------+ +----------- JOB BOARD: DETAIL *D1 -------------+
| Pinned: Train Gmail Inbox                     | | Quest: Inbox Starter Trial                    |
| Calendar      Reliable  keep                  | | weakness: reply/calendar confusion            |
| Gmail Inbox   Learning  train                 | | reward: Gmail -> Calendar handoff             |
| Notes         Learning  drill                 | | next trial pack: Email triage                 |
| Pipedrive     Locked    unlock later          | |                                               |
|                                                  | |                                               |
| Filters: needs training / equipped / locked      | | [Train] [Open Skill] [Equip Before Calendar]  |
+--------------------------------------------------+ +-----------------------------------------------+
```

Foldable Job Board rules:

- Keep comparison list visible while inspecting one skill.
- Use the detail pane for failed drills and recommended actions.
- Avoid modal stacks.
- Touch targets should support repeated triage.

### Mobile Implementation

Mobile Job Board should be a prioritized queue, not a table.

Mobile Job Board wireframe:

```text
+----------------------+
| Job Board *J1        |
| 3 skills need review |
+----------------------+
| Pinned Quest         |
| Inbox Starter Trial  |
| weakness: reply scope|
| reward: handoff      |
| [Train Starter]      |
+----------------------+
| Calendar             |
| Reliable / 93%       |
| status: keep equipped|
| [View]               |
+----------------------+
| Notes                |
| Learning / 68%       |
| quest: title missing |
| [Drill]              |
+----------------------+
| Locked               |
| Pipedrive            |
| needs Gmail Reliable |
+----------------------+
```

Mobile Job Board rules:

- Sort by recommended action, not alphabetically.
- Show one primary action per card.
- Keep metrics compact.
- Link to Skill / Train Surface for details.
- Do not show a dense multi-column table.

### Job Board Acceptance Criteria

Job Board is ready when:

```text
user can see what to train next
user can see why a skill is weak
user can jump to Skill / Train Surface
user can identify reliable equipped skills
user can see locked workflow skills without mistaking them for finished work
mobile becomes a prioritized queue
foldable becomes list + detail
desktop becomes comparison table + detail
```

## Cross-Screen Navigation

The three approved screens should form one loop:

```text
Home / Inventory
  select skill -> Skill / Train Surface
  issue command -> Verified dry-run plan
  open board -> Job Board

Skill / Train Surface
  train -> before/after eval
  equip -> Home / Inventory
  failed drills -> Job Board

Job Board
  choose weak skill -> Skill / Train Surface
  keep equipped skill -> Home / Inventory
  inspect failures -> Skill / Train Surface
```

Navigation rules:

- Home is the default route.
- Skill / Train Surface is reached from a selected skill.
- Job Board is reached from Home and from training results.
- World Map is not in the MVP navigation.
- Device-specific layouts should not introduce different product routes.

### Game Loop Navigation

The player should never feel like they are navigating software sections. They
should feel like they are cycling through a compact loop:

```text
Skillbook
  choose what is equipped

Trial Page
  improve one skill

Quest Board
  choose the next useful challenge

Skillbook
  equip the improved skill and cast again
```

Every transition should preserve motivation:

```text
Home -> Skill
  "Train Calendar to make this request reliable."

Skill -> Home
  "Calendar is now Reliable. Equip it."

Skill -> Job Board
  "One weakness remains. Claim the next trial."

Job Board -> Skill
  "This quest will unlock Gmail -> Calendar."

Job Board -> Home
  "Your equipped chain is good enough to use now."
```

The best version of this product has no dead menu clicks. Every screen exit is
either:

```text
try the equipped skill
train a weak skill
claim the next task
inspect a reward
```

## Form Factor Strategy Applied To Approved Screens

### Desktop

Desktop is the primary launch target. Use a Chrome side panel or side-panel-like
layout for persistent browser context [S1].

Desktop should support:

```text
Home / Inventory
  current app + side panel inventory + command + plan

Skill / Train Surface
  focused training workbench with target, writes, drills, telemetry, eval

Job Board
  dense comparison table plus selected detail pane
```

### Unfolded Foldable

Unfolded foldable is the secondary showcase target. It should use two panes for
each approved screen.

Foldable should support:

```text
Home / Inventory
  left: inventory/account roots
  right: selected skill/command/plan

Skill / Train Surface
  left: target/writes/drills
  right: training run/eval/completion actions

Job Board
  left: prioritized skill list
  right: selected skill detail/actions
```

If the user is on Android Chrome, do not assume normal extension support.
Support explicit PWA/bookmarklet/share capture instead [S6][S7].

### Mobile

Mobile is a companion surface. It should not attempt desktop parity.

Mobile should support:

```text
Home / Inventory
  compact equipped skill, capture, command, plan summary

Skill / Train Surface
  stepper: target -> drills -> train -> result

Job Board
  prioritized queue of weak/reliable/locked skills
```

Mobile capture paths:

```text
PWA paste
bookmarklet capture
share sheet
QR handoff from desktop
```

Mobile must never hide:

```text
current screen name
selected skill
primary action
dry-run status
plan or result summary
```

## Visual Direction Review

The UI reference assets added under `docs/ui/` should now be treated as direct
design references and internal source candidates for slicing, transparency
cleanup, palette normalization, and icon-processing experiments. Keep the raw
files untouched and create processed derivatives in `docs/ui/processed/`.

Before public distribution, verify rights for any asset that ships verbatim.
For internal prototypes and design implementation, use these exact source files
as the footnoted references below.

*H1 Home / Inventory frame and slot system:
[raw file](<docs/ui/Game Boy Advance - Pokemon FireRed _ LeafGreen - Menu Elements - PC Interface.png>).
Use for modular boxes, selected slot treatment, compact account roots, and
small inventory rhythm.

*S1 Skill / Train Surface page metaphor:
[raw file](<docs/ui/PC _ Computer - Heroes of Might and Magic 3 - Miscellaneous - Spellbook.png>).
Use for focused spellbook structure, page tabs, selected skill emphasis, and
the feeling of training one capability at a time.

*J1 Job Board queue/card system:
[raw file](<docs/ui/Wii - Fortune Street - Miscellaneous - Menu Boxes.png>).
Use for status rows, colored category bands, repeatable board cards, and dense
but readable list framing.

*I1 Skill/action icon system:
[raw file](<docs/ui/Sega CD - Shining Force CD - Miscellaneous - Weapon & Spell Icons.gif>).
Use as the strongest source for skill icons, allowed-write icons, lock states,
monochrome variants, and 8-bit/pixel treatment tests.

*D1 Selected detail and inspection panel:
[raw file](<docs/ui/PC _ Computer - Agatha Christie_ Murder on the Orient Express - Inventory - Interface.png>).
Use for selected skill detail, inspected object framing, and narrow detail
panels. Do not let the noir palette dominate the whole app.

*P1 Training progress and readiness meter:
[raw file](<docs/ui/Game Boy Advance - Flashback Legend (Prototype) - Miscellaneous - Screens (Faces).png>).
Use for compact empty-to-full progress/readiness meters and completion state.

*C1 Drill result callout / compact text box:
[raw file](<docs/ui/SNES - Dragon Ball Z_ Legend of the Super Saiyan (JPN) - Miscellaneous - Interface Icons and Text Box.png>).
Use for pass/fail drill callouts, small result banners, and scoped tutorial
prompts.

Recommended visual synthesis:

```text
Home / Inventory
  *H1 Pokemon PC slot clarity + *I1 processed skill icons + Emberglass identity.

Skill / Train Surface
  *S1 HoMM3 spellbook focus + *P1 progress meters + training/eval telemetry.

Job Board
  *J1 Fortune Street board cards + *D1 selected inspection panel + operational
  table/queue clarity.
```

The large skill graph reference is intentionally not in the wireframes:

[raw file](<docs/ui/DS _ DSi - SD Gundam G Generation DS - Miscellaneous - Unit Tree.png>)

It is conceptually relevant for account progression, but too dense for the core
three-screen product. If a simple skill tree is added inside Home / Inventory or
Skill / Train Surface, limit it to:

```text
one root account skill
two first branches
one locked advanced branch
three to seven visible nodes total
clear prerequisites
no giant zoomable graph in the MVP
```

Visual anti-patterns:

- shipping public builds with unverified third-party asset rights;
- overwriting raw reference files instead of producing processed derivatives;
- making the interface look like one source game;
- adding character equipment before account skills work;
- treating icons as decoration without status/capability data;
- making mobile a tiny desktop table.

## UI Implementation Priorities

Build in this order:

```text
1. Home / Inventory desktop
2. Skill / Train Surface desktop
3. Job Board desktop shell
4. Home / Inventory mobile
5. Skill / Train Surface mobile stepper
6. Job Board mobile queue
7. Unfolded foldable two-pane variants
```

Why this order:

- Desktop proves the full information architecture.
- Mobile forces the team to preserve the core loop under constraint.
- Foldable is then a natural two-pane adaptation, not a separate concept.

Do not build more visual worlds until these three screens are coherent.

## Engineering Architecture Review

The prior engineering architecture review remains relevant. The compressed
architecture below preserves its major decisions while removing repeated prose.

### Current Boundary

Current product:

```text
load model
  -> train skill
      -> equip skill
          -> generate typed macro / write plan
              -> verify plan against a contract
                  -> show dry-run output
```

Not current product:

```text
execute the write
mutate user accounts
drive DOM interactions
call provider APIs on the user's behalf
store provider credentials
autonomously click or submit forms
```

The model should be a planner/compiler, not an actor. It emits constrained
macros over typed ports. Deterministic code parses, verifies, resolves,
approves, and later executes.

### Existing Runtime Shape

Useful separations already exist:

```text
src/qwgpu/*
  WebGPU runtime, kernels, trainer, buffer pool, dispatch planning.

src/services/model_session.js
  WebGPU initialization, tokenizer, weight loading, streaming generation.

src/services/training_controller.js
  masked completion-only LoRA training against macro targets.

src/services/adapter_registry.js
  adapter registration and runtime hot-swap.

src/services/store.js
  local adapter metadata/blob persistence.

src/skills.js and src/skills/*
  ports, contracts, provider profiles, examples, evals, macro verification.
```

Preserve this rule:

```text
Core runtime should not import chrome APIs.
```

The engine should be portable across:

- normal web demo;
- side panel;
- future extension document;
- test harness.

### Bottom-to-Top Stack

```text
Browser hardware and APIs
  WebGPU, IndexedDB, extension APIs, content scripts, OAuth flows

Emberglass engine
  tokenizer, ModelSession, QwenWGPU runtime, LoRA trainer, adapter registry

Skill substrate
  ports, provider profiles, contracts, corpora, evals, lessons

Planner
  prompt -> model -> macro text -> parser -> contract verifier

Surface bridge
  active tab context, selected text, DOM summary, account/surface detection

Extension shell
  side panel UI, service worker broker, content script messaging

Future action layer
  executor registry, approvals, API/DOM writes, receipts, audit

Product UX
  Home / Inventory, Skill / Train Surface, Job Board, dry-run action pane
```

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

### Extension Direction

Build the desktop MVP as a Chrome side panel because the product is about the
app or tab the user is currently looking at. Chrome's Side Panel API hosts
extension content alongside the main webpage and supports persistent companion
experiences [S1].

Initial side-panel responsibilities:

- host Account Atlas / Train Surface UI;
- show model/adaptor readiness;
- show selected skill and equipped chain;
- read active tab URL/title/origin through extension messaging;
- ask a read-only content script for context;
- generate macro plans;
- verify macro plans;
- show dry-run plans.

Initial side-panel non-responsibilities:

- execute provider writes;
- submit DOM forms;
- hold OAuth refresh tokens;
- treat generated macros as trusted code.

The Manifest V3 service worker should be a broker, not the model host:

- route messages;
- manage extension events;
- broker permissions;
- later broker auth;
- wake up content scripts;
- coordinate side-panel opening.

For MVP, the model should run in the side panel. If background continuity is
needed later, evaluate an offscreen document or worker as a separate technical
spike.

### Extension Topology

Recommended:

```text
manifest.json
  sidePanel, scripting, storage, activeTab or scoped host permissions
  identity later, only when action/auth begins

service_worker.js
  extension broker, tab routing, side-panel routing, permission broker

sidepanel.html / sidepanel.js
  Account Atlas UI, WebGPU runtime, LoRA trainer, adapter store,
  plan generation, dry-run verification

content/current_surface.js
  read-only surface bridge, metadata, selected text, DOM summaries

future content/action_dom.js
  DOM executor, approved ActionPlan only, no provider tokens

future providers/*.js
  API executors, token requests through AuthBroker only
```

Message paths:

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
```

### Surface Bridge

The first extension-specific feature must be read-only context capture.
Chrome content scripts can read details of web pages, make changes, and pass
information to the parent extension [S2]. Emberglass should intentionally use
only the read/pass-information part in the first extension milestone.

Suggested `SurfaceSnapshot`:

```ts
interface SurfaceSnapshot {
  version: '0.1';
  capturedAt: string;
  tab: { id: number; url: string; origin: string; title: string };
  page: { appHint?: string; surfaceHint?: string; routeHint?: string };
  selection?: { text: string; length: number };
  focus?: {
    tagName: string;
    role?: string;
    ariaLabel?: string;
    placeholder?: string;
    valuePreview?: string;
  };
  domSummary: Array<{
    role?: string;
    tagName: string;
    textPreview?: string;
    ariaLabel?: string;
    stableSelectorHint?: string;
  }>;
  privacy: { redactionsApplied: string[]; maxChars: number };
}
```

Capture policy:

- user gesture first;
- visible origin;
- size limits;
- redaction;
- no cookies;
- no OAuth tokens;
- no localStorage/sessionStorage extraction;
- no hidden full-page scraping by default.

### Skill Blocks and Ports

The Inbox & Calendar block is the template.

Recommended skill block shape:

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

The canonical port is what the model targets:

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
JSON.

Contracts turn model output into pass/fail criteria:

- emitted calls must be known ops;
- emitted args must be valid for the op;
- date/time literals must be ISO format;
- calendar events must not be zero-length;
- slot search windows must satisfy `after < before`;
- out-of-scope requests must bounce cleanly.

Contracts should be executable without the model and without the provider.

Provider profiles translate canonical ops to provider concepts. They are not
executors yet.

### Planner and Dry-Run Layer

Current planner flow:

```text
user request
  -> selected/equipped skill
      -> prompt with skill system contract
          -> model emits macro text
              -> parser
                  -> contract verifier
                      -> dry-run plan
```

Before app actions, macro parsing should harden into a real parser:

- string literal parsing;
- number/boolean parsing;
- variable assignment;
- reference validation;
- no arbitrary expression evaluation;
- clear syntax errors.

Suggested future `VerifiedPlan`:

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
    issues: Array<{ severity: 'error' | 'warning'; code: string; message: string }>;
  };
  dryRun: Array<{
    callIndex: number;
    title: string;
    description: string;
    risk: 'none' | 'low' | 'medium' | 'high';
  }>;
}
```

This is the handoff point to the future action layer.

## Future App-Actions Layer

This should be a separate milestone and architecture review.

Future pipeline:

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

The action layer consumes a verified plan. It never consumes raw model output.

### ActionPlan

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
```

### ApprovalPacket

The user approves a deterministic plan, not vague natural language:

```ts
interface ApprovalPacket {
  actionPlanId: string;
  title: string;
  summary: string;
  accountLabel: string;
  origin?: string;
  steps: Array<{ id: string; preview: string; changes: string[]; risk: string }>;
  permissions: PermissionRequirement[];
  canEditBeforeRun: boolean;
  canRunPartially: boolean;
}
```

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

Without receipts, the system should not mutate accounts.

### API vs DOM Actions

Prefer API-first for structured writes:

- Calendar event creation;
- email drafts and replies;
- GitHub issues and PR comments;
- CRM people/deals/activities;
- labels/categories;
- notes/doc creation.

Use DOM actions when:

- no public API exists;
- the feature is UI-only;
- the user is on the exact screen;
- draft/fill behavior is preferable to direct commit.

DOM trust ladder:

```text
highlight
fill
draft
click-with-confirmation
submit
```

Start with highlight/fill/draft only.

DOM executor constraints:

- never receive provider tokens;
- never execute raw model output;
- only receive approved structured instructions;
- check origin and route;
- check target element still matches expected selector/role/text;
- fail closed when the page changes.

### Auth and Token Boundary

Provider tokens belong to the future action layer, not the runtime.

Rules:

- model never sees tokens;
- content scripts never receive provider tokens;
- adapters/training examples should not include tokens;
- OAuth state belongs to the extension auth broker;
- Chrome's Identity API is the appropriate Google OAuth broker when this layer
  exists [S4];
- short-lived tokens should prefer memory or `chrome.storage.session`;
- long-lived refresh tokens should be avoided unless explicitly justified.

Chrome's session storage is memory-backed while the extension is loaded and is
cleared on browser restart, extension reload, update, or disable; that makes it
a better default for sensitive transient state than persistent stores [S5].

## Evaluation, Benchmarks, and Performance

Evaluation should prove product usefulness, not just technical correctness.

### Current Measured Gates

Non-GPU checks run during this review:

```text
npm run test:skills
npm run test:ratchet
node test/_detect.mjs
node test/_plan.mjs
```

Measured state:

| Metric | Current |
|---|---:|
| Skills | 12 |
| Training examples | 555 |
| Valid macros | 516 |
| OUT_OF_SCOPE examples | 39 |
| Held-out eval examples | 17 |
| Contract-checked examples | 572 |
| Dock tiles | 34 |
| Forgeable dock tiles | 12 |
| Calendar providers | 3: google, microsoft, zoho |
| Provider macros checked | 252 |
| Ratchet status | PASS |
| Detector cases | 9 PASS |
| Dry-run plans | 45 |
| Dry-run steps/receipts | 69 |

Interpretation:

- The planning substrate is real enough to talk about.
- Provider portability is beginning to be measurable.
- The dry-run boundary is currently enforceable.
- Held-out eval is too small for product confidence and should grow.

### Current Browser Performance Snapshot

README-recorded Chrome Canary/WebGPU benchmark, June 25, 2026, 3B-shaped mock
weights:

| Measurement | Result |
|---|---:|
| Greedy decode @ ctx 128 | 114.3 tok/s |
| Greedy decode @ ctx 1,024 | 118.0 tok/s |
| Greedy decode @ ctx 4,096 | 101.3 tok/s |
| Greedy decode @ ctx 7,800 | 82.8 tok/s |
| GPU top-k sampling, topK=40 | 21.4 tok/s |
| Training step, rank 8, 17 active labels | 31.9 train tok/s |

Prefill latency:

| Prompt length | Latency |
|---:|---:|
| 64 tokens | 99.0 ms |
| 256 tokens | 240.0 ms |
| 1,024 tokens | 1,012.9 ms |
| 4,096 tokens | 6,148.1 ms |
| 8,192 tokens | 18,397.2 ms |

Product interpretation:

- Decode is strong enough for a technical demo.
- User-facing benchmark should not be only tok/s.
- Sampling path is much slower than greedy decode, so generation settings matter.
- First-run model readiness and training time may matter more than decode speed.
- The most important product metric is time-to-first-valid-trained-plan.

### Product Metrics That Matter

Primary metric:

```text
time to first valid trained Calendar dry-run
```

Supporting metrics:

| Metric | Why it matters |
|---|---|
| Time to model ready | first-run abandonment |
| Time to adapter live | training must feel feasible |
| Parseable macro rate | output must be machine-checkable |
| Contract pass rate | trust and correctness |
| OUT_OF_SCOPE pass rate | safety and boundary discipline |
| Base vs trained delta | proves training matters |
| Dry-run comprehension | user understands what would happen |
| Prompt-to-plan latency | feels interactive |

### Calendar Eval Recommendation

Calendar should get a serious eval suite before more surfaces are promoted.

Categories:

```text
date normalization
duration handling
slot finding
email-calendar chains
OUT_OF_SCOPE near misses
prompt-injection pressure
invalid API invention pressure
ambiguous natural language
```

Every eval report should show:

- base model score;
- trained adapter score;
- delta;
- per-category pass rate;
- top failures;
- representative macro samples;
- median prompt-to-plan latency.

The shareable graph:

```text
Base model contract pass rate -> trained Calendar Skill pass rate
```

### Benchmark Report Template

Each release should report:

```text
Environment
  browser, OS, GPU/CPU, memory, WebGPU features

Model
  base model, weight mode, adapter rank, context window

Load
  cold load seconds, warm load seconds, shader compile ms, cache state

Inference
  prefill ms: 64, 256, 1024, 4096, 8192
  greedy tok/s: ctx 128, 1024, 4096, 7800
  sampled tok/s
  LoRA active tok/s

Training
  starter skill train time
  train tok/s
  step ms, optimizer ms, final loss

Product Funnel
  time to model ready
  time to adapter live
  time to first valid dry-run
  first-run failure rate

Quality
  parseable macro rate
  contract pass rate
  OUT_OF_SCOPE pass rate
  held-out eval pass rate
  base vs trained delta
```

## Competitive Analysis

### Competitive Categories

Emberglass competes with:

```text
Browser-local inference engines
  WebLLM, Transformers.js, ONNX Runtime Web, MediaPipe/LiteRT [S9][S11].

High-performance WebGPU demos
  WebML Community Gemma/Qwen/Llama spaces and custom kernel demos [S8].

Built-in browser AI
  Chrome Prompt API / Gemini Nano [S12][S13].

AI browsers and browser agents
  Comet, Dia, Operator-style browser agents [S14][S15][S16][S17].

Workflow automation
  Zapier, Make, app-native automation, CRM/email/calendar assistants.
```

Emberglass should own:

```text
trainable local browser skills for app/account surfaces
```

### WebML Community Gemma WebGPU Kernels

Strengths:

- strong "wow" demo for in-browser inference;
- Hugging Face distribution;
- custom WebGPU kernel credibility;
- community attention.

Weakness relative to Emberglass:

- not centered on user-specific local training;
- not app/account-surface oriented;
- not about skill ownership;
- not about contract-verified write plans.

Implication:

Do not market Emberglass as "faster than them" without controlled benchmarks.
Market it as a different product shape:

```text
They prove browser inference can be fast.
Emberglass proves browser models can become user-owned app skills.
```

### WebLLM

Strengths:

- mature browser-local inference engine;
- WebGPU acceleration;
- OpenAI-compatible API;
- strong developer ecosystem [S9][S10].

Weakness relative to Emberglass:

- more infrastructure than product metaphor;
- not centered on account-surface training;
- not centered on skill ownership and verified plans.

Implication:

WebLLM is a serious infrastructure reference. Emberglass must win on product
specificity and memorability.

### Transformers.js / Hugging Face WebGPU

Strengths:

- huge model ecosystem;
- easy JavaScript developer adoption;
- WebGPU guide and demos;
- Hugging Face distribution [S11].

Weakness relative to Emberglass:

- generic model runtime;
- no native account-skill story;
- no core contract-verified planning UX.

Implication:

"Runs in browser" is becoming commodity. "Trains a private skill for this app"
is the wedge.

### Chrome Prompt API / Gemini Nano

Strengths:

- built into Chrome;
- local model story without us owning model download UX;
- extension-friendly;
- Chrome docs already discuss extension use cases like extracting event details
  from pages for calendar-entry workflows [S12][S13].

Weakness relative to Emberglass:

- browser/vendor controlled model;
- not user-owned skill training;
- not a skill progression product;
- not centered on contract-verified local adapters.

Implication:

Chrome built-in AI raises the bar. Local AI alone is not enough. Emberglass has
to win on personalization, skill ownership, and verification.

### AI Browsers and Agents

Strengths:

- broad promise: "do things for me";
- familiar assistant UX;
- strong distribution potential [S14][S15][S16][S17].

Weaknesses:

- trust and security concerns;
- prompt-injection risk;
- generic automation can be brittle;
- users may not know what will happen before it happens.

Emberglass opportunity:

```text
local skill
typed action space
contract verification
dry-run first
approval later
receipts for every action
```

That is the anti-mystery-agent story.

### Workflow Automation

Strengths:

- mature integrations;
- reliable API execution;
- existing business use.

Weakness relative to Emberglass:

- setup-heavy;
- not local/private;
- not trained from the user's current surface;
- feels like plumbing, not a personal skill.

Implication:

Do not start by competing on execution breadth. Start by making a personal
local skill feel easy and magical.

## Suggestions for Success

1. Pick one flagship skill.

Calendar is the best first wedge because everyone understands it, the dry-run
object is inspectable, and the contract is concrete.

2. Make before/after training visible.

Show drill score improvement and exactly what the skill learned:

```text
ISO times
end = start + duration
no invented ops
OUT_OF_SCOPE bounce
```

3. Keep dry-run as a feature.

Say:

```text
Verified plan ready. No account changed.
```

Do not frame dry-run as a missing capability.

4. Put benchmarks in product terms.

Lead with:

```text
time to train Calendar Skill
time to first verified plan
base-vs-trained contract lift
```

Keep tok/s as technical proof.

5. Make the first extension read-only.

The initial extension trust story:

```text
I can understand the current tab.
I can train from visible context.
I can produce verified plans.
I cannot secretly mutate accounts.
```

6. Avoid premature breadth.

The catalog can hint at future depth, but Calendar should feel finished before
12 skills are promoted equally.

7. Use "Account Skills" or "Browser Skills" as the category.

Make people say:

```text
I trained a Calendar skill in my browser.
```

That sentence is memorable.

## Recommended Planning Artifacts

Before more implementation, create:

1. Calendar first-run script.
2. Landing page copy with one-liner, proof points, and dry-run trust line.
3. Calendar eval spec with base-vs-trained scorecard.
4. Benchmark report template focused on time-to-first-valid-plan.
5. Home / Inventory wireframe with desktop, mobile, and unfolded foldable
   variants.
6. Skill / Train Surface wireframe with desktop, mobile, and unfolded foldable
   variants.
7. Job Board wireframe with desktop, mobile, and unfolded foldable variants.
8. SurfaceSnapshot privacy/redaction spec.
9. Future action-layer trust model.
10. Competitive positioning one-pager.

## Final Recommendation

The product is worth pursuing if the team commits to:

```text
Product wedge
  Calendar Skill first.

Marketing lens
  account skills, not LoRA demo.

Trust boundary
  dry-run first, no hidden actions.

Architecture
  side panel shell, read-only bridge, action layer later.

UI
  approved three-screen loop: Home / Inventory, Skill / Train Surface, Job
  Board. Device layouts are variants, not replacement screens.

Evaluation
  base-vs-trained contract lift, not only tok/s.

Competitive stance
  do not race pure inference demos; own trainable app skills.
```

The most important sentence:

```text
Train a private browser skill for the app you are using, equip it, and turn
requests into verified write plans.
```

If the first two minutes prove that sentence, the product is interesting enough
to earn real usage.

## Source Notes

External references used as context. Inline citations in this document use the
same IDs.

- [S1] Chrome side panel API:
  https://developer.chrome.com/docs/extensions/reference/api/sidePanel
  - Used for the recommendation to make the desktop extension shell a persistent
    browser companion.
- [S2] Chrome content scripts:
  https://developer.chrome.com/docs/extensions/develop/concepts/content-scripts
  - Used for the read-only current-tab capture recommendation and the bridge
    boundary between page DOM and extension.
- [S3] Chrome scripting API:
  https://developer.chrome.com/docs/extensions/reference/api/scripting
  - Used for the optional runtime-injection path behind `activeTab` and
    `scripting` permissions.
- [S4] Chrome identity API:
  https://developer.chrome.com/docs/extensions/reference/api/identity
  - Used for the future Google OAuth broker recommendation.
- [S5] Chrome storage API:
  https://developer.chrome.com/docs/extensions/reference/api/storage
  - Used for the recommendation to prefer `storage.session` for sensitive
    transient extension state.
- [S6] Chrome Web Store Help, extension install/manage:
  https://support.google.com/chrome_webstore/answer/2664769
  - Used for the mobile stance: Chrome's phone flow adds extensions to desktop,
    so the product should not promise normal mobile Chrome extension parity.
- [S7] Chromium extensions discussion, Chrome Android extension support:
  https://groups.google.com/a/chromium.org/g/chromium-extensions/c/LscNuM8AIaw
  - Historical support reference for the Chrome Android extension limitation.
- [S8] WebML Community Gemma 4 WebGPU Kernels Hugging Face Space:
  https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels
  - Used as the high-performance WebGPU demo comparison point.
- [S9] WebLLM documentation:
  https://webllm.mlc.ai/docs/
  - Used as the browser-local inference engine reference.
- [S10] WebLLM GitHub:
  https://github.com/mlc-ai/web-llm
  - Used as supporting context for WebLLM's developer ecosystem.
- [S11] Transformers.js WebGPU guide:
  https://huggingface.co/docs/transformers.js/en/guides/webgpu
  - Used as the Hugging Face/Transformers browser WebGPU comparison point.
- [S12] Chrome Prompt API:
  https://developer.chrome.com/docs/ai/prompt-api
  - Used for the Chrome built-in local language model comparison.
- [S13] Chrome built-in AI API status:
  https://developer.chrome.com/docs/ai/built-in-apis
  - Used for the Chrome built-in AI maturity and extension availability context.
- [S14] OpenAI Operator:
  https://openai.com/index/introducing-operator/
  - Used for the broad browser-agent comparison and trust/approval framing.
- [S15] OpenAI Computer-Using Agent:
  https://openai.com/index/computer-using-agent/
  - Used as context for GUI/browser action agents.
- [S16] Perplexity Comet:
  https://www.perplexity.ai/comet/
  - Used as an AI-browser comparison point.
- [S17] Dia Browser:
  https://www.diabrowser.com/
  - Used as an AI-browser comparison point.
