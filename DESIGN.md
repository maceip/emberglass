# Emberglass Design Direction

This document captures the current product and interface direction for the
in-browser LoRA training app.

The app is becoming a skill RPG, but the core metaphor is not a humanoid
character with gear. The durable object is an account/surface skillbook:
trained adapters are pages, nodes, or items that represent expertise over a
real account, app, or segmented app surface.

## Core Product Idea

Emberglass lets a user train VibeThinker-3B locally on a browser tab, app,
account, or DOM/API surface. The resulting LoRA adapter becomes an equippable
skill. Once equipped, the app can take a natural-language action request and
produce validated writes, such as API calls, DOM actions, or code/macros.

The important design correction:

```text
Not:
  abstract school -> generic skill

Instead:
  account / app surface -> specialized surface -> workflow handoff
```

Example ladder:

```text
Gmail account
  -> Pipedrive account
      -> Gong.io account
          -> Sales follow-up operator
```

For large apps, a node can be a segmented surface inside one account:

```text
Salesforce account
  - Leads
  - Opportunities
  - Accounts
  - Reports
  - Tasks

Slack workspace
  - Channels
  - DMs
  - Threads
  - Workflow posts

Gmail account
  - Inbox search
  - Labels
  - Compose
  - Scheduled send
```

The progression graph is rooted in accounts and surfaces, not broad categories.
Edges are workflow handoffs:

```text
Gmail -> Pipedrive
  email lead extraction -> CRM record/write

Pipedrive -> Gong
  deal/contact context -> call transcript/context

Gong -> Gmail
  call insight -> follow-up email

Gmail -> Calendar
  email intent -> meeting creation

GitHub -> Linear
  issue/PR context -> task tracking
```

## Design Principles

- The interface should feel like a frozen RPG menu, inventory, atlas, or
  spellbook, not a tabbed SaaS admin panel.
- The stable unit is the trained account/surface skill, not a character avatar.
- The avatar/mannequin idea can remain ambient, but it should not carry the
  product model until we know it adds meaning.
- The first screen should center loadout, equipped chain, selected surface, and
  action writes.
- Training starts from a selected account/surface node. It is not a separate
  generic Learn tab.
- Job Board and World Map are part of the product story and should be designed
  in lockstep, but they should remain lighter implementation investments until
  account/surface training works well.
- Icons should use real app/company SVGs when available, with uniform style
  modes such as brand, monochrome, and pixel/8-bit.

## Product Vocabulary

- Account: A connected app/account/workspace, such as Gmail, Pipedrive, Gong,
  Slack, Salesforce, or GitHub.
- Surface: A trainable part of an account, such as Gmail Labels, Pipedrive
  Deals, Salesforce Leads, Slack Threads, or Gong Calls.
- Node: A surface shown in the progression graph.
- Skill: A trained LoRA adapter for one account/surface node.
- Chain: An equipped sequence of skills across workflow handoffs.
- Write: A generated API call, DOM action, code action, or macro line.
- Drill: A validation/evaluation prompt for a trained surface.
- Job Board: A comparison and evaluation view for trained surfaces.
- World Map: A progression/exploration view for account/surface unlocks and
  workflow chains.

## Canonical Product Loop

```text
Open or connect account
  -> map/select a surface
      -> train the surface
          -> validate drills
              -> equip surface or chain
                  -> type action request
                      -> inspect generated writes
```

## Screen Delineation

The product should have one coherent design world with four main screens:

```text
1. Home / Atlas
   Account Atlas plus equipped workflow chain.
   This is the primary screen and should be real.

2. Train Surface
   Train the selected account/surface node.
   This is the core loop and should be real.

3. Job Board
   Compare/evaluate trained surfaces.
   This should be a designed shell with light real data first.

4. World Map
   Browse account graph and workflow unlocks.
   This should be a designed shell/progression preview first.
```

Implementation depth:

```text
Home / Atlas:     real
Train Surface:    real
Job Board:        real shell, light data
World Map:        real shell, mostly illustrative/progression preview
```

## 1. Home / Atlas: Account Atlas

This is the main screen. It replaces a generic Act/Learn split with a single
loadout and account progression interface.

```text
+-------------------- ACCOUNT ATLAS ---------------------+
| Emberglass        GPU ready / VibeThinker-3B / settings |
+---------------+--------------------------+-------------+
| ACCOUNTS      | SURFACE LADDER           | SELECTED    |
|               |                          | SURFACE     |
| [Gmail]       | Gmail Inbox              | Pipedrive   |
| [Pipedrive]   |   -> Lead extraction     | Deals       |
| [Gong]        |       -> Pipedrive Deals | Lv 2        |
| [GitHub]      |            -> Gong Calls | trained     |
| [Notion]      |                 -> Follow-up Writer     |
|               |                          | Actions     |
| Equipped chain|                          | create_deal |
| Gmail -> CRM  |                          | add_note    |
|               |                          | log_call    |
| [Edit Chain]  |                          | [Train]     |
| [World Map]   |                          | [Equip]     |
| [Job Board]   |                          | [Inspect]   |
+---------------+--------------------------+-------------+
| COMMAND: "turn promising Gong calls into Pipedrive tasks"|
| [Execute Writes]                                        |
+---------------- ACTION TRACE --------------------------+
| validated macro / API writes / DOM writes               |
+--------------------------------------------------------+
```

Home should answer:

- What accounts/surfaces exist?
- What has already been trained?
- What skill or chain is equipped?
- What can this chain write?
- What happened when the user issued a request?

## 2. Train Surface

Training is the core loop. It should be launched from the selected surface node.
It is not a separate top-level tab.

```text
+---------- TRAIN SURFACE: Pipedrive / Deals ------------+
| Source                                                   |
| [current tab] [connected account] [paste examples]       |
|                                                          |
| Surface map                                              |
| People   Organizations   Deals   Activities   Notes      |
|                                                          |
| Training target                                          |
| natural request -> typed Pipedrive write macro           |
|                                                          |
| Drills                                                   |
| "make a deal from this email"                            |
| create_person(...)                                       |
| create_deal(...)                                         |
| add_activity(...)                                        |
|                                                          |
| Forge                                                    |
| rank / steps / examples / local-private status           |
| [Begin Training]                                         |
|                                                          |
| Run state                                                |
| loss graph / train tok/s / active labels / validation    |
|                                                          |
| [Equip Surface] [Add to Chain] [Export Adapter]          |
+----------------------------------------------------------+
```

Training should show:

- The source: current tab, connected account, pasted examples, local files, or
  built-in drills.
- The exact surface being trained.
- The typed action space and example writes.
- Loss, training throughput, active labels, and validation.
- Clear completion actions: equip, add to chain, export.

## 3. Job Board: Designed Fast-Follow

The Job Board is a first-class screen in the story, but should not be overbuilt
until the training/evaluation loop is stronger.

Purpose:

- Compare trained surfaces.
- Show levels, validation, last trained time, and failure modes.
- Recommend retraining or chain placement.

```text
+-------------------- JOB BOARD --------------------+
| Compare trained account surfaces                   |
+--------------+--------+------------+--------------+
| Surface      | Level  | Validation | Last trained |
| Gmail Inbox  | Lv 4   | 92%        | today        |
| Pipedrive    | Lv 2   | 81%        | today        |
| Gong Calls   | Lv 1   | 64%        | yesterday    |
+--------------+--------+------------+--------------+
| Selected: Pipedrive Deals                          |
| strengths / failed drills / retrain suggestion     |
| [Retrain] [Equip] [Export] [Open in Atlas]          |
+----------------------------------------------------+
```

Implementation stance:

- Use current run metadata first.
- Do not fake mature scoring.
- Keep the visual design coherent with Home.
- Let the screen grow as validation gets better.

## 4. World Map: Designed Fast-Follow

The World Map is the progression/exploration screen. It should be designed in
the same visual language as Home, but can initially be derived from known
surfaces and static graph metadata.

Purpose:

- Show unlocked and locked account/surface nodes.
- Show workflow edges.
- Show equipped chains.
- Make the "RPG progression" story clear.

```text
+---------------- ACCOUNT WORLD MAP ----------------+
|                                                    |
| Gmail Inbox -----> Pipedrive Deals -----> Gong     |
|     |                  |                  |         |
|     v                  v                  v         |
| Calendar          CRM Tasks          Follow-up Mail |
|                                                    |
| Locked nodes are visible but dimmed                 |
| Trained nodes glow                                  |
| Equipped chain is highlighted                       |
+----------------------------------------------------+
| Selected edge: Gmail -> Pipedrive                   |
| Handoff: email lead extraction -> CRM write         |
| [Train missing node] [Equip this chain]             |
+----------------------------------------------------+
```

Implementation stance:

- It can start as a mostly visual/progression shell.
- It should still use real surfaces where available.
- It should not block core Home and Train Surface work.

## Earlier Concepts And How They Survive

### Equipment Screen

The first useful framing was an equipment/inventory screen:

```text
+-------------------- EMBERGLASS -------------------+
| GPU / model status                         settings |
+---------------+-----------------------+------------+
| INVENTORY     | EQUIPPED SKILL        | DETAILS    |
|               |                       |            |
| [Gmail] Lv3   |  [large app icon]     | ops: 11    |
| [GitHub] Lv2  |  Inbox & Calendar     | scope      |
| [Slack] locked|  action chips         | train stats|
|               |                       | export     |
| + Forge Skill |  "What do you want?"  | scrap      |
|               |  [ prompt input     ] |            |
|               |  [ Execute ]          |            |
+---------------+-----------------------+------------+
| ACTION TRACE / GENERATED API CALLS / VALIDATION     |
+-----------------------------------------------------+
```

This survives as the Home / Inventory screen, but the inventory is now rooted in
accounts/surfaces and workflow chains.

### MapleStory Wardrobe, Revised

The initial character-editor idea had the right energy, but the wrong object.
Instead of a humanoid avatar wearing gear, the object should be a spellbook or
atlas being filled with account/surface expertise.

```text
+-------------------- ACCOUNT SPELLBOOK -------------------+
|                                                          |
|      CORE ACCOUNT PAGES              SELECTED PAGE       |
|      [Gmail]                         Pipedrive / Deals   |
|      [Calendar]                      Lv 0 untrained      |
|      [Notes]                         Actions listed      |
|                                                          |
|  Specialized pages branch downward from real accounts     |
|                                                          |
+------------ SURFACE GRID -------------+------------------+
| [Gmail Inbox] [Pipedrive Deals]       | TRAINING PREVIEW |
| [Gong Calls]  [Salesforce Leads]      | drills + writes  |
| [Slack Threads] [locked] [locked]     |                  |
+---------------- COMMAND BAR ----------+------------------+
| "ask equipped chain to write/do..."       [Execute]       |
+----------------------------------------------------------+
```

This concept survives as the visual and emotional frame for Home and Train
Surface, without forcing a character/avatar model.

### Job Board

The Job Board remains a planned first-class screen for comparison and evaluation.
It should be designed with Home now, but implemented lightly first.

### Roguelike Map

The Roguelike Map remains a planned first-class progression screen. It should
show account graph unlocks and handoff paths, but should not consume heavy
implementation time until account/surface training works well.

## Shared Components To Build Toward

These components should be shared across Home, Train Surface, Job Board, and
World Map:

- AccountTile
- SurfaceNode
- EquippedChain
- TrainPanel
- ValidationBadge
- ActionTrace
- DrillPreview
- WritePreview
- SurfaceIcon

## Icon Direction

Skill/account icons should come from the vendored logo set when available. The
icon pipeline should support:

- brand SVG tiles
- uniform monochrome tiles, such as gold/cyan
- locked/disabled variants
- 8-bit/pixel variants

The icon style is part of the RPG presentation, but it should stay separate from
the training model. A skill should remain usable if the SVG is missing.

## Current Implementation Snapshot

The current application shell uses this naming:

```text
Account Atlas
  - left rail: trained surfaces plus Job Board / World Map shell buttons
  - stage: selected account surface and surface count
  - command pane: action request -> generated writes

Train Surface
  - overlay launched from the Atlas
  - account/app surface picker
  - drill examples, local training status, export
```

Some internal identifiers still use `skill` because the existing action-space
registry, tests, and CSS classes were already named that way. User-facing copy
should prefer account, surface, chain, write, drill, and Atlas.

## Current Priority

Build the core account/surface loop first:

```text
Home / Inventory
  -> Train Surface
      -> Equip Surface or Chain
          -> Execute Writes
              -> Inspect Action Trace
```

Keep Job Board and World Map designed in lockstep, but defer deeper
implementation until the underlying account/surface training path is reliable.
