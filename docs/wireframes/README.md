# Emberglass wireframes — the three-screen game loop

Mid-fidelity wireframes for the Emberglass "accounts as skills" app, grounded in
the game-design pass of the product/architecture review
(`architecture_review_including_app_actions.md`, UI/UX section from line 691).

> **Fidelity disclaimer.** Themed product wireframes, not pixel-perfect visual
> design and not the live app. They reuse the real Emberglass tokens and are
> skinned with processed retro-UI bitmaps (see *Assets* below), but layout,
> copy, and data are illustrative.

## What these are (and are not)

- **100% client-side.** No web server requirement beyond serving files, no CDN,
  no OAuth, no API/DOM writes. Everything is a **dry-run** mock — "no account
  changed" is the whole point.
- They demonstrate the product loop. The checked-in Chrome extension now wires
  the side-panel form factor and read-only current-tab capture; these HTML
  screens remain the standalone design/product surface.

## The three screens (one game loop)

Read the product as a game: notice a weak/locked skill → train a short trial →
watch it improve → equip it → cast a request → receive a new task.

| File | Player feel | Purpose | Dominant verb | Review |
| --- | --- | --- | --- | --- |
| `home.html` | **Skillbook** | Home / Inventory — see and equip trained account skills, cast a request | Equip / Cast | line 1056 |
| `skill.html` | **Trial Page / Forge** | Skill / Train Surface — make one skill stronger through named trials, see before/after | Train | line 1252 |
| `job-board.html` | **Quest Board** | Job Board — what is reliable, what failed, what to train next | Claim | line 1497 |
| `index.html` | Gallery | Frames the three screens at desktop / foldable / mobile widths + loop nav | — | — |

Each screen is **one responsive layout** (desktop → foldable two-pane → mobile
stepper/queue); device form factors do **not** introduce new product routes
(review "Cross-Screen Navigation"). Resize any page to see it adapt:

- **Foldable / tablet (~720–1080px, the 920px gallery frame):** Home keeps the
  app surface beside the skillbook; Skill drops to a two-pane workbench (the
  before/after panel spans full width); Board keeps table + detail side-by-side.
- **Mobile (≤700px):** Home becomes the pocket skillbook (the app surface is
  dropped, equipped skill + one command + one plan stay); **Skill's trials become
  a numbered vertical stepper**; **Board swaps the comparison table for a
  priority-ordered card queue** (train-me-next first, locked promises last) — not
  a column-hidden table.

## Interactivity (these are clickable, not just pictures)

The review's strongest note was that wireframes that don't *do* anything can't be
judged. So the loop is wired, client-only:

- **Skill / Train:** **Train Starter Skill** runs the forge — the meter fills, the
  weak *Boundary Check* trial flips to **pass**, the state badge climbs
  Reliable → **Mastered (97%)**, the before/after updates, and **Equip** unlocks.
- **Home:** click any unlocked tile to **inspect** it (the selected card + allowed
  writes update); **Cast** runs a short "verifying…" beat then stamps a verified,
  contract-passed dry-run **plan seal**; **Equip** moves the skill into the heroic
  slot and re-points the cast.
- **Carry-over:** Equipping (on either screen) writes to `sessionStorage`, so the
  skill you equip on the Trial Page is the one equipped when you land on Home.

It is still all **dry-run** — no account, network, model, or DOM write happens.

## Chrome side panel

**implementation_required** — extension code was removed pending architecture approval.
Design only: `docs/extension-architecture-design.md`. No `extension/` directory or
`npm run test:extension` until approved.

## Shared state

All three screens read one client-only module, `state.js`, which mirrors the
`AccountSkill` / `EquippedChain` / `DrillFailure` interfaces from the review (no
separate mock state per screen). `ui.js` holds the shared component builders
(state badges, skill tiles, equipped chain, plan seal, quest cards, forge meter)
so the game vocabulary stays consistent everywhere. Earned state is derived in one
place (`stateLabel`): no score ⇒ **Untrained** (no number shown), `<90` Learning,
90–96 Reliable, 97+ Mastered, and **Rusty only via an explicit decay flag** (never
guessed from a score — so there are no "Untrained 71%" contradictions).

## Assets (processed retro-UI derivatives)

Raw reference sheets live in `docs/ui/` and are **never modified**. Reproducible
derivatives are generated into `docs/ui/processed/` by
`scripts/process_ui_assets.py` and catalogued in `docs/ui/processed/manifest.json`
(every derivative records its exact raw source filename + crop bbox + marker):

| Marker | Source game sheet | Used for |
| --- | --- | --- |
| `*I1` | Shining Force CD — Weapon & Spell Icons | skill + allowed-write icons — `@2x` pixel variant wired via `srcset`; `-locked` variant for locked skills |
| `*S1` | Heroes of Might & Magic 3 — Spellbook | Skill / Train Surface page metaphor (CSS) |
| `*J1` | Fortune Street — Menu Boxes | Job Board status bands + detail panel (cover backgrounds) |
| `*P1` | Flashback Legend — Screens | forge/training progress meter track |
| `*C1` | Dragon Ball Z — Text Box | ornate reward/boundary callout (`border-image`, slice 6 = real border; interior text discarded) |
| `*H1` | Pokémon FireRed/LeafGreen — PC Interface | **provenance only** — boxes carry a header bar + wallpaper, so they don't 9-slice into a uniform ring; the heroic equipped slot is realized in CSS |
| `*D1` | Agatha Christie — Inventory Interface | **provenance only** (noir palette intentionally not skinned) |

**Honest note on the `*I1` icons.** Shining Force is a *fantasy* spell sheet, so
no tile is a literal "draft a reply" or "create event" glyph. What we *can*
guarantee — and now do — is that **every on-screen glyph is distinct**: each of the
six account skills and each of the eleven allowed-writes that actually render
(calendar / gmail / notes) gets its own tile, so the "allowed writes" row never
shows the same glyph twice and a skill never collides with its own action. Real-
world meaning is carried by (a) the **brand SVG** overlaid on each skill tile
(Google Calendar, Gmail, Keep, GitHub, Pipedrive…) and (b) the write's text label.
We only generate the icons the screens reference — the unused fantasy tiles and the
`-mono` silhouettes (produced earlier but never themed) were **dropped** rather than
shipped as dead weight. The kept tiles retain the source spell-tile background
(corners keyed transparent); they're styled tiles, not glyphs matted onto full
transparency.

Regenerate: `python3 scripts/process_ui_assets.py all` (Pillow). The processed
bitmaps publish under `docs/`, so the screens are self-contained — no `vendor/`
or network dependency. Brand SVGs are bundled in `docs/wireframes/logos/`.

## Viewing

- `npm run serve` (port 8013) → `http://localhost:8013/docs/wireframes/`
- Live (GitHub Pages, served from `main:/docs`):
  `https://maceip.github.io/emberglass/wireframes/`

## Smoke test

`npm run test:wireframes` spawns a local static server, loads each screen headless
(Playwright), and asserts: zero page errors, the never-hide surfaces (command /
primary action, dry-run trust line, dominant-verb surface, rendered `*I1` icons),
the `*H1`/`*S1`/`*J1` markers are present, and no broken processed/brand assets.

UI evidence screenshots: `npm run evidence:ui` → `docs/evidence/ui/manifest.json`.
