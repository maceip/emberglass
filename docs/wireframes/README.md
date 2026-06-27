# Emberglass wireframes — the three-screen game loop

Mid-fidelity, **static** wireframes for the Emberglass "accounts as skills" app,
grounded in the game-design pass of the product/architecture review
(`architecture_review_including_app_actions.md`, UI/UX section from line 691).

> **Fidelity disclaimer.** Themed product wireframes, not pixel-perfect visual
> design and not the live app. They reuse the real Emberglass tokens and are
> skinned with processed retro-UI bitmaps (see *Assets* below), but layout,
> copy, and data are illustrative.

## What these are (and are not)

- **100% client-side / static.** No web server requirement beyond serving files,
  no `fetch`, no CDN, no account data, no OAuth, no API/DOM writes. Everything is
  a **dry-run** mock — "no account changed" is the whole point.
- They demonstrate the *shape* of the experience. They do **not** wire up the
  model, the planner, capture/extension code, or any backend.

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
(review "Cross-Screen Navigation"). Resize any page to see it adapt.

## Shared state

All three screens read one client-only module, `state.js`, which mirrors the
`AccountSkill` / `EquippedChain` / `DrillFailure` interfaces from the review (no
separate mock state per screen). `ui.js` holds the shared component builders
(state badges, skill tiles, equipped chain, plan seal, quest cards, forge meter)
so the game vocabulary stays consistent everywhere.

## Assets (processed retro-UI derivatives)

Raw reference sheets live in `docs/ui/` and are **never modified**. Reproducible
derivatives are generated into `docs/ui/processed/` by
`scripts/process_ui_assets.py` and catalogued in `docs/ui/processed/manifest.json`
(every derivative records its exact raw source filename + crop bbox + marker):

| Marker | Source game sheet | Used for |
| --- | --- | --- |
| `*I1` | Shining Force CD — Weapon & Spell Icons | skill + allowed-write icons — `@2x` pixel variant wired via `srcset`; `-mono` + `-locked` also produced (`-locked` used for locked skills) |
| `*S1` | Heroes of Might & Magic 3 — Spellbook | Skill / Train Surface page metaphor (CSS) |
| `*J1` | Fortune Street — Menu Boxes | Job Board status bands + detail panel (cover backgrounds) |
| `*P1` | Flashback Legend — Screens | forge/training progress meter track |
| `*C1` | Dragon Ball Z — Text Box | ornate reward/boundary callout (`border-image`, slice 6 = real border; interior text discarded) |
| `*H1` | Pokémon FireRed/LeafGreen — PC Interface | **provenance only** — boxes carry a header bar + wallpaper, so they don't 9-slice into a uniform ring; the heroic equipped slot is realized in CSS |
| `*D1` | Agatha Christie — Inventory Interface | **provenance only** (noir palette intentionally not skinned) |

The kept `*I1` icons retain the source spell-tile background (rounded corners are
keyed transparent); they are styled tiles, not glyphs matted onto full transparency.

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
