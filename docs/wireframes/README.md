# Emberglass form-factor wireframes

Mid-fidelity, **static** wireframes for the Emberglass "accounts as skills" app, built
from scratch off the product/marketing/architecture review
(`architecture_review_including_app_actions.md`).

> **Fidelity disclaimer.** These are themed product wireframes, not pixel-perfect
> visual design and not the live app. They reuse the real Emberglass tokens (from
> `docs/index.html` `:root`) so they read as Emberglass, but layout, copy, and
> spacing are illustrative.

## What these are (and are not)

- **100% client-side / static.** No web server requirement, no `fetch`, no external
  CDN, no account data, no OAuth, no API/DOM writes. Everything shown is a **dry-run**
  mock — "nothing sent" is the whole point.
- They demonstrate the *shape* of the experience across form factors. They do **not**
  wire up the model, the planner, capture/extension code, or any backend.

## Screens

| File | Form factor | What it shows | Review source |
| --- | --- | --- | --- |
| `index.html` | Gallery | Device frames previewing all three screens | — |
| `desktop.html` | Desktop | Browser tab + side-panel **skillbook** (top rail, equipped chain, current-surface card, command, verified plan) + inventory drawer state | lines 805, 880 |
| `foldable.html` | Foldable / tablet | Two-page **command book** (Atlas + chains / active skill + command + plan) + split-screen PWA + bookmarklet fallback | lines 912, 963 |
| `mobile.html` | Mobile phone | **Pocket skillbook** (one skill, one command, one plan, big copy/save) + capture paths (PWA paste / bookmarklet / share / QR handoff) | lines 989, 1037 |

Deferred fast-follows (not built here): Job Board (line 1111), World Map (line 1142).

## Responsive priority (hide-order)

Shared in `wireframes.css` (review lines 1194–1212). As space shrinks, surfaces drop
in this order via `.pri-*` classes:

1. coming-soon skills → 2. world map → 3. job board → 4. full inventory →
5. raw macro details → 6. eval telemetry

Surfaces marked `.keep` are **never** hidden: active skill, dry-run/action status,
command input, verification result. Resize any standalone page to see this in action.

These live under `docs/wireframes/` so they publish on GitHub Pages (which serves
from `main:/docs`). The handful of brand SVGs they use are bundled in
`docs/wireframes/logos/` so the screens are fully self-contained — no `vendor/`
dependency, no network. If a logo ever fails to load it falls back to an inline glyph.

## Viewing

- `npm run serve` (port 8013) → open `http://localhost:8013/docs/wireframes/`
- or open the `.html` files directly in a browser.
- Live (GitHub Pages): `https://maceip.github.io/emberglass/wireframes/`

## Smoke test

`npm run test:wireframes` spawns a local static server, loads each page headless
(Playwright), and asserts zero page errors plus the presence of the never-hide
surfaces (command, verified plan, dry-run status, equipped skill).
