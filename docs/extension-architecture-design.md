# Chrome Extension / Side-Panel Architecture Design

**Status:** design-only (Saturday review gate). No extension code until this is approved.

## Purpose

Read-only capture of the **current tab** to ground skill training and planning. The extension is a companion to the Emberglass web app, not a detached dashboard.

## Form factor

- **MV3 extension** with a **side panel** (not a popup-only or full-page replacement).
- **Service worker** for lifecycle, messaging, and permission prompts.
- **Content script** injected only on user-triggered capture (never passive always-on scraping).

## Hard boundaries

1. **Read-only current-tab capture only** — no writes from the extension.
2. **User-triggered** — capture starts from an explicit user gesture in the side panel.
3. **No provider tokens in prompts, content scripts, or logs** — OAuth stays in the web app / backend broker; extension receives only sanitized surface snapshots.
4. **Side panel as companion** — training and planning UI remain in `docs/product` / main web app; extension supplies context packets.
5. **No extension code in repo** until architecture approval (prior prototype removed per Saturday review).

## Message flow

```text
User clicks "Capture tab" in side panel
  → service worker checks host permission for active tab origin
  → content script extracts DOM snapshot / metadata (provider detect via detect.ts origin map)
  → sanitized CapturePacket → web app via externally_connectable / native messaging bridge
  → web app stores capture as training context (provenance: user_tab_capture)
```

## CapturePacket (sketch)

```ts
interface CapturePacket {
  capturedAt: string;
  origin: string;
  providerHint: string | null;  // from inbox-calendar/detect.ts
  title: string;
  url: string;                  // path only if sensitive query params stripped
  surfaceFingerprint: string;   // hash of stable DOM landmarks
  excerpt: string;              // bounded text, no tokens
}
```

## Security

- Host permissions are **optional per origin**, requested at capture time when possible.
- Content script bundle contains **no secrets**, no HF tokens, no Google refresh tokens.
- Captures are **ephemeral by default** unless user saves to a skill training session.

## Relationship to app-action layer

Extension provides **read context** only. Writes flow through the web app action layer (`docs/app-action-layer-design.md`) after ApprovalPacket.

## Current state

`extension/` directory absent. `src/skills/inbox-calendar/detect.ts` holds origin→provider mapping for future bridge. Marked `implementation_required` in product UI.
