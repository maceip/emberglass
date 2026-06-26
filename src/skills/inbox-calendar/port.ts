// port.ts — the canonical, provider-agnostic action space for the Inbox & Calendar block.
//
// This is the PORT: the fixed vocabulary the model is allowed to emit, independent of
// whether the eventual write-layer talks to Google, Outlook/Graph, or Zoho. Adapters
// (per provider) generate corpora that target THIS port; the contract verifies against it.
import type { Op } from '../types.ts';

export const DOMAIN = 'an Inbox & Calendar operator';
export const SCOPE = 'inbox or calendar';

// Anchored "today" + the normalization rule, stated in the system prompt at train time.
// The live app can substitute the real current date later (the write-layer's job).
export const CONTEXT =
  'Assume today is Monday 2026-06-29, local time. Express every date and time as ISO 8601 ' +
  '(YYYY-MM-DDTHH:MM) and always set end = start + the requested duration.';

export const OPS: Op[] = [
  { name: 'find_email', params: ['query'], ret: 'thread' },
  { name: 'compose_email', params: ['to', 'subject', 'body'] },
  { name: 'reply_email', params: ['thread', 'body'] },
  { name: 'forward_email', params: ['thread', 'to', 'note'] },
  { name: 'archive_email', params: ['thread'] },
  { name: 'label_email', params: ['thread', 'label'] },
  { name: 'schedule_send', params: ['to', 'subject', 'body', 'when'] },
  { name: 'create_event', params: ['title', 'start', 'end', 'remind_min'] },
  { name: 'set_reminder', params: ['text', 'when'] },
  { name: 'find_slot', params: ['duration_min', 'after', 'before'], ret: 'slot' },
  { name: 'rsvp', params: ['event', 'response'] },
];

// Display + UX metadata for the block (the dock tile / equip panel).
export const META = {
  key: 'inbox-calendar',
  label: 'Inbox & Calendar',
  icon: '\u2709',
  desc: 'Compiles requests like \u201cemail my mom and book a reminder to respond\u201d into a verifiable macro over a fixed set of inbox/calendar actions; bounces anything else.',
  suggest: "Email the design team this week's notes, then put a 30-minute review on my calendar for Monday morning.",
};
