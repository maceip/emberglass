// pools.ts — the Inbox & Calendar training vocab, shared across providers.
//
// PROVENANCE: generated teaching examples — not real user/test-account eval evidence.
// User-facing claims must distinguish this corpus from provider eval or live account data.
//
// These fixtures describe the USER's world (who they email, what about, when), not any
// provider's API. So Google, Outlook, and Zoho all draw from the same pool — the provider
// profile only changes conventions + the op->method executor map, never this vocabulary.
import type { ProviderPools } from '../types.ts';

export const CALENDAR_POOLS: ProviderPools = {
  people: ['mom', 'Sarah', 'Alex', 'the design team', 'my manager', 'Priya', 'John', 'the landlord', 'accounting', 'Dana', 'Marcus', 'the recruiter'],
  topics: ['the Q3 roadmap', 'the launch', 'the budget', 'onboarding', 'the API redesign', 'the offsite', 'the bug report', 'the contract', 'the renewal', 'the demo'],
  // each "when" carries the natural phrasing (for the request) + its ISO start (for the macro)
  whens: [
    { nat: 'today at 5pm', iso: '2026-06-29T17:00' },
    { nat: 'tomorrow at 9am', iso: '2026-06-30T09:00' },
    { nat: 'Wednesday at 2pm', iso: '2026-07-01T14:00' },
    { nat: 'Thursday at 4:30pm', iso: '2026-07-02T16:30' },
    { nat: 'Friday at 11am', iso: '2026-07-03T11:00' },
    { nat: 'next Monday at 10am', iso: '2026-07-06T10:00' },
    { nat: 'tonight at 7pm', iso: '2026-06-29T19:00' },
  ],
  // search windows for find_slot — after STRICTLY before before
  windows: [
    { nat: 'tomorrow afternoon', after: '2026-06-30T13:00', before: '2026-06-30T18:00' },
    { nat: 'Wednesday morning', after: '2026-07-01T09:00', before: '2026-07-01T12:00' },
    { nat: 'Friday afternoon', after: '2026-07-03T13:00', before: '2026-07-03T17:00' },
    { nat: 'sometime Thursday', after: '2026-07-02T09:00', before: '2026-07-02T18:00' },
  ],
  labels: ['housing', 'urgent', 'finance', 'travel', 'follow-up', 'receipts'],
  durations: [30, 45, 60],
  rsvps: [
    { resp: 'yes', verb: 'rsvp yes to' },
    { resp: 'no', verb: 'decline' },
    { resp: 'maybe', verb: 'tentatively accept' },
  ],
};
