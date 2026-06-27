// inbox-calendar/lessons.ts — seed FoT lessons for the calendar family.
//
// Each lesson's `evidence` points at the contract id (or signal) it explains, so every
// invariant we ENFORCE has a human-readable reason that travels with the block. Importing
// this module registers the lessons in the shared store (side effect, run via the block index).
import type { Lesson } from '../types.ts';
import { learn } from '../lessons.ts';

export const CALENDAR_LESSONS: Lesson[] = [
  {
    id: 'spec-valid', family: 'inbox-calendar', origin: 'seed', evidence: 'spec-valid',
    text: 'Emit only canonical port ops with their exact params, or bounce with OUT_OF_SCOPE — never invent an API.',
  },
  {
    id: 'iso-times', family: 'inbox-calendar', origin: 'seed', evidence: 'iso-times',
    text: 'Normalize every date/time to ISO 8601 (YYYY-MM-DDTHH:MM); never echo natural-language time into the macro.',
  },
  {
    id: 'duration', family: 'inbox-calendar', origin: 'seed', evidence: 'zero-duration-event',
    text: 'Always set end = start + the requested duration; a calendar event is never zero-length.',
  },
  {
    id: 'ordered-window', family: 'inbox-calendar', origin: 'seed', evidence: 'unordered-slot-window',
    text: 'find_slot search windows must satisfy after < before.',
  },
  {
    id: 'bounce-near-miss', family: 'inbox-calendar', origin: 'seed', evidence: 'oos',
    text: 'Bounce inbox-adjacent near-misses (summarize my inbox, translate this email, unsubscribe) — scope is compile-to-macro, not content tasks.',
  },
  {
    id: 'canonical-not-provider', family: 'inbox-calendar', origin: 'learned', evidence: 'more-providers',
    text: 'Target the canonical port, not a provider dialect; the ProviderProfile opMap translates ops to Google/Graph/Zoho at execution time.',
  },
];

CALENDAR_LESSONS.forEach(learn);
