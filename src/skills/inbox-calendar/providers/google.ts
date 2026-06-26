// providers/google.ts — the Google (Gmail + Calendar) ProviderProfile.
//
// This is the checked-in "Discovery snapshot" for the block: a curated subset of the Gmail
// and Google Calendar REST surfaces, capturing only what the adapter needs — the time
// convention (RFC3339), the Gmail search-query operators, and the canonical-op -> API-method
// map for the eventual write-layer. It is intentionally a trimmed, honest excerpt (not the
// multi-thousand-line raw discovery doc); the vocab pools are illustrative training fixtures.
import type { ProviderProfile } from '../../types.ts';

export const GOOGLE_PROFILE: ProviderProfile = {
  provider: 'google',
  label: 'Google (Gmail + Calendar)',
  discovery: {
    source: [
      'https://gmail.googleapis.com/$discovery/rest?version=v1',
      'https://www.googleapis.com/discovery/v1/apis/calendar/v3/rest',
    ],
    revision: '2026-06-26-curated',
    note: 'Curated subset. Times normalized to YYYY-MM-DDTHH:MM for the macro; Google uses RFC3339 (start.dateTime + timeZone). schedule_send/set_reminder have no clean public method — see opMap notes.',
  },
  conventions: {
    timeFormat: 'RFC3339', // events use start.dateTime/end.dateTime + timeZone; macro emits YYYY-MM-DDTHH:MM
    searchSyntax: 'gmail-q', // from:, subject:, label:, after:, before:, has:
  },
  // canonical PORT op -> Google Discovery method id (write-layer target; not emitted in macros)
  opMap: {
    find_email: 'gmail.users.messages.list', // q= search operators
    compose_email: 'gmail.users.messages.send',
    reply_email: 'gmail.users.messages.send', // threadId + In-Reply-To header
    forward_email: 'gmail.users.messages.send',
    archive_email: 'gmail.users.messages.modify', // removeLabelIds: [INBOX]
    label_email: 'gmail.users.messages.modify', // addLabelIds: [<labelId>]
    schedule_send: 'gmail.users.drafts.create', // no public scheduled-send method; client schedules the send
    create_event: 'calendar.events.insert',
    set_reminder: 'calendar.events.insert', // popup reminder override; Reminders API is not public
    find_slot: 'calendar.freebusy.query',
    rsvp: 'calendar.events.patch', // attendees[].responseStatus
  },
  pools: {
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
  },
};
