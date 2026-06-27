// providers/google.ts — the Google (Gmail + Calendar) ProviderProfile.
//
// This is the checked-in "Discovery snapshot" for the block: a curated subset of the Gmail
// and Google Calendar REST surfaces, capturing only what the adapter needs — the time
// convention (RFC3339), the Gmail search-query operators, and the canonical-op -> API-method
// map for the eventual write-layer. It is intentionally a trimmed, honest excerpt (not the
// multi-thousand-line raw discovery doc); the vocab pools are illustrative training fixtures.
import type { ProviderProfile } from '../../types.ts';
import { CALENDAR_POOLS } from '../pools.ts';

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
    recurrence: 'RRULE', // RFC5545 RRULE in event.recurrence[]
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
  pools: CALENDAR_POOLS,
};
