// providers/outlook.ts — the Microsoft 365 (Outlook mail + Calendar) ProviderProfile.
//
// Same canonical PORT as Google; only conventions + the op->method executor map differ.
// Graph represents instants as { dateTime, timeZone } (not a bare RFC3339 string), mailbox
// search uses KQL via $search/$filter, and recurrence is Graph's patternedRecurrence object.
// Curated subset of the Microsoft Graph v1.0 surface — honest where a clean method is missing.
import type { ProviderProfile } from '../../types.ts';
import { CALENDAR_POOLS } from '../pools.ts';

export const OUTLOOK_PROFILE: ProviderProfile = {
  provider: 'microsoft',
  label: 'Microsoft 365 (Outlook + Calendar)',
  discovery: {
    source: [
      'https://learn.microsoft.com/en-us/graph/api/resources/mail-api-overview',
      'https://learn.microsoft.com/en-us/graph/api/resources/calendar',
    ],
    revision: '2026-06-26-curated',
    note: 'Curated subset of Graph v1.0. Macro emits canonical YYYY-MM-DDTHH:MM; the write-layer expands to { dateTime, timeZone }. Deferred/scheduled send has no first-class Graph method.',
  },
  conventions: {
    timeFormat: 'graph-dateTime', // { dateTime: "2026-06-29T17:00:00", timeZone: "..." }
    searchSyntax: 'kql', // $search="from:bob subject:x" / $filter
    recurrence: 'patternedRecurrence', // recurrence.pattern + recurrence.range (not raw RRULE)
  },
  opMap: {
    find_email: 'GET /me/messages?$search', // KQL search
    compose_email: 'POST /me/sendMail',
    reply_email: 'POST /me/messages/{id}/reply',
    forward_email: 'POST /me/messages/{id}/forward',
    archive_email: 'POST /me/messages/{id}/move', // destinationId: "archive"
    label_email: 'PATCH /me/messages/{id}', // categories: [<category>]
    schedule_send: 'POST /me/messages', // no first-class deferred send; client schedules
    create_event: 'POST /me/events',
    set_reminder: 'POST /me/events', // isReminderOn + reminderMinutesBeforeStart
    find_slot: 'POST /me/calendar/getSchedule', // or findMeetingTimes
    rsvp: 'POST /me/events/{id}/accept|tentativelyAccept|decline',
  },
  pools: CALENDAR_POOLS,
};
