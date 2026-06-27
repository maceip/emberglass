// providers/zoho.ts — the Zoho (Mail + Calendar) ProviderProfile.
//
// Same canonical PORT again; Zoho splits across the Mail API and the Calendar API. Times are
// RFC3339-ish; recurrence follows the iCalendar RRULE convention. Curated subset — honest
// where Zoho's surface diverges (search params, scheduled send).
import type { ProviderProfile } from '../../types.ts';
import { CALENDAR_POOLS } from '../pools.ts';

export const ZOHO_PROFILE: ProviderProfile = {
  provider: 'zoho',
  label: 'Zoho (Mail + Calendar)',
  discovery: {
    source: [
      'https://www.zoho.com/mail/help/api/',
      'https://www.zoho.com/calendar/help/api/',
    ],
    revision: '2026-06-26-curated',
    note: 'Curated subset of the Zoho Mail + Calendar REST APIs. Macro emits canonical YYYY-MM-DDTHH:MM; the write-layer formats per endpoint. Scheduled send is limited.',
  },
  conventions: {
    timeFormat: 'RFC3339', // event start/end as ISO; Calendar API also accepts dateandtime
    searchSyntax: 'zoho-search', // searchKey + receivedFromAddress / subject params
    recurrence: 'RRULE', // iCalendar RRULE in the event payload
  },
  opMap: {
    find_email: 'GET /api/accounts/{id}/messages/search',
    compose_email: 'POST /api/accounts/{id}/messages',
    reply_email: 'POST /api/accounts/{id}/messages/{messageId}', // action: reply
    forward_email: 'POST /api/accounts/{id}/messages/{messageId}', // action: forward
    archive_email: 'PUT /api/accounts/{id}/updatemessage', // move to Archive folder
    label_email: 'POST /api/accounts/{id}/messages/{messageId}/tags',
    schedule_send: 'POST /api/accounts/{id}/messages', // isSchedule + scheduleType
    create_event: 'POST /api/calendars/{uid}/events',
    set_reminder: 'POST /api/calendars/{uid}/events', // event reminder block
    find_slot: 'GET /api/calendars/freebusy',
    rsvp: 'POST /api/calendars/{uid}/events/{eventId}', // attendee status update
  },
  pools: CALENDAR_POOLS,
};
