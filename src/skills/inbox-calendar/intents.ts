// intents.ts — the Inbox & Calendar block's recipes as DATA, not code.
//
// Each intent draws slots (in RNG order) from a ProviderProfile's vocab pools, picks a
// natural-language phrasing, and renders a spec-valid macro over the canonical PORT.
// Templates use ${slot}, ${slot.field}, and the computed ${end} (= start + duration).
// Swapping Google -> Outlook/Zoho is a profile swap; these intents stay put.
import type { Intent, Example } from '../types.ts';

export const INTENTS: Intent[] = [
  {
    n: 8,
    draw: ['person', 'topic'],
    phrasings: [
      'email ${person} about ${topic}',
      'ping ${person} about ${topic}',
      'shoot ${person} a quick note on ${topic}',
      'draft a message to ${person} re ${topic}',
    ],
    macro: 'compose_email(to="${person}", subject="${topic}", body="Quick note about ${topic} — let me know your thoughts.")',
  },
  {
    n: 7,
    draw: ['person', 'topic'],
    phrasings: [
      'find the email from ${person} about ${topic}',
      "pull up ${person}'s message on ${topic}",
      'search my inbox for ${topic} from ${person}',
    ],
    macro: 'find_email(query="from:${person} ${topic}")',
  },
  {
    n: 7,
    draw: ['person', 'topic', 'when'],
    phrasings: [
      "reply to ${person}'s email about ${topic} that I'll review it by ${when.nat}",
      "tell ${person} in the ${topic} thread I'll get back by ${when.nat}",
    ],
    macro: 't = find_email(query="from:${person} ${topic}")\nreply_email(thread=t, body="Thanks — I\'ll review this by ${when.iso}.")',
  },
  {
    n: 6,
    draw: ['person', 'topic'],
    phrasings: [
      'forward the ${topic} email to ${person}',
      'send ${person} the ${topic} thread for their records',
    ],
    macro: 't = find_email(query="${topic}")\nforward_email(thread=t, to="${person}", note="FYI — for your records.")',
  },
  {
    n: 6,
    draw: ['topic'],
    phrasings: [
      'archive the emails about ${topic}',
      'clear out the ${topic} threads',
      'archive everything about ${topic}',
    ],
    macro: 't = find_email(query="${topic}")\narchive_email(thread=t)',
  },
  {
    n: 6,
    draw: ['person', 'label'],
    phrasings: [
      'label ${person}\'s email as ${label}',
      'tag the message from ${person} ${label}',
      'mark ${person}\'s thread ${label}',
    ],
    macro: 't = find_email(query="from:${person}")\nlabel_email(thread=t, label="${label}")',
  },
  {
    n: 6,
    draw: ['person', 'topic', 'when'],
    phrasings: [
      'schedule a thank-you to ${person} for ${topic}, send it ${when.nat}',
      'queue a note to ${person} about ${topic} to go out ${when.nat}',
    ],
    macro: 'schedule_send(to="${person}", subject="Thank you", body="Thanks for ${topic}.", when="${when.iso}")',
  },
  {
    n: 9,
    draw: ['person', 'topic', 'when', 'dur'],
    phrasings: [
      'set up a ${dur}-minute meeting about ${topic} with ${person} ${when.nat}',
      'book ${dur} minutes with ${person} on ${topic} ${when.nat}',
      'put a ${dur}-min ${topic} sync with ${person} on my calendar ${when.nat}',
    ],
    macro: 'create_event(title="${topic} with ${person}", start="${when.iso}", end="${end}", remind_min=10)',
  },
  {
    n: 6,
    draw: ['topic', 'when'],
    phrasings: [
      'remind me to follow up on ${topic} ${when.nat}',
      'set a reminder about ${topic} for ${when.nat}',
    ],
    macro: 'set_reminder(text="Follow up on ${topic}", when="${when.iso}")',
  },
  {
    n: 8,
    draw: ['topic', 'window', 'dur'],
    phrasings: [
      'find a ${dur}-minute slot ${window.nat} and book ${topic}',
      'grab ${dur} minutes ${window.nat} for ${topic}',
    ],
    macro: 's = find_slot(duration_min=${dur}, after="${window.after}", before="${window.before}")\ncreate_event(title="${topic}", start=s.start, end=s.end, remind_min=10)',
  },
  {
    n: 7,
    draw: ['topic', 'rsvp'],
    phrasings: [
      '${rsvp.verb} the ${topic} invite',
      'respond ${rsvp.resp} to the ${topic} meeting invite',
    ],
    macro: 't = find_email(query="${topic} invite")\nrsvp(event=t, response="${rsvp.resp}")',
  },
];

// Out-of-scope probes (provider-independent): inbox-adjacent near-misses that must bounce.
export const OOS: Example[] = [
  'order me a pizza', 'what is the capital of France?', 'play some jazz', 'book me a flight to Tokyo',
  'summarize my entire inbox', 'translate this email to French', 'unsubscribe me from all newsletters', "what's the weather tomorrow?",
].map((q): Example => [q, 'OUT_OF_SCOPE']);
