/* Emberglass wireframes — shared mock state.
 * One underlying model reused by Home / Inventory, Skill / Train Surface, and
 * Job Board (the review forbids separate mock states per screen). Client-only,
 * no build step, no network. Mirrors the AccountSkill / EquippedChain /
 * DrillFailure interfaces from architecture_review_including_app_actions.md.
 *
 * Asset references (processed derivatives of docs/ui/* retro sheets):
 *   icons  *I1  ../ui/processed/icons/<iconKey>.png
 *   frames *J1/*P1/*C1/*H1 under ../ui/processed/{job-board,skill,home}/
 */

export const RANK = ['Locked', 'Untrained', 'Learning', 'Reliable', 'Mastered', 'Rusty'];

/** Earned-state label from status + score (game-readable before numbers). */
export function stateLabel(skill) {
  if (skill.status === 'locked') return 'Locked';
  if (skill.status === 'training') return 'Learning';
  if (skill.status === 'available' && !skill.lastTrainedAt) return 'Untrained';
  const s = skill.evalScore ?? 0;
  if (s >= 97) return 'Mastered';
  if (s >= 90) return 'Reliable';
  if (s >= 70) return 'Learning';
  if (skill.lastTrainedAt) return 'Rusty';
  return 'Untrained';
}

export const STATE_HUE = {
  Locked: 'locked', Untrained: 'idle', Learning: 'learning',
  Reliable: 'reliable', Mastered: 'reliable', Rusty: 'review',
};

/** Skill icon url for a given iconKey (processed *I1 icon). */
export const iconUrl = (key) => `../ui/processed/icons/${key}.png`;
/** Provider brand logo (bundled SVG) for a given logo name. */
export const logoUrl = (name) => `logos/${name}.svg`;
/** Processed frame derivative. */
export const frameUrl = (rel) => `../ui/processed/${rel}`;

/** The account skills. accountRoot/surface/status mirror the review interface. */
export const SKILLS = [
  {
    id: 'calendar', accountRoot: 'calendar', surface: 'calendar.google.com',
    displayName: 'Calendar', provider: 'Google', iconKey: 'skill-calendar', logo: 'google-calendar',
    level: 1, status: 'equipped', evalScore: 93, contractPassRate: 93, oosPassRate: 96,
    baseScore: 61, lastTrainedAt: 'today',
    allowedWrites: ['create_event', 'find_slot', 'set_reminder', 'rsvp'],
    recentFailures: [{ category: 'ambiguous time', prompt: 'set it for later', expected: 'ask / resolve', actual: 'guessed 12:00' }],
    prerequisites: [], suggested: 'keep equipped',
  },
  {
    id: 'gmail', accountRoot: 'email', surface: 'mail.google.com',
    displayName: 'Gmail Inbox', provider: 'Google', iconKey: 'skill-email', logo: 'google-gmail',
    level: 1, status: 'available', evalScore: 71, contractPassRate: 71, oosPassRate: 64,
    baseScore: 52, lastTrainedAt: null,
    allowedWrites: ['draft_reply', 'label', 'archive', 'extract_meeting_request'],
    recentFailures: [{ category: 'reply/calendar confusion', prompt: 'reply to Sarah about the roadmap', expected: 'draft_reply', actual: 'create_event' }],
    prerequisites: [], suggested: 'train starter',
  },
  {
    id: 'notes', accountRoot: 'notes', surface: 'keep.google.com',
    displayName: 'Notes', provider: 'Google', iconKey: 'skill-notes', logo: 'google-keep',
    level: 1, status: 'available', evalScore: 68, contractPassRate: 68, oosPassRate: 70,
    baseScore: 50, lastTrainedAt: 'yesterday',
    allowedWrites: ['create_note', 'append_note', 'add_checklist'],
    recentFailures: [{ category: 'title missing', prompt: 'note from the sync', expected: 'titled note', actual: 'untitled blob' }],
    prerequisites: [], suggested: 'drill',
  },
  {
    id: 'pipedrive', accountRoot: 'crm', surface: 'app.pipedrive.com',
    displayName: 'Pipedrive Deals', provider: 'Pipedrive', iconKey: 'skill-crm', logo: 'pipedrive',
    level: 0, status: 'locked', evalScore: null, allowedWrites: ['update_deal', 'log_activity'],
    recentFailures: [], prerequisites: ['gmail'],
    unlocks: 'update deal after a call', requiresText: 'Gmail Inbox Reliable', suggested: 'unlock later',
  },
  {
    id: 'gong', accountRoot: 'calls', surface: 'app.gong.io',
    displayName: 'Gong Calls', provider: 'Gong', iconKey: 'skill-calls', logo: null,
    level: 0, status: 'locked', evalScore: null, allowedWrites: ['summarize_call', 'create_followup'],
    recentFailures: [], prerequisites: ['gmail', 'calendar'],
    unlocks: 'turn a call into follow-ups', requiresText: 'Gmail + Calendar Reliable', suggested: 'unlock later',
  },
  {
    id: 'github', accountRoot: 'code', surface: 'github.com',
    displayName: 'GitHub', provider: 'GitHub', iconKey: 'skill-code', logo: 'github',
    level: 0, status: 'locked', evalScore: null, allowedWrites: ['open_issue', 'comment', 'label_pr'],
    recentFailures: [], prerequisites: ['notes'],
    unlocks: 'file issues from notes', requiresText: 'Notes Reliable', suggested: 'unlock later',
  },
];

/** Equipped chain (single skill at first run; chain mode demonstrated too). */
export const EQUIPPED = { skills: ['calendar'], mode: 'single', currentSurface: 'calendar.google.com' };

/** Named trials for the Skill / Train Surface (challenge, not a config form). */
export const TRIALS = {
  calendar: [
    { id: 't1', name: 'Simple Event', prompt: 'Schedule 30m tomorrow', learns: 'create_event', reward: 'create a clean event plan', status: 'pass' },
    { id: 't2', name: 'Time Reading', prompt: '"tomorrow afternoon"', learns: 'resolve vague time', reward: 'resolve vague time safely', status: 'pass' },
    { id: 't3', name: 'Open Slot', prompt: 'Find 30m Friday', learns: 'find_slot', reward: 'suggest an available time', status: 'pass' },
    { id: 't4', name: 'Boundary Check', prompt: '"Email Sarah"', learns: 'refuse non-calendar request', reward: 'reject out-of-scope asks', status: 'weak' },
  ],
};

/** Job Board pinned quests (weakness + action + reward). */
export const PINNED = [
  { id: 'q-gmail', skill: 'gmail', title: 'Inbox Starter Trial', weakness: 'reply/calendar confusion', action: 'Train Gmail Inbox starter', reward: 'unlock Gmail → Calendar handoff' },
  { id: 'q-notes', skill: 'notes', title: 'Tidy Notes Drill', weakness: 'title missing', action: 'Drill Notes titling', reward: 'reliable meeting notes' },
];

/** The flagship cast: command -> verified dry-run plan (seal). */
export const CAST = {
  command: 'Schedule 30m with Sarah tomorrow afternoon',
  steps: [
    { op: 'create_event', risk: 'sensitive-write', args: 'title="Sarah sync", start="2026-06-30T14:00", end="2026-06-30T14:30", remind_min=10' },
  ],
  contractOk: true,
  capability: 'calendar:write',
};

/** The chain cast (Gmail -> Calendar handoff) for the chain demo. */
export const CHAIN_CAST = {
  command: "Turn Sarah's email into a 30m meeting tomorrow",
  steps: [
    { op: 'find_email', risk: 'read-only', args: 'from="Sarah", about="Q3 roadmap"' },
    { op: 'find_slot', risk: 'read-only', args: 'date="2026-06-30", window="afternoon", len_min=30' },
    { op: 'create_event', risk: 'sensitive-write', args: 'title="Sarah sync", start="2026-06-30T14:00", end="2026-06-30T14:30"' },
  ],
  contractOk: true,
  capability: 'mail:read + calendar:write',
};

export const skillById = (id) => SKILLS.find((s) => s.id === id);
export const equippedSkills = () => EQUIPPED.skills.map(skillById).filter(Boolean);
export const trainableSkills = () => SKILLS.filter((s) => s.status !== 'locked');
export const lockedSkills = () => SKILLS.filter((s) => s.status === 'locked');
