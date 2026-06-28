/* GOLD-TARGET calendar eval (L3 verification source of truth).
 *
 * Each case is: a prompt + the action we expect the trained skill to take +
 * machine-checkable assertions about that action. This is the "calendar write
 * target" — we verify the *intended* action matches the target and is contract-
 * clean. It does NOT execute against a real Google Calendar (that's L4, unbuilt);
 * a logged-out/dry-run system can still verify the plan with nothing but this file.
 *
 * verifyMacro() returns a per-check breakdown so the success rate is honest:
 * a case only PASSES if the expected op is present AND the contract holds AND the
 * target args match. Used by test/run_live_gpu.mjs against the real adapter.
 */

const ISO_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/;
const lines = (m) => String(m || '').split('\n');

/** Pull the first line that invokes `op(` and parse its key="value" args. */
export function findOp(macro, op) {
  for (const ln of lines(macro)) {
    const re = new RegExp(`\\b${op}\\s*\\(`);
    if (re.test(ln)) {
      const args = {};
      for (const m of ln.matchAll(/(\w+)\s*=\s*"([^"]*)"/g)) args[m[1]] = m[2];
      return { line: ln.trim(), args };
    }
  }
  return null;
}
export const hasOp = (macro, ops) => ops.some((op) => findOp(macro, op) !== null);

/** The calendar-family contract, evaluated in plain JS (mirrors contract.ts). */
export function contractHolds(macro) {
  const violations = [];
  for (const ln of lines(macro)) {
    for (const x of ln.matchAll(/(?:start|end|when|after|before)="([^"]+)"/g)) {
      if (!ISO_RE.test(x[1])) violations.push(`non-ISO time "${x[1]}"`);
    }
    const ce = ln.match(/create_event\(.*start="([^"]+)".*end="([^"]+)"/);
    if (ce && ce[1] === ce[2]) violations.push('zero-duration event');
    const fs = ln.match(/find_slot\(.*after="([^"]+)".*before="([^"]+)"/);
    if (fs && !(fs[1] < fs[2])) violations.push('unordered slot window');
  }
  return { ok: violations.length === 0, violations };
}

const minutesBetween = (a, b) => (Date.parse(b) - Date.parse(a)) / 60000;
const weekday = (iso) => new Date(iso + ':00').getDay(); // 0=Sun..6=Sat (local)
const hour = (iso) => (iso ? Number((String(iso).match(/T(\d{2}):/) || [])[1]) : NaN);

export const GOLD = [
  {
    id: 'notes-then-30m-review',
    prompt: "Email the design team this week's notes, then put a 30-minute review on my calendar for Monday morning.",
    expectOps: ['create_event'],
    target: (macro) => {
      const e = findOp(macro, 'create_event');
      if (!e) return { ok: false, why: 'no create_event' };
      const dur = minutesBetween(e.args.start, e.args.end);
      const checks = {
        duration_30m: dur === 30,
        monday: weekday(e.args.start) === 1,
        morning: hour(e.args.start) >= 6 && hour(e.args.start) < 12,
        also_emails: hasOp(macro, ['compose_email', 'schedule_send', 'draft_reply']),
      };
      return { ok: checks.duration_30m && checks.monday && checks.morning, checks, got: e.line };
    },
  },
  {
    id: '1on1-tue-2pm-45m',
    prompt: 'Schedule a 1:1 with Dana next Tuesday at 2pm for 45 minutes.',
    expectOps: ['create_event'],
    target: (macro) => {
      const e = findOp(macro, 'create_event');
      if (!e) return { ok: false, why: 'no create_event' };
      const checks = {
        duration_45m: minutesBetween(e.args.start, e.args.end) === 45,
        tuesday: weekday(e.args.start) === 2,
        at_14: hour(e.args.start) === 14,
        mentions_dana: /dana/i.test(e.line),
      };
      return { ok: checks.duration_45m && checks.tuesday && checks.at_14, checks, got: e.line };
    },
  },
  {
    id: 'free-slot-before-friday',
    prompt: 'Find me a free 30-minute slot before Friday for a design review.',
    expectOps: ['find_slot'],
    target: (macro) => {
      const f = findOp(macro, 'find_slot');
      if (!f) return { ok: false, why: 'no find_slot' };
      const checks = {
        ordered: f.args.after < f.args.before,
        before_friday: weekday(f.args.before) <= 5,
        thirty: !f.args.duration || /30/.test(f.args.duration),
      };
      return { ok: checks.ordered && checks.before_friday, checks, got: f.line };
    },
  },
  {
    id: 'reminder-tomorrow-9am',
    prompt: 'Remind me to send the invoice tomorrow at 9am.',
    expectOps: ['set_reminder', 'create_event'],
    target: (macro) => {
      const r = findOp(macro, 'set_reminder') || findOp(macro, 'create_event');
      if (!r) return { ok: false, why: 'no reminder/event' };
      const when = r.args.when || r.args.start || '';
      const checks = { iso: ISO_RE.test(when), at_9: hour(when) === 9, mentions_invoice: /invoice/i.test(r.line) };
      return { ok: checks.iso && checks.at_9, checks, got: r.line };
    },
  },
];

/** Full per-case verdict: op present + contract clean + target match. */
export function verifyMacro(macro, gold) {
  const opPresent = hasOp(macro, gold.expectOps);
  const contract = contractHolds(macro);
  const target = gold.target(macro);
  return {
    id: gold.id,
    pass: opPresent && contract.ok && target.ok,
    opPresent,
    contractOk: contract.ok,
    contractViolations: contract.violations,
    targetOk: target.ok,
    targetChecks: target.checks || {},
    got: target.got || '',
  };
}
