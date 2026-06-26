// generate.ts — the family-generic corpus generator.
//
// Given a seed, a ProviderProfile (vocab + conventions), the INTENTS, and the OOS probes,
// it deterministically renders a trainable corpus + a held-out eval split. This is the part
// that is shared across Google/Outlook/Zoho — only the profile changes.
import type { Example, GenResult, Intent, ProviderProfile, SlotName } from '../types.ts';

// deterministic RNG (seed pinned by the caller so the corpus is byte-stable across builds)
function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
function mulberry32(a: number): () => number {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// the one bit of time math the family owns: end = start + duration, ISO in / ISO out
function isoAdd(iso: string, mins: number): string {
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/)!;
  const t = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) + mins * 60000);
  const p = (n: number): string => String(n).padStart(2, '0');
  return `${t.getUTCFullYear()}-${p(t.getUTCMonth() + 1)}-${p(t.getUTCDate())}T${p(t.getUTCHours())}:${p(t.getUTCMinutes())}`;
}
function fill(tpl: string, ctx: Record<string, string>): string {
  return tpl.replace(/\$\{([^}]+)\}/g, (_, k) => (k in ctx ? ctx[k] : `\${${k}}`));
}

// maps a slot name to its pool on the profile
function poolFor(slot: SlotName, profile: ProviderProfile): readonly unknown[] {
  const p = profile.pools;
  switch (slot) {
    case 'person': return p.people;
    case 'topic': return p.topics;
    case 'when': return p.whens;
    case 'window': return p.windows;
    case 'label': return p.labels;
    case 'dur': return p.durations;
    case 'rsvp': return p.rsvps;
  }
}

export function generateCorpus(
  seed: string,
  profile: ProviderProfile,
  intents: Intent[],
  oos: Example[],
): GenResult {
  const rng = mulberry32(hashStr(seed));
  const pick = <T>(a: readonly T[]): T => a[Math.floor(rng() * a.length)];

  const makeOne = (intent: Intent): Example => {
    // draw slots in declared order (RNG order matters), then build the template context
    const raw: Record<string, any> = {};
    for (const slot of intent.draw) raw[slot] = pick(poolFor(slot, profile));
    const ctx: Record<string, string> = {};
    if ('person' in raw) ctx.person = raw.person;
    if ('topic' in raw) ctx.topic = raw.topic;
    if ('label' in raw) ctx.label = raw.label;
    if ('dur' in raw) ctx.dur = String(raw.dur);
    if ('when' in raw) { ctx['when.nat'] = raw.when.nat; ctx['when.iso'] = raw.when.iso; }
    if ('window' in raw) { ctx['window.nat'] = raw.window.nat; ctx['window.after'] = raw.window.after; ctx['window.before'] = raw.window.before; }
    if ('rsvp' in raw) { ctx['rsvp.resp'] = raw.rsvp.resp; ctx['rsvp.verb'] = raw.rsvp.verb; }
    if ('when' in raw && 'dur' in raw) ctx.end = isoAdd(raw.when.iso, raw.dur);
    // phrasing is picked AFTER the slot draws (preserves the original RNG sequence)
    const request = fill(pick(intent.phrasings), ctx);
    return [request, fill(intent.macro, ctx)];
  };

  const seen = new Set<string>();
  const all: Example[] = [];
  for (const intent of intents) {
    let made = 0, tries = 0;
    while (made < intent.n && tries < intent.n * 16) {
      tries++;
      const pair = makeOne(intent);
      if (seen.has(pair[0])) continue;
      seen.add(pair[0]); all.push(pair); made++;
    }
  }
  // deterministic split: every 5th in-scope pair is held out for eval
  const examples: Example[] = [], evals: Example[] = [];
  all.forEach((p, i) => ((i % 5 === 4) ? evals : examples).push(p));
  // OOS: every 4th held out for eval
  oos.forEach((q, i) => ((i % 4 === 3) ? evals : examples).push(q));
  return { examples, eval: evals };
}
