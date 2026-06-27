// lessons.ts — a minimal family-keyed FoT (forest-of-thoughts) lessons store.
//
// Lessons are distilled, reusable insights about a skill family ("normalize times to ISO",
// "bounce inbox-adjacent near-misses"). Blocks SEED their lessons; future tuning runs can
// LEARN new ones. distill(family) renders them into compact guidance that can later be fed
// into the system-prompt context — the federated knowledge that compounds across skills.
import type { Lesson } from './types.ts';

const STORE = new Map<string, Map<string, Lesson>>();

// upsert by (family, id): a later 'learned' lesson can refine an earlier 'seed' of the same id
export function learn(lesson: Lesson): Lesson {
  let fam = STORE.get(lesson.family);
  if (!fam) { fam = new Map(); STORE.set(lesson.family, fam); }
  fam.set(lesson.id, lesson);
  return lesson;
}

export function lessonsFor(family: string): Lesson[] {
  return [...(STORE.get(family)?.values() ?? [])];
}

export function families(): string[] {
  return [...STORE.keys()];
}

// compact, prompt-ready guidance for a family (one bullet per lesson)
export function distill(family: string): string {
  return lessonsFor(family).map((l) => `\u2022 ${l.text}`).join('\n');
}
