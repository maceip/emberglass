/* FoT lessons gate.
 * Importing skills.js pulls the calendar block (index.ts), which registers its seed lessons
 * in the shared store. Asserts: calendar has its lessons, distill() is prompt-ready, and —
 * the key property — every calendar CONTRACT invariant id is explained by some lesson, so
 * what we ENFORCE and what we EXPLAIN never drift apart. */
import { SKILLS } from '../src/skills.js';
import { lessonsFor, distill, families } from '../src/skills/lessons.ts';

const fam = 'inbox-calendar';
const fails = [];
const ls = lessonsFor(fam);

if (ls.length < 5) fails.push(`calendar lessons too few: ${ls.length}`);
for (const l of ls) {
  if (!l.id || !l.text || l.family !== fam) fails.push(`malformed lesson: ${JSON.stringify(l)}`);
  if (l.origin !== 'seed' && l.origin !== 'learned') fails.push(`bad origin on ${l.id}`);
}
const d = distill(fam);
if (!d.includes('ISO 8601')) fails.push('distill() missing the ISO normalization lesson');
if (!/\u2022/.test(d)) fails.push('distill() not bullet-formatted');

// FoT <-> CONTRACT link: every invariant we enforce must have a lesson that explains it.
const cal = SKILLS.find((s) => s.key === fam);
const contractIds = [
  ...(cal.contract.assertions || []).map((a) => a.id),
  ...(cal.contract.forbidden || []).map((f) => f.id),
];
const explained = new Set(ls.map((l) => l.evidence));
for (const id of contractIds) {
  if (!explained.has(id)) fails.push(`contract invariant '${id}' has no explaining lesson`);
}

console.log(`families: ${families().join(', ') || '(none)'}`);
console.log(`${fam} lessons: ${ls.length}   contract invariants: ${contractIds.length}   explained: ${contractIds.filter((id) => explained.has(id)).length}/${contractIds.length}`);
console.log(distill(fam));

if (fails.length) { console.log('\nFAILURES:'); for (const f of fails) console.log('  - ' + f); }
console.log(fails.length ? 'LESSONS_FAIL' : 'LESSONS_PASS');
process.exit(fails.length ? 1 : 0);
