/* Data integrity test for the skills library (no browser, no model).
 * Asserts: (1) >= 500 ready-to-train examples across services,
 *          (2) every non-OOS macro PARSES and VALIDATES against its own spec,
 *          (3) every OOS example bounces with status "oos",
 *          (4) each skill carries a spec-derived system prompt and a suggest.
 * This is the "does what we say" gate applied to the whole training corpus. */
import { SKILLS, POPULAR_2026, verifyMacro, checkContract } from '../src/skills.js';
import { generateCorpus } from '../src/skills/inbox-calendar/generate.ts';
import { INTENTS, OOS } from '../src/skills/inbox-calendar/intents.ts';
import { PROVIDERS } from '../src/skills/inbox-calendar/providers/index.ts';

let total = 0, oos = 0, valid = 0;
const failures = [];
const perSkill = [];

for (const s of SKILLS) {
  let n = 0, sValid = 0, sOos = 0;
  if (!s.system || !/OUT_OF_SCOPE/.test(s.system)) failures.push(`${s.key}: missing/!spec system prompt`);
  if (!s.suggest) failures.push(`${s.key}: missing suggest`);
  for (const [req, macro] of s.examples) {
    total++; n++;
    const res = verifyMacro(macro, s.spec);
    if (macro === 'OUT_OF_SCOPE') {
      oos++; sOos++;
      if (res.status !== 'oos') failures.push(`${s.key}: OOS not bouncing → "${req}"`);
    } else {
      if (res.status === 'ok') { valid++; sValid++; }
      else failures.push(`${s.key}: invalid macro for "${req}" → ${res.status} ${JSON.stringify(res.issues)}`);
    }
    if (!req || typeof req !== 'string') failures.push(`${s.key}: empty request`);
  }
  perSkill.push(`${s.label.padEnd(18)} ${String(n).padStart(3)} ex  (${sValid} valid, ${sOos} oos)`);
}

// held-out eval split: spec-valid AND disjoint from the training set
let evalTotal = 0;
for (const s of SKILLS) {
  if (!Array.isArray(s.eval) || !s.eval.length) continue;
  const trainReqs = new Set(s.examples.map(([q]) => q));
  for (const [req, macro] of s.eval) {
    evalTotal++;
    if (trainReqs.has(req)) failures.push(`${s.key}: eval leaks into train → "${req}"`);
    const res = verifyMacro(macro, s.spec);
    const okEval = macro === 'OUT_OF_SCOPE' ? res.status === 'oos' : res.status === 'ok';
    if (!okEval) failures.push(`${s.key}: eval pair invalid → "${req}" (${res.status})`);
  }
}

// every emitted macro (train + held-out eval) must satisfy its skill's declarative CONTRACT
// — spec validity for all skills, plus family invariants (ISO times, non-zero events,
// ordered slot windows) for calendar. The contract is now the single source of truth.
let contractChecked = 0;
for (const s of SKILLS) {
  for (const [req, macro] of s.examples.concat(s.eval || [])) {
    contractChecked++;
    const v = checkContract(s.contract, macro, s.spec);
    if (!v.ok) failures.push(`${s.key}: contract [${v.violations.map((x) => x.id).join(', ')}] → "${req}"`);
  }
}
// flagship calendar must still ship a real held-out eval split
const cal = SKILLS.find((s) => s.key === 'inbox-calendar');
if (!cal.eval || cal.eval.length < 8) failures.push('calendar: missing/small held-out eval split');

// provider portability: the SAME canonical port + intents must (1) render contract-clean for
// EVERY calendar provider and (2) each provider must map every canonical op to an executor.
// This is the proof the port abstraction holds — only conventions + opMap differ per provider.
let providerChecks = 0;
const calOps = cal.spec.ops.map((o) => o.name);
for (const [id, profile] of Object.entries(PROVIDERS)) {
  const missing = calOps.filter((n) => !(n in profile.opMap));
  if (missing.length) failures.push(`provider ${id}: opMap missing [${missing.join(', ')}]`);
  const { examples: pex, eval: pev } = generateCorpus(`inbox-calendar:${id}`, profile, INTENTS, OOS);
  for (const [req, macro] of pex.concat(pev)) {
    providerChecks++;
    const v = checkContract(cal.contract, macro, cal.spec);
    if (!v.ok) failures.push(`provider ${id}: contract [${v.violations.map((x) => x.id).join(', ')}] → "${req}"`);
  }
}

// dock catalog sanity: every skill key appears in POPULAR_2026 with an icon
const dockKeys = new Set(POPULAR_2026.map((d) => d.key));
for (const s of SKILLS) if (!dockKeys.has(s.key)) failures.push(`dock: missing tile for skill ${s.key}`);
const dockBad = POPULAR_2026.filter((d) => !d.bg || !d.glyph);
if (dockBad.length) failures.push(`dock: ${dockBad.length} tiles missing bg/glyph`);

console.log(perSkill.join('\n'));
console.log('────────────────────────────────────────────');
console.log(`skills: ${SKILLS.length}   examples: ${total}   valid-macros: ${valid}   oos: ${oos}   held-out eval: ${evalTotal}   contract-checked: ${contractChecked}`);
console.log(`dock tiles: ${POPULAR_2026.length} (${POPULAR_2026.filter((d) => d.skill).length} forgeable, ${POPULAR_2026.filter((d) => !d.skill).length} locked)`);
console.log(`calendar providers: ${Object.keys(PROVIDERS).length} (${Object.keys(PROVIDERS).join(', ')})   provider-macros-checked: ${providerChecks}`);

const pass = total >= 500 && failures.length === 0 && (valid + oos === total);
if (failures.length) { console.log('\nFAILURES:'); for (const f of failures.slice(0, 25)) console.log('  - ' + f); }
console.log(pass ? 'SKILLS_DATA_PASS' : 'SKILLS_DATA_FAIL');
process.exit(pass ? 0 : 1);
