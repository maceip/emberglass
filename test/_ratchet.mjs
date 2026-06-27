/* Ratchet gate (substrate-style protected baseline).
 *
 * Measures the metrics we can compute without a trained model — corpus integrity and
 * provider portability — and asserts each meets its checked-in floor in
 * test/baselines/ratchet.json. Floors are higher-is-better minimums and are TIGHTEN-ONLY:
 *   node test/_ratchet.mjs          # enforce: every measured metric >= its floor
 *   node test/_ratchet.mjs --bump   # ratchet up: floor = max(floor, measured), then rewrite
 * The mechanism is designed so trained held-out accuracy floors slot in unchanged later. */
import { readFileSync, writeFileSync } from 'node:fs';
import { SKILLS, verifyMacro, checkContract, planFor, EXECUTORS } from '../src/skills.js';
import { generateCorpus } from '../src/skills/inbox-calendar/generate.ts';
import { INTENTS, OOS } from '../src/skills/inbox-calendar/intents.ts';
import { PROVIDERS } from '../src/skills/inbox-calendar/providers/index.ts';

const BASELINE_URL = new URL('./baselines/ratchet.json', import.meta.url);
const baseline = JSON.parse(readFileSync(BASELINE_URL, 'utf8'));
const bump = process.argv.includes('--bump');

function measure() {
  let total = 0, valid = 0, oos = 0, evalTotal = 0, contractChecked = 0, contractClean = 0;
  for (const s of SKILLS) {
    for (const [, macro] of s.examples) {
      total++;
      if (macro === 'OUT_OF_SCOPE') oos++;
      else if (verifyMacro(macro, s.spec).status === 'ok') valid++;
    }
    evalTotal += (s.eval || []).length;
    for (const [, macro] of s.examples.concat(s.eval || [])) {
      contractChecked++;
      if (checkContract(s.contract, macro, s.spec).ok) contractClean++;
    }
  }
  const cal = SKILLS.find((s) => s.key === 'inbox-calendar');
  const calOps = cal.spec.ops.map((o) => o.name);
  let provMacros = 0, provClean = 0, opComplete = 0;
  for (const [id, profile] of Object.entries(PROVIDERS)) {
    if (calOps.every((n) => n in profile.opMap)) opComplete++;
    const { examples, eval: ev } = generateCorpus(`inbox-calendar:${id}`, profile, INTENTS, OOS);
    for (const [, macro] of examples.concat(ev)) {
      provMacros++;
      if (checkContract(cal.contract, macro, cal.spec).ok) provClean++;
    }
  }
  // action planner: dry-run plan steps across providers + the safety invariant
  let planSteps = 0;
  for (const id of Object.keys(PROVIDERS)) {
    for (const [, macro] of cal.eval || []) {
      if (macro === 'OUT_OF_SCOPE') continue;
      planSteps += planFor('inbox-calendar', macro, { providerId: id }).steps.length;
    }
  }
  const executorsDryRun = Object.keys(EXECUTORS).length === 1 && EXECUTORS['dry-run'] ? 1 : 0;

  return {
    corpus_total: total,
    valid_macros: valid,
    held_out_eval: evalTotal,
    contract_clean_rate: contractChecked ? contractClean / contractChecked : 0,
    skills: SKILLS.length,
    calendar_providers: Object.keys(PROVIDERS).length,
    provider_macros_checked: provMacros,
    provider_contract_clean_rate: provMacros ? provClean / provMacros : 0,
    provider_op_coverage_complete: opComplete,
    plan_steps_checked: planSteps,
    executors_are_dry_run: executorsDryRun,
  };
}

const measured = measure();
const fmt = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(4));
const rows = [];
const regressions = [];
const advances = [];

for (const [metric, floor] of Object.entries(baseline.metrics)) {
  const got = measured[metric];
  if (got === undefined) { regressions.push(`${metric}: NOT MEASURED (floor ${fmt(floor)})`); continue; }
  const ok = got >= floor;
  if (!ok) regressions.push(`${metric}: ${fmt(got)} < floor ${fmt(floor)}`);
  else if (got > floor) advances.push(`${metric}: ${fmt(got)} > floor ${fmt(floor)} (ratchet can advance)`);
  rows.push(`${metric.padEnd(32)} floor ${fmt(floor).padStart(8)}   measured ${fmt(got).padStart(8)}   ${ok ? (got > floor ? '▲' : '=') : '✗'}`);
}

console.log(rows.join('\n'));

if (bump) {
  for (const [metric, floor] of Object.entries(baseline.metrics)) {
    const got = measured[metric];
    if (got !== undefined) baseline.metrics[metric] = Math.max(floor, got); // tighten-only
  }
  writeFileSync(BASELINE_URL, JSON.stringify(baseline, null, 2) + '\n');
  console.log('\nRATCHET_BUMPED — floors raised to current measurements (tighten-only).');
  process.exit(0);
}

if (advances.length) { console.log('\nADVANCES AVAILABLE (run --bump to lock in):'); for (const a of advances) console.log('  ▲ ' + a); }
if (regressions.length) { console.log('\nREGRESSIONS (below protected floor):'); for (const r of regressions) console.log('  ✗ ' + r); }
console.log(regressions.length ? 'RATCHET_FAIL' : 'RATCHET_PASS');
process.exit(regressions.length ? 1 : 0);
