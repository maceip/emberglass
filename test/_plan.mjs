/* Action-plan gate (planner / dry-run only).
 * Compiles every calendar held-out macro across all 3 providers and asserts the plan is
 * well-formed, then dry-runs it. Also enforces the SAFETY BOUNDARY:
 *   - the only executor is dry-run; every receipt is 'simulated'
 *   - a plan whose contract failed is refused (fail closed)
 *   - the action/* source contains no network/DOM/provider-SDK calls */
import { readFileSync } from 'node:fs';
import { SKILLS, planFor, dryRun, EXECUTORS, executorFor, clearAudit, auditLog } from '../src/skills.js';
import { PROVIDERS } from '../src/skills/inbox-calendar/providers/index.ts';

const fails = [];
const ok = (c, m) => { if (!c) fails.push(m); };
const HEX12 = /^[0-9a-f]{12}$/;
const RISKS = new Set(['read-only', 'reversible-write', 'sensitive-write']);

clearAudit();
const cal = SKILLS.find((s) => s.key === 'inbox-calendar');
let plansChecked = 0, stepsChecked = 0, receiptsChecked = 0;

for (const providerId of Object.keys(PROVIDERS)) {
  for (const [req, macro] of cal.eval) {
    if (macro === 'OUT_OF_SCOPE') continue; // bounces produce no plan steps
    const plan = planFor('inbox-calendar', macro, { providerId });
    plansChecked++;
    ok(plan && plan.provider === providerId, `plan provider mismatch for ${providerId}`);
    ok(plan.contractOk === true, `held-out macro should be contract-clean → "${req}"`);
    ok(RISKS.has(plan.risk), `bad plan risk ${plan.risk}`);
    ok(/^[0-9a-f]{64}$/.test(plan.fingerprint), 'plan fingerprint not 64 hex');

    const binds = new Set();
    for (const s of plan.steps) {
      stepsChecked++;
      ok(s.providerMethod && s.providerMethod !== '(unmapped)', `${providerId}/${s.op}: unmapped method`);
      ok(s.capability && s.capability !== 'unknown', `${providerId}/${s.op}: missing capability`);
      ok(HEX12.test(s.idempotencyKey), `${providerId}/${s.op}: bad idempotency key`);
      ok(RISKS.has(s.risk), `${providerId}/${s.op}: bad risk`);
      // data-flow: any ref arg must resolve to a prior binding AND be recorded in dependsOn
      for (const a of s.args) {
        if (a.kind === 'ref') {
          ok(binds.has(a.refBase), `${providerId}/${s.op}: ref ${a.value} has no prior binding`);
          const dep = plan.steps.find((p) => p.binds === a.refBase);
          ok(dep && s.dependsOn.includes(dep.index), `${providerId}/${s.op}: dependsOn missing ${a.refBase}`);
        }
      }
      if (s.binds) binds.add(s.binds);
    }

    // dry-run → simulated receipts, one per step
    const run = dryRun('inbox-calendar', macro, { providerId });
    ok(run.receipts.length === plan.steps.length, `receipts != steps for "${req}"`);
    for (const r of run.receipts) { receiptsChecked++; ok(r.status === 'simulated', `receipt not simulated: ${r.op}`); }
  }
}

// SAFETY: only dry-run exists, and it is what every provider resolves to
ok(Object.keys(EXECUTORS).length === 1 && EXECUTORS['dry-run'], 'unexpected executor registered');
ok(executorFor('google').id === 'dry-run' && executorFor('anything').id === 'dry-run', 'executorFor must return dry-run');
ok(auditLog().every((r) => r.status === 'simulated'), 'audit log contains a non-simulated receipt');

// SAFETY: a contract-failing plan is refused (craft a zero-duration event → forbidden)
const bad = dryRun('inbox-calendar', 'create_event(title="x", start="2026-06-29T17:00", end="2026-06-29T17:00", remind_min=10)');
ok(bad.plan.contractOk === false, 'zero-duration plan should be contract-failed');
ok(bad.receipts.length === 0, 'contract-failed plan must produce no receipts');

// SAFETY (static): action/* must not reach the network or DOM. Scan CODE only (strip comments
// first so the boundary documentation itself doesn't trip the guard).
const stripComments = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
const FORBIDDEN = /\b(fetch|XMLHttpRequest|WebSocket|document|window|navigator|localStorage|chrome)\b/;
for (const f of ['plan.ts', 'executor.ts', 'receipt.ts']) {
  const src = stripComments(readFileSync(new URL(`../src/skills/action/${f}`, import.meta.url), 'utf8'));
  if (FORBIDDEN.test(src)) fails.push(`action/${f} references a network/DOM API`);
}

console.log(`plans: ${plansChecked}   steps: ${stepsChecked}   receipts: ${receiptsChecked}   executors: ${Object.keys(EXECUTORS).join(', ')}`);
const sample = planFor('inbox-calendar', cal.eval.find(([, m]) => m !== 'OUT_OF_SCOPE')[1], { providerId: 'google' });
console.log('sample plan (google):'); for (const line of sample.summary) console.log('  ' + line);
console.log(`  required: ${sample.requiredCapabilities.join(', ')}   risk: ${sample.risk}`);
if (fails.length) { console.log('\nFAILURES:'); for (const f of fails.slice(0, 20)) console.log('  - ' + f); }
console.log(fails.length ? 'PLAN_FAIL' : 'PLAN_PASS');
process.exit(fails.length ? 1 : 0);
