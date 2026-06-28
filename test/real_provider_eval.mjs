/*
 * Saturday review.MD — Real Provider Evaluation Card
 *
 * This is the harness for producing real evidence against a dedicated test account.
 *
 * Rules (from the document):
 * - One dedicated test account path for one provider.
 * - Evidence MUST include: real auth, real read, planned write, confirmed write,
 *   read-after-write verification, and cleanup.
 * - The artifact must be produced without relying on local provider state.
 * - Reject if provider data is constructed locally or the pass only proves internal planner behavior.
 *
 * Current state: dry-run only (as required until the action layer is designed).
 * This file will be extended with real account wiring when a dedicated test account
 * is provided. It must never fall back to local mocks or synthetic stores.
 *
 * To run against a real account (future):
 *   node test/real_provider_eval.mjs --provider google --account <dedicated-test>
 *
 * Until then, this script will refuse to claim "real evaluation" success.
 */

import { SKILLS, planFor, dryRun, clearAudit, auditLog } from '../src/skills.js';
import { PROVIDERS } from '../src/skills/inbox-calendar/providers/index.ts';

const args = process.argv.slice(2);
const providerArg = (args.find(a => a.startsWith('--provider=')) || '').split('=')[1] || 'google';
const account = (args.find(a => a.startsWith('--account=')) || '').split('=')[1];

clearAudit();

console.log('Saturday review.MD — Real Provider Evaluation Card');
console.log('Provider:', providerArg);
console.log('Account:', account || '(none provided — this run cannot produce real evidence)');

if (!account) {
  console.log('\nNo dedicated real test account provided.');
  console.log('Per the Real Provider Evaluation Card:');
  console.log('  - Evidence must come from real auth/read/write/verify/cleanup on a dedicated account.');
  console.log('  - Local stores or synthetic data are rejected.');
  console.log('\nThis harness currently supports only the planning + dry-run layer (correct per current ratchet).');
  console.log('Real account execution path is not yet wired.');
  console.log('\nTo produce a valid artifact for this card, run with a real dedicated account once available.');
  process.exit(2);
}

// When a real account is provided, the following would exercise real auth/read etc.
// For now we only demonstrate the contract-clean planning path on held-out eval.

const cal = SKILLS.find(s => s.key === 'inbox-calendar');
let plans = 0, verified = 0;

for (const [req, macro] of cal.eval) {
  if (macro === 'OUT_OF_SCOPE') continue;
  const plan = planFor('inbox-calendar', macro, { providerId: providerArg });
  if (plan && plan.contractOk) {
    plans++;
    const dr = dryRun('inbox-calendar', macro, { providerId: providerArg });
    if (dr && dr.receipts.every(r => r.status === 'simulated')) {
      verified++;
    }
  }
}

console.log(`\nPlanning layer on held-out: ${plans} plans, ${verified} dry-run verified (simulated receipts).`);

console.log('\nNOTE: This is NOT yet a completed Real Provider Evaluation per the Saturday review card.');
console.log('A real dedicated account + real auth/read/write/verify/cleanup steps + raw artifact are still required.');
console.log('Required shape: see test/provider_eval_artifact_template.json');

if (account) {
  // Placeholder for future real account steps. Must never be synthetic.
  console.log('\nReal account steps would go here (auth, read current state, perform planned change, read-after-write, cleanup).');
  console.log('Output must match provider_eval_artifact_template.json structure.');
  process.exit(2);
}
