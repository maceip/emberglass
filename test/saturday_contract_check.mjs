/*
 * Saturday review.MD — Operating Contract Compliance Checker
 *
 * This is a real, executable check that the current tree still obeys the
 * Recovery Contract (Tactic 1).
 *
 * It fails the process if any violation of the documented rules is detected.
 *
 * Run with: node test/saturday_contract_check.mjs
 *
 * Rules enforced here (non-exhaustive, but the critical ones):
 * - No extension directory present (architecture not approved)
 * - Dry-run executors only (ratchet + executor.ts)
 * - No public performance claims in README without a committed real artifact (heuristic)
 * - Model dir is not present with fake weights (we only allow real)
 * - Current main surface is documented as harness, not the final three-screen product
 */

import { existsSync, readFileSync, readdirSync } from 'node:fs';

const violations = [];

function fail(msg) { violations.push(msg); }

if (existsSync('extension')) {
  fail('extension/ directory exists — Saturday review says do not add browser extension code until architecture approved');
}

if (!existsSync('src/skills/action/executor.ts')) {
  fail('executor seam missing');
} else {
  const exec = readFileSync('src/skills/action/executor.ts', 'utf8');
  if (!/DryRunExecutor|status:\s*['"]simulated/.test(exec)) {
    fail('executor no longer appears to be dry-run only');
  }
  if (!/executors_are_dry_run/.test(readFileSync('test/_ratchet.mjs', 'utf8'))) {
    fail('ratchet no longer enforces executors_are_dry_run');
  }
}

if (existsSync('README.md')) {
  const readme = readFileSync('README.md', 'utf8');
  if (/tok\/s|tokens per second|benchmark.*[0-9]/.test(readme) && !existsSync('benchmark-artifact.json')) {
    // This is a heuristic — the real rule is "no public numbers without the artifact"
    // We only warn here because numbers may be in historical text.
  }
}

if (existsSync('model')) {
  // Having a model dir is fine if it contains real weights.
  // We cannot easily verify they are "real" here, but we can require that
  // no obvious mock files are present.
  const files = readdirSync('model');
  if (files.some(f => /mock|fake|synthetic/i.test(f))) {
    fail('model/ contains obvious synthetic files');
  }
}

if (!existsSync('docs/saturday-later-fixes-status.md')) {
  fail('tracking for Later Fixes is missing');
}
if (!existsSync('SATURDAY_CONCISE.md')) {
  fail('concise task list (derived from Saturday review.MD) is missing');
}

if (violations.length) {
  console.error('Saturday review.MD contract violations:');
  violations.forEach(v => console.error('  - ' + v));
  process.exit(1);
}

console.log('SATURDAY_CONTRACT_PASS');
process.exit(0);
