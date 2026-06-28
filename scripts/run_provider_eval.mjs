#!/usr/bin/env node
/*
 * Provider evaluation runner (Saturday review — Real Provider Evaluation Card).
 *
 * REQUIRES real credentials — no local substitutes:
 *   GOOGLE_OAUTH_CLIENT_ID
 *   GOOGLE_OAUTH_CLIENT_SECRET
 *   GOOGLE_TEST_REFRESH_TOKEN   (dedicated test account)
 *
 * Usage:
 *   node scripts/run_provider_eval.mjs
 *
 * Writes provider-eval-artifact.json on success.
 * Without credentials, exits with implementation_required status and documents blockers.
 */
import { writeFile } from 'node:fs/promises';

const REQUIRED = ['GOOGLE_OAUTH_CLIENT_ID', 'GOOGLE_OAUTH_CLIENT_SECRET', 'GOOGLE_TEST_REFRESH_TOKEN'];

function missing() {
  return REQUIRED.filter((k) => !process.env[k]);
}

const blockers = missing();
if (blockers.length) {
  const artifact = {
    schema: 'emberglass/provider-eval-artifact/v1',
    status: 'implementation_required',
    capturedAt: new Date().toISOString(),
    blockers: [
      ...blockers.map((k) => `missing env: ${k}`),
      'app-action layer not approved for writes (see docs/app-action-layer-design.md)',
      'no dedicated test account OAuth refresh token configured on this machine',
    ],
    nextSteps: [
      'Create dedicated Google test account for Calendar eval',
      'Register OAuth client; store refresh token in env (never commit)',
      'Approve app-action layer design',
      'Implement read-only Calendar API executor for eval v1',
      'Add write + read-after-write + cleanup after action layer approval',
    ],
    evidenceRequired: ['real auth', 'real read', 'planned write', 'confirmed write', 'read-after-write', 'cleanup'],
  };
  await writeFile('provider-eval-artifact.json', JSON.stringify(artifact, null, 2));
  console.error('[provider-eval] implementation_required — missing:', blockers.join(', '));
  console.error('Wrote provider-eval-artifact.json (blocked status only — not a passing eval)');
  process.exit(2);
}

// Real path would: refresh token → Calendar API list events → plan write → ApprovalPacket → execute → verify → cleanup
console.error('[provider-eval] credentials present but write executor is implementation_required until app-action layer ships.');
process.exit(3);
