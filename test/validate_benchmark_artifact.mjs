/*
 * Saturday review.MD — Real Browser Benchmark Card validator
 *
 * Enforces that any committed benchmark-artifact.json contains the required evidence fields.
 * Run: node test/validate_benchmark_artifact.mjs
 *
 * This is real enforcement. Fails if the artifact is missing or incomplete.
 */

import { existsSync, readFileSync } from 'node:fs';

const ART = 'benchmark-artifact.json';

if (!existsSync(ART)) {
  console.error('No benchmark-artifact.json. Per Real Browser Benchmark Card: must be produced from real /model + real browser run.');
  process.exit(2);
}

let data;
try {
  data = JSON.parse(readFileSync(ART, 'utf8'));
} catch (e) {
  console.error('Invalid JSON in benchmark-artifact.json');
  process.exit(1);
}

const requiredTop = ['schema', 'generatedBy', 'capturedAt', 'environment', 'model', 'rows'];
const missingTop = requiredTop.filter(k => !(k in data));
if (missingTop.length) {
  console.error('Missing top-level fields:', missingTop);
  process.exit(1);
}

if (!data.environment || !data.environment.userAgent) {
  console.error('Missing environment.userAgent (needed for Chrome version)');
  process.exit(1);
}

if (!data.model || !data.model.path) {
  console.error('Missing model.path');
  process.exit(1);
}

const rows = Array.isArray(data.rows) ? data.rows : [];
const hasLoad = rows.some(r => r.type === 'load' && !r.skipped);
const hasDecode = rows.some(r => r.type === 'greedy-decode' && r.tokPerSec != null);
const hasTrain = rows.some(r => r.type === 'train-step');

if (!hasLoad || !hasDecode || !hasTrain) {
  console.error('Artifact must contain real load + decode + train measurements from a successful run.');
  process.exit(1);
}

console.log('BENCHMARK_ARTIFACT_VALID');
process.exit(0);
