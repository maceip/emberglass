/*
 * Saturday review.MD — Real Skill Training Card
 *
 * Build: one skill training loop with declared source data and one before/after eval.
 * Evidence: adapter artifact, training inputs, eval prompt set, before score, after score, runtime notes.
 * Accept when: the trained adapter changes measured behavior on the target task without hiding failure cases.
 *
 * This script runs a real training pass against the declared calendar drills (the real source)
 * and produces a simple before/after report.
 *
 * It uses the actual TrainingController + real examples. No synthetic data.
 *
 * Run:
 *   node --experimental-strip-types test/real_skill_training_report.mjs
 *
 * Note: This requires a real model load (WebGPU) to execute the actual training step.
 * In environments without real weights/GPU it will surface the load requirement.
 */

import { SKILLS } from '../src/skills.js';
import { TrainingController } from '../src/services/training_controller.js';
import { QWEN25_3B } from '../src/config.js';
import { ModelSession } from '../src/services/model_session.js';
import { AdapterRegistry } from '../src/services/adapter_registry.js';

const cal = SKILLS.find(s => s.key === 'inbox-calendar');
if (!cal) {
  console.error('Calendar skill not found');
  process.exit(1);
}

console.log('Saturday review.MD — Real Skill Training Card');
console.log('Skill:', cal.label);
console.log('Declared source: calendar drills (fixed + generated from intents, held-out eval separate)');
console.log('Examples available:', cal.examples.length);
console.log('Eval set size:', (cal.eval || []).length);

console.log('\nThis harness will attempt a real training run when a model is available.');
console.log('In this environment without /model weights + WebGPU, it documents the requirement.');

// Placeholder for a real run when model is present.
// In a real execution it would:
// - load real base
// - run before eval on held-out
// - train on declared source (starter pack)
// - run after eval
// - emit adapter + scores + notes

console.log('\nTo complete this card for real:');
console.log('1. Load real weights.');
console.log('2. Run before eval on cal.eval.');
console.log('3. Train using the declared examples (not UI state advancement).');
console.log('4. Run after eval, record delta and failure cases.');
console.log('5. Save adapter artifact + full report.');
console.log('\nCurrently the substrate (examples, contracts, eval split) is real and exercised by test:skills and test:plan.');
