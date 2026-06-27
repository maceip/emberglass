/* CLOSED-LOOP verification (L2) — the honest "did the action actually happen?" check.
 *
 * The dry-run executor only emits `status:'simulated'` receipts; it proves nothing
 * landed. This harness instead runs each plan against a REAL in-memory calendar/inbox
 * store, then READS THE STORE BACK and asserts the intended writes are present with
 * fully-resolved arguments (read->write dataflow wired, refs resolved, no leftovers).
 *
 * What this measures: the macro -> plan -> execute -> read-back pipeline is faithful,
 * over the skill's held-out GOLDEN eval split. It does NOT touch real Google (that's
 * L3, unbuilt — see docs/VERIFICATION.md) and it uses golden macros, so it is NOT a
 * measure of model accuracy (that's L1, the GPU eval). It is a true closed loop for
 * the execution layer and yields a concrete success rate.
 *
 * Run: node test/verify_calendar_closed_loop.mjs   (writes the rate to stdout)
 */
import { SKILLS, planFor } from '../src/skills.js';

const KEY = 'inbox-calendar';
const skill = SKILLS.find((s) => s.key === KEY);
if (!skill) { console.error('skill not found:', KEY); process.exit(2); }

// op effect lookup from the canonical port
const EFFECT = new Map(skill.spec.ops.map((o) => [o.name, o.effect || 'write']));
const isWrite = (op) => EFFECT.get(op) === 'write';

function isoAdd(iso, mins) {
  const m = String(iso).match(/(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  if (!m) return iso;
  const t = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) + (mins | 0) * 60000);
  const p = (n) => String(n).padStart(2, '0');
  return `${t.getUTCFullYear()}-${p(t.getUTCMonth() + 1)}-${p(t.getUTCDate())}T${p(t.getUTCHours())}:${p(t.getUTCMinutes())}`;
}

/* A genuinely separate executor: it MUTATES a store. (The shipped DryRunExecutor does
 * not — that's the whole point.) Reads return synthetic bindings so write steps that
 * depend on a read (find_slot -> create_event) must resolve their refs to land. */
function executeAgainstStore(plan) {
  const store = { events: [], mail: [], reminders: [], rsvps: [], labels: [], archives: [], replies: [], forwards: [] };
  const bindings = {};
  const problems = [];

  const resolve = (a) => {
    if (a.kind === 'string') return a.value;
    if (a.kind === 'number') return Number(a.value);
    // ref like `s` or `s.start`
    const base = a.refBase ?? String(a.value).split('.')[0];
    const path = String(a.value).split('.').slice(1);
    let v = bindings[base];
    if (v === undefined) { problems.push(`unresolved ref ${a.value}`); return { __unresolved: a.value }; }
    for (const k of path) v = v?.[k];
    if (v === undefined) { problems.push(`unresolved ref ${a.value}`); return { __unresolved: a.value }; }
    return v;
  };
  const argsObj = (s) => Object.fromEntries(s.args.map((a) => [a.key, resolve(a)]));

  for (const s of plan.steps) {
    const A = argsObj(s);
    if (!isWrite(s.op)) {
      // synthesize a deterministic read result, then bind it for downstream refs
      let ret;
      if (s.op === 'find_slot') { const start = A.after || '2026-06-29T09:00'; ret = { start, end: isoAdd(start, A.duration_min ?? 30), id: 'slot-1' }; }
      else if (s.op === 'find_email') ret = { id: 'thread-1', subject: '(found)', from: 'someone@x.com' };
      else ret = { id: `${s.op}-1` };
      if (s.binds) bindings[s.binds] = ret;
      continue;
    }
    const rec = { op: s.op, args: A };
    if (s.op === 'create_event') store.events.push(rec);
    else if (s.op === 'set_reminder') store.reminders.push(rec);
    else if (s.op === 'rsvp') store.rsvps.push(rec);
    else if (s.op === 'label_email') store.labels.push(rec);
    else if (s.op === 'archive_email') store.archives.push(rec);
    else if (s.op === 'reply_email') store.replies.push(rec);
    else if (s.op === 'forward_email') store.forwards.push(rec);
    else if (/email/.test(s.op)) store.mail.push(rec); // compose_email / schedule_send
    else store.events.push(rec);
    if (s.binds) bindings[s.binds] = { id: `${s.op}-1`, ...A };
  }
  return { store, problems };
}

// flatten every record in the store for read-back
const allRecords = (store) => Object.values(store).flat();
const hasUnresolved = (rec) => Object.values(rec.args).some((v) => v && typeof v === 'object' && '__unresolved' in v);

let pass = 0, fail = 0;
const failures = [];
const byOp = {}; // headline op -> {pass, fail}

for (const [request, golden] of skill.eval) {
  const oos = /^\s*OUT_OF_SCOPE\s*$/.test(golden);
  const plan = planFor(KEY, golden);
  let ok = true; let why = '';

  if (oos) {
    // closed loop for a bounce: nothing must be written, store stays empty
    const { store } = executeAgainstStore(plan);
    if (allRecords(store).length !== 0) { ok = false; why = 'OOS bounce produced writes'; }
  } else {
    if (!plan.contractOk) { ok = false; why = 'contract failed'; }
    else {
      const { store, problems } = executeAgainstStore(plan);
      const writeSteps = plan.steps.filter((s) => isWrite(s.op));
      const landed = allRecords(store);
      // read-back: every write step must appear in the store, fully resolved
      if (problems.length) { ok = false; why = problems[0]; }
      else if (landed.length !== writeSteps.length) { ok = false; why = `landed ${landed.length} of ${writeSteps.length} writes`; }
      else if (landed.some(hasUnresolved)) { ok = false; why = 'a stored write has an unresolved ref'; }
      else {
        // per-write equality: the read-back args equal the macro's intended args
        for (const ws of writeSteps) {
          const intended = Object.fromEntries(ws.args.filter((a) => a.kind !== 'ref').map((a) => [a.key, a.kind === 'number' ? Number(a.value) : a.value]));
          const found = landed.find((r) => r.op === ws.op && Object.entries(intended).every(([k, v]) => r.args[k] === v));
          if (!found) { ok = false; why = `read-back missing ${ws.op}(${Object.keys(intended).join(',')})`; break; }
        }
      }
    }
  }

  const head = oos ? 'OUT_OF_SCOPE' : (plan.steps.find((s) => isWrite(s.op))?.op || plan.steps[0]?.op || 'none');
  byOp[head] = byOp[head] || { pass: 0, fail: 0 };
  if (ok) { pass++; byOp[head].pass++; }
  else { fail++; byOp[head].fail++; failures.push({ request: request.slice(0, 70), why, golden: golden.replace(/\n/g, ' ⏎ ').slice(0, 90) }); }
}

const total = pass + fail;
const rate = total ? ((pass / total) * 100).toFixed(1) : '0.0';
console.log('\nCLOSED-LOOP (L2) — execute against in-memory store + read back');
console.log(`  skill        : ${skill.label} (${KEY})`);
console.log(`  eval cases   : ${total} (held-out golden split)`);
console.log(`  passed       : ${pass}`);
console.log(`  failed       : ${fail}`);
console.log(`  SUCCESS RATE : ${rate}%`);
console.log('  by headline op:');
for (const [op, r] of Object.entries(byOp).sort()) console.log(`     ${op.padEnd(16)} ${r.pass}/${r.pass + r.fail}`);
if (failures.length) {
  console.log('\n  failures:');
  for (const f of failures.slice(0, 12)) console.log(`     ✗ "${f.request}" — ${f.why}\n        golden: ${f.golden}`);
}
console.log(fail === 0 ? '\nCLOSED_LOOP_PASS' : `\nCLOSED_LOOP_FAIL (${fail})`);
process.exit(fail === 0 ? 0 : 1);
