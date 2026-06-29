/*
 * Saturday review.MD — Real Provider Evaluation Card (the current blocker) is the dedicated Google test account + your login on calendar.google.com; the harness (test/real_provider_eval.mjs) is already written for a real full cycle against the dedicated test account (auth → real readBefore on Calendar API → contract-verified plan → real create_event write → readAfter + delta → cleanup delete) and refuses and prints instructions if it can't do it for real — preferred: Cursor Desktop → Dashboard → Cloud Agents → Secrets add GOOGLE_TEST_TOKEN (OAuth2 access token for the dedicated test account with https://www.googleapis.com/auth/calendar scope) then say "go" and I'll run, produce the artifact, commit/push/update PR; alt: run locally where your calendar login lives and push the artifact.
 */

import { SKILLS, planFor, clearAudit } from '../src/skills.js';
import { PROVIDERS } from '../src/skills/inbox-calendar/providers/index.ts';

const args = process.argv.slice(2);
const providerArg = (args.find(a => a.startsWith('--provider=')) || '').split('=')[1] || 'google';
const accountLabel = (args.find(a => a.startsWith('--account=')) || '').split('=')[1] || process.env.GOOGLE_TEST_ACCOUNT || 'dedicated-test-google';
const explicitToken = (args.find(a => a.startsWith('--token=')) || '').split('=')[1];

clearAudit();

const TOKEN = explicitToken || process.env.GOOGLE_TEST_TOKEN || process.env.GOOGLE_CALENDAR_TOKEN || null;

console.log('Saturday review.MD — Real Provider Evaluation Card');
console.log('Provider:', providerArg);
console.log('Account label:', accountLabel);
console.log('Token present:', !!TOKEN, TOKEN ? '(length ' + TOKEN.length + ')' : '(MISSING — see instructions below)');

const cal = SKILLS.find(s => s.key === 'inbox-calendar');
if (!cal) {
  console.error('inbox-calendar skill not found');
  process.exit(1);
}

const profile = PROVIDERS[providerArg] || PROVIDERS.google;

function isoMin(dt) { return dt.toISOString(); }

async function googleFetch(path, init = {}) {
  if (!TOKEN) throw new Error('no token for real Google call');
  const url = 'https://www.googleapis.com' + path;
  const res = await fetch(url, {
    ...init,
    headers: {
      'Authorization': `Bearer ${TOKEN}`,
      'Content-Type': 'application/json',
      ...(init.headers || {}),
    },
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`Google API ${res.status} ${path}: ${txt.slice(0, 300)}`);
  }
  return res.json();
}

async function realAuthAndCalendars() {
  // Minimal auth probe + list of calendars (real read)
  const list = await googleFetch('/calendar/v3/users/me/calendarList?maxResults=5');
  return {
    status: 'success',
    detail: 'OAuth Bearer token accepted',
    calendars: (list.items || []).map(c => ({ id: c.id, summary: c.summary, primary: !!c.primary })),
  };
}

async function realReadBefore() {
  const now = new Date();
  const soon = new Date(now.getTime() + 1000 * 60 * 60 * 24 * 7); // next 7d
  const params = new URLSearchParams({
    timeMin: isoMin(now),
    timeMax: isoMin(soon),
    singleEvents: 'true',
    orderBy: 'startTime',
    maxResults: '10',
  });
  const data = await googleFetch(`/calendar/v3/calendars/primary/events?${params}`);
  const events = (data.items || []).map(e => ({
    id: e.id,
    summary: e.summary || '(no title)',
    start: e.start?.dateTime || e.start?.date,
  }));
  return { status: 'success', resource: 'primary calendar events (next 7d)', count: events.length, sample: events.slice(0, 3) };
}

function pickSafeCreateEventMacro() {
  // A minimal, contract-passing create_event using the port.
  // We synthesize a near-future slot so it is unlikely to collide.
  const start = new Date(Date.now() + 1000 * 60 * 30); // +30min
  const end = new Date(start.getTime() + 1000 * 60 * 15); // 15min duration
  const fmt = (d) => d.toISOString().slice(0, 16); // YYYY-MM-DDTHH:MM as the port expects
  const s = fmt(start);
  const e = fmt(end);
  // This must be accepted by the calendar contract (create_event(title, start, end, remind_min))
  return `create_event(title="EMBERGLASS-EVAL-${Date.now()}", start="${s}", end="${e}", remind_min="5")`;
}

async function realWrite(planStep) {
  // Translate the planned create_event into a real Calendar API insert.
  // This lives ONLY in the eval harness (test-only real executor for the dedicated account).
  // The shared action layer and browser app remain dry-run until the full action layer review.
  const args = {};
  for (const a of planStep.args || []) {
    if (a.kind === 'string') args[a.key] = a.value;
  }
  const title = args.title || 'EMBERGLASS-EVAL';
  const start = args.start;
  const end = args.end;
  if (!start || !end) throw new Error('plan missing start/end for create_event');

  const body = {
    summary: title,
    start: { dateTime: start + ':00', timeZone: 'UTC' },
    end: { dateTime: end + ':00', timeZone: 'UTC' },
    reminders: { useDefault: false, overrides: [{ method: 'popup', minutes: parseInt(args.remind_min || '5', 10) }] },
  };

  const created = await googleFetch('/calendar/v3/calendars/primary/events', {
    method: 'POST',
    body: JSON.stringify(body),
  });
  return {
    status: 'confirmed',
    providerRequestId: created.id,
    htmlLink: created.htmlLink,
    detail: `created event id=${created.id}`,
  };
}

async function realReadAfter(eventId) {
  const got = await googleFetch(`/calendar/v3/calendars/primary/events/${encodeURIComponent(eventId)}`);
  return {
    status: 'verified',
    delta: `found event "${got.summary}" at ${got.start?.dateTime || got.start?.date}`,
    event: { id: got.id, summary: got.summary },
  };
}

async function realCleanup(eventId) {
  await googleFetch(`/calendar/v3/calendars/primary/events/${encodeURIComponent(eventId)}`, { method: 'DELETE' });
  return { status: 'success', detail: `deleted ${eventId}` };
}

async function runRealGoogleCycle() {
  const steps = [];
  const receipts = [];

  // 1. auth
  try {
    const auth = await realAuthAndCalendars();
    steps.push({ name: 'auth', status: auth.status, detail: auth.detail });
  } catch (e) {
    steps.push({ name: 'auth', status: 'error', detail: String(e.message || e) });
    return { ok: false, steps, artifact: null };
  }

  // 2. readBefore (real state)
  let before;
  try {
    before = await realReadBefore();
    steps.push({ name: 'readBefore', status: before.status, resource: before.resource, count: before.count, sample: before.sample });
  } catch (e) {
    steps.push({ name: 'readBefore', status: 'error', detail: String(e.message || e) });
    return { ok: false, steps, artifact: null };
  }

  // 3. plan a safe write (using the real declared contract + profile)
  const macro = pickSafeCreateEventMacro();
  const plan = planFor('inbox-calendar', macro, { providerId: providerArg });
  if (!plan || !plan.contractOk) {
    steps.push({ name: 'plan', status: 'contract-failed', detail: 'planFor did not produce contractOk plan' });
    return { ok: false, steps, artifact: null };
  }
  const writeStep = plan.steps.find(s => s.op === 'create_event') || plan.steps[plan.steps.length - 1];
  steps.push({ name: 'plan', status: 'contract-passed', planId: plan.fingerprint, op: writeStep.op, providerMethod: writeStep.providerMethod });

  // 4. real write on the dedicated account
  let writeRes;
  try {
    writeRes = await realWrite(writeStep);
    steps.push({ name: 'write', status: writeRes.status, providerRequestId: writeRes.providerRequestId, detail: writeRes.detail });
    receipts.push({ step: 0, op: 'create_event', provider: 'google', method: 'calendar.events.insert', status: 'confirmed', providerRequestId: writeRes.providerRequestId, at: new Date().toISOString() });
  } catch (e) {
    steps.push({ name: 'write', status: 'error', detail: String(e.message || e) });
    return { ok: false, steps, artifact: null };
  }

  // 5. readAfter
  try {
    const after = await realReadAfter(writeRes.providerRequestId);
    steps.push({ name: 'readAfter', status: after.status, delta: after.delta, event: after.event });
  } catch (e) {
    steps.push({ name: 'readAfter', status: 'error', detail: String(e.message || e) });
    // still attempt cleanup
  }

  // 6. cleanup (return account to previous state)
  try {
    const cl = await realCleanup(writeRes.providerRequestId);
    steps.push({ name: 'cleanup', status: cl.status, detail: cl.detail });
  } catch (e) {
    steps.push({ name: 'cleanup', status: 'error', detail: String(e.message || e) });
  }

  const artifact = {
    schema: 'emberglass/real-provider-eval-artifact/v1',
    provider: providerArg,
    accountLabel,
    capturedAt: new Date().toISOString(),
    steps,
    receipts,
    notes: 'Real operations against dedicated Google test account only. No local state. Browser login context at calendar.google.com is the end-user surface; this harness used direct Calendar API v3.',
  };

  return { ok: steps.every(s => ['success', 'contract-passed', 'confirmed', 'verified'].includes(s.status)), steps, artifact };
}

async function main() {
  // Always run the planning verification (this part is pure and always "real" w.r.t. the declared drills/contract)
  let plans = 0, contractOk = 0;
  for (const [req, macro] of (cal.eval || [])) {
    if (macro === 'OUT_OF_SCOPE') continue;
    const p = planFor('inbox-calendar', macro, { providerId: providerArg });
    if (p) {
      plans++;
      if (p.contractOk) contractOk++;
    }
  }
  console.log(`\nPlanning verification on held-out eval: ${plans} plans, ${contractOk} contractOk (provider=${providerArg}).`);

  if (providerArg !== 'google') {
    console.log('\nOnly google provider path is wired for real execution in this harness right now.');
    console.log('For other providers, supply the account and the real cycle can be extended similarly.');
    process.exit(2);
  }

  if (!TOKEN) {
    console.log('\n=== MISSING REAL CREDENTIALS ===');
    console.log('No GOOGLE_TEST_TOKEN found in environment.');
    console.log('');
    console.log('To give this cloud agent your Google test account:');
    console.log('  1. In Cursor Desktop: open Dashboard (or the cloud agents section).');
    console.log('  2. Cloud Agents > Secrets > add a new secret:');
    console.log('       Name:  GOOGLE_TEST_TOKEN');
    console.log('       Value: <access token for the dedicated test Google account with https://www.googleapis.com/auth/calendar scope>');
    console.log('  3. (Optional) also set GOOGLE_TEST_ACCOUNT to a short label.');
    console.log('  4. Reconnect or restart the cloud agent for this workspace so the secret is injected.');
    console.log('');
    console.log('Once injected, re-run:');
    console.log('  npm run test:real-provider-eval -- --provider=google --account=your-test-label');
    console.log('');
    console.log('The browser-side "login on the calendar" (you logged in at calendar.google.com) is the');
    console.log('runtime context for a normal user of the Skillbook. This harness uses the API for the eval card.');
    console.log('================================');
    // Still exit non-zero until we have a full real artifact
    process.exit(3);
  }

  // Real path
  const { ok, steps, artifact } = await runRealGoogleCycle();

  if (artifact) {
    const outPath = 'test/provider-eval-artifact-google.json';
    // Use fs to write (node built-in)
    const { writeFileSync } = await import('node:fs');
    writeFileSync(outPath, JSON.stringify(artifact, null, 2));
    console.log('\nWrote real artifact to', outPath);
  }

  console.log('\nCycle steps:');
  for (const s of steps) console.dir(s);

  if (!ok) {
    console.log('\nReal provider eval cycle did not complete successfully. See errors above.');
    process.exit(4);
  }

  console.log('\n=== REAL PROVIDER EVALUATION CARD — PASS (Google) ===');
  console.log('Full auth/read/plan/confirmed-write/read-after/cleanup performed against dedicated account.');
  console.log('Artifact committed (when present) is the evidence. No synthetic data.');
}

main().catch(e => { console.error(e); process.exit(99); });