/*
 * Emberglass — browser harness for the custom WebGPU VibeThinker-3B runtime.
 * Two panes: INFERENCE (runs the model live) and TRAIN (in-browser LoRA fine-tune).
 * The inference pane defaults to the BASE model (neon badge) to create the incentive
 * to TRAIN; training a small adapter is fast and the result hot-swaps live so the
 * before/after is visible immediately in the same tab. Nothing leaves the device
 * except the optional "train on a URL" lane (which uses a public reader proxy).
 */
import { QWEN25_3B } from './config.js';
import { urlReader, hfReader, fileReader } from './readers.js';
import { AdapterRegistry } from './services/adapter_registry.js';
import { ModelSession } from './services/model_session.js';
import { TrainingController } from './services/training_controller.js';
import { downloadLoraAdapter, exportLoraAdapter } from './lora_export.js';
import { loadLoraAdapterGPU } from './lora_gpu.js';
import * as store from './services/store.js';

const $ = (id) => document.getElementById(id);
const log = (m) => { const s = $('railMsg'); if (s) s.textContent = m; console.log('[emberglass]', m); };

// step infographic controller for a `.steps` strip
function steps(id) {
  const el = $(id), m = {};
  el.querySelectorAll('.step').forEach((s) => (m[s.dataset.s] = s));
  const all = () => Object.values(m);
  return {
    reset() { all().forEach((s) => s.classList.remove('active', 'done', 'loop')); },
    active(k) { m[k]?.classList.add('active'); },
    activeOnly(k) { all().forEach((s) => s.classList.remove('active')); m[k]?.classList.add('active'); },
    done(k) { m[k]?.classList.remove('active', 'loop'); m[k]?.classList.add('done'); },
    loop(keys, on) { keys.forEach((k) => m[k]?.classList.toggle('loop', on)); },
  };
}
// animated stopwatch that counts up; returns a stop() fn
function startClock(id) {
  const el = $(id), t = el.querySelector('.t'), t0 = performance.now();
  let run = true;
  el.classList.add('on');
  (function f() { if (!run) return; t.textContent = ((performance.now() - t0) / 1000).toFixed(1) + 's'; requestAnimationFrame(f); })();
  return () => { run = false; el.classList.remove('on'); };
}

// ── shared session ──────────────────────────────────────────────────────────
const session = new ModelSession({ cfg: QWEN25_3B, log });
const adapters = new AdapterRegistry();
const state = {
  loaded: false,
  busy: false,
  err: null,
  tuned: null, // { name, kind:'guided'|'own', build(userText)->messages[], suggest }
  activeRunId: null, // history run currently applied
  dirHandle: null, // File System Access workspace folder
};

// Decoding settings that match WeiboAI's recommended usage for VibeThinker-3B
// (model card / paper / GitHub): sampling, NOT greedy. top_k=-1 isn't feasible
// on the GPU sampler (capped at 64), but top_p=0.95 over the top-64 is an
// excellent nucleus approximation. maxTokens is bounded for the browser — this
// is a long-CoT reasoner (rec. 40K+), so a small budget truncates its thinking.
const GEN = { maxTokens: 2048, temperature: 0.6, topP: 0.95, topK: 64 };

// ── guided "skill": Inbox & Calendar knife ───────────────────────────────────
// A skill teaches the model to compile a plain request into a MACRO over a small,
// typed action space (the "blades" of this knife). This is a constrained-codegen
// task — the model's strength — not open-domain recall. We SUPPLY the action space
// (in the system prompt + reinforced by the LoRA) so the model never has to invent
// it; out-of-scope requests must bounce. The emitted macro is the verifiable artifact.
const SKILL_SYS =
  'You are an Inbox & Calendar operator. Convert the request into a macro using ONLY these operations:\n' +
  'find_email(query) -> thread; compose_email(to, subject, body); reply_email(thread, body); ' +
  'forward_email(thread, to, note); archive_email(thread); label_email(thread, label); ' +
  'schedule_send(to, subject, body, when); create_event(title, start, end, remind_min); ' +
  'set_reminder(text, when); find_slot(duration_min, after, before) -> slot; rsvp(event, response).\n' +
  'Output ONLY the macro, one call per line, no prose. If the request is outside inbox or calendar, output exactly: OUT_OF_SCOPE.';
// Each pair is [request, macro]. Diverse on purpose so the LoRA learns the grammar
// of the action space (generalizes to unseen requests), not specific phrasings.
const GUIDED = [
  ['email my mom and book a calendar event to remind me to respond',
    'compose_email(to="mom", subject="Hi mom", body="Just checking in — talk soon!")\ncreate_event(title="Respond to mom", start="tomorrow 09:00", end="tomorrow 09:15", remind_min=10)'],
  ['reply to the thread from Sarah declining the meeting, then remind me Friday to follow up',
    't = find_email(query="from:Sarah meeting")\nreply_email(thread=t, body="Thanks for the invite — I can\'t make this one, but let\'s find another time.")\nset_reminder(text="Follow up with Sarah", when="Friday 09:00")'],
  ['forward the invoice email from billing to my accountant',
    't = find_email(query="from:billing invoice")\nforward_email(thread=t, to="accountant", note="For the books — thanks!")'],
  ['schedule a 30 minute focus block tomorrow afternoon',
    's = find_slot(duration_min=30, after="tomorrow 13:00", before="tomorrow 18:00")\ncreate_event(title="Focus block", start=s.start, end=s.end, remind_min=5)'],
  ['draft a thank-you note to the team and send it at 8am tomorrow',
    'schedule_send(to="team", subject="Thank you", body="Great work this week — thank you all.", when="tomorrow 08:00")'],
  ['archive all the newsletters in my inbox',
    't = find_email(query="category:newsletters")\narchive_email(thread=t)'],
  ['set up a 1:1 with Alex next Tuesday at 2pm for 25 minutes',
    'create_event(title="1:1 with Alex", start="next Tuesday 14:00", end="next Tuesday 14:25", remind_min=10)'],
  ['reply yes to the standup invite and add it to my calendar',
    't = find_email(query="standup invite")\nrsvp(event=t, response="yes")'],
  ['find John\'s last email and reply that I\'ll review it by Thursday',
    't = find_email(query="from:John")\nreply_email(thread=t, body="Thanks — I\'ll review this and get back to you by Thursday.")'],
  ['label the email from the landlord as housing and remind me to respond tonight',
    't = find_email(query="from:landlord")\nlabel_email(thread=t, label="housing")\nset_reminder(text="Respond to landlord", when="today 19:00")'],
  ['book lunch with Priya Thursday noon and email her the invite',
    'create_event(title="Lunch with Priya", start="Thursday 12:00", end="Thursday 13:00", remind_min=30)\ncompose_email(to="Priya", subject="Lunch Thursday", body="Sent you a calendar invite for Thursday noon — looking forward to it!")'],
  ['clear my unread promotions and remind me to check email after lunch',
    't = find_email(query="is:unread category:promotions")\narchive_email(thread=t)\nset_reminder(text="Check email", when="today 13:30")'],
  ['order me a pizza', 'OUT_OF_SCOPE'],
  ['what is the capital of France?', 'OUT_OF_SCOPE'],
];
// An UNSEEN request (not in the set) — proves the macro grammar generalizes.
const GUIDED_SUGGEST = "Email the design team this week's notes, then put a 30-minute review on my calendar for Monday morning.";
let trainLosses = [];

// ── status rail: the single place that surfaces model state ───────────────────
function setBadge() {
  const rail = $('rail'), chip = $('railChip');
  if (!rail || !chip) return;
  if (state.err) { rail.dataset.state = 'err'; chip.textContent = 'Load failed'; return; }
  if (state.busy === 'load') { rail.dataset.state = 'busy'; chip.textContent = 'Loading…'; return; }
  if (!state.loaded) { rail.dataset.state = 'idle'; chip.textContent = 'Model not loaded'; return; }
  const sel = $('adapterSel')?.value || 'none';
  if (sel === 'none') { rail.dataset.state = 'ok'; chip.textContent = 'Live · base'; }
  else { rail.dataset.state = 'tuned'; chip.textContent = 'Live · tuned: ' + sel; }
}
function lockInference(on) {
  $('inferLock').style.display = on ? 'flex' : 'none';
  $('run').disabled = on || !state.loaded || state.busy === 'gen';
}
function gateButtons() {
  const ready = state.loaded && !state.busy;
  $('run').disabled = !ready;
  $('trainGuided').disabled = !ready;
  $('trainOwn').disabled = !ready || !ownExamples().length;
  for (const id of ['load', 'loadHF']) $(id).disabled = !!state.busy;
  // progressive disclosure: Step 2 (ask) stays hidden entirely until the model loads
  const ask = $('askSection');
  if (ask) ask.hidden = !state.loaded;
}

// ── model load ───────────────────────────────────────────────────────────────
async function loadWith(reader, label) {
  if (state.busy) return;
  state.busy = 'load'; state.err = null; setBadge(); gateButtons();
  try {
    await session.loadWith(reader, label);
    state.loaded = true;
    log('Model ready. Ask it anything below — or hit Train to teach it something new.');
  } catch (e) {
    state.err = e.message;
    log('Load error: ' + e.message);
    console.error(e);
  } finally {
    state.busy = false; setBadge(); gateButtons();
  }
}

// ── inference ─────────────────────────────────────────────────────────────────
function buildMessages(userText) {
  const sel = $('adapterSel')?.value || 'none';
  if (sel !== 'none' && state.tuned && state.tuned.name === sel) return state.tuned.build(userText);
  // Recommended usage is user-only: the chat template injects the default
  // system prompt. A custom "be concise" system message would suppress the
  // long chain-of-thought that is this model's whole strength.
  return [{ role: 'user', content: userText }];
}
async function runInference() {
  if (!state.loaded || state.busy) return;
  const userText = $('prompt').value.trim();
  if (!userText) { log('type something to ask first'); return; }
  state.busy = 'gen'; gateButtons();
  const sel = $('adapterSel')?.value || 'none';
  adapters.applyToRuntime(sel, session.rt);
  const out = $('out');
  out.textContent = '';
  const node = document.createTextNode('');
  out.appendChild(node);
  const st = steps('inferSteps'); st.reset();
  const cap = $('inferCap');
  const stop = startClock('inferClock');
  $('inferProc').classList.add('on');
  st.active('tok'); cap.textContent = 'Tokenizing your prompt with the VibeThinker tokenizer…';
  const t0 = performance.now();
  let n = 0, first = true;
  try {
    const msgs = buildMessages(userText);
    st.done('tok'); st.active('prefill'); cap.textContent = 'Reading the prompt into the KV cache (prefill)…';
    for await (const d of session.generate(msgs, { maxTokens: GEN.maxTokens, temperature: GEN.temperature, topP: GEN.topP, topK: GEN.topK })) {
      if (first) { first = false; st.done('prefill'); st.active('decode'); cap.textContent = 'Generating the answer one token at a time…'; }
      node.appendData(d); n++;
      $('tokps').textContent = `${n} tok · ${(n / ((performance.now() - t0) / 1000)).toFixed(1)} tok/s`;
      out.scrollTop = out.scrollHeight;
    }
    const dt = (performance.now() - t0) / 1000;
    $('tokps').textContent = `${n} tok · ${(n / dt).toFixed(1)} tok/s · ${dt.toFixed(1)}s`;
    st.done('prefill'); st.done('decode'); st.done('done');
    cap.textContent = `Done — ${sel === 'none' ? 'base model' : 'tuned adapter "' + sel + '"'}.`;
    log(`done (${sel === 'none' ? 'base model' : 'tuned adapter'}).`);
  } catch (e) {
    out.appendData('\n\n[error] ' + e.message); cap.textContent = 'error: ' + e.message; console.error(e);
  } finally {
    stop(); $('inferProc').classList.remove('on'); state.busy = false; gateButtons();
  }
}

// ── training: shared runner ───────────────────────────────────────────────────
async function runTraining({ examples, lr, epochs, accum, base, kind, system, build, suggest }) {
  if (!state.loaded) { log('load the model first (INFERENCE pane).'); switchTab('infer'); return; }
  if (state.busy) return;
  const name = uniqueName(base);
  const runId = store.newId();
  state.busy = 'train';
  lockInference(true); gateButtons();
  $('trainWidget').style.display = '';
  resetTrainTelemetry();
  const windows = Math.max(1, Math.ceil(examples.length / accum));
  const total = windows * epochs;
  let lastLoss = null;
  const ctrl = new TrainingController({
    session, adapters, log: () => {},
    trainerOptions: { lr, maxTrainSeq: 384, lmHeadBlock: 128, maxGradNorm: 1.0, weightDecay: 0.0, warmupSteps: Math.min(4, total), totalSteps: total, gradAccumSteps: accum },
  });
  const st = steps('trainSteps'); st.reset();
  const cap = $('trainCap');
  const stop = startClock('trainClock');
  st.active('prep'); cap.textContent = 'Building masked, shifted-label examples and tokenizing on the GPU…';
  renderMaskPreview(ctrl, examples[0]);
  ctrl.initAdapter(name, { rank: 16, alpha: 32 });
  trainProgress(0, total, null, 'warming up…');
  const t0 = performance.now();
  try {
    st.done('prep'); st.loop(['fwd', 'bwd', 'opt'], true);
    cap.textContent = 'Looping forward → backward → AdamW over your examples (full-network backprop)…';
    await ctrl.train(examples, {
      epochs,
      onStep: (r) => {
        const { step, loss } = r;
        lastLoss = loss;
        updateTrainTelemetry(step, total, r);
        trainProgress(step, total, loss, `teaching · step ${step}/${total} · loss ${loss.toFixed(3)} · ${fmtNum(r.trainTokPerSec)} tok/s`);
        cap.textContent =
          `Step ${step}/${total} — forward ${fmtMs(r.microStepMs)} → backward → AdamW ${fmtMs(r.optimizerStepMs)} · loss ${loss.toFixed(3)}`;
      },
    });
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    st.loop(['fwd', 'bwd', 'opt'], false); st.done('fwd'); st.done('bwd'); st.done('opt');
    st.active('swap');
    state.tuned = { name, kind, build, suggest, ctrl };
    state.activeRunId = runId;
    addAdapterOption(name);
    $('adapterSel').value = name;
    st.done('swap');
    trainProgress(total, total, null, `done in ${dt}s — adapter "${name}" is live`);
    cap.textContent = `Adapter "${name}" hot-swapped into inference — live. Trained in ${dt}s.`;
    $('downloadAdapter').style.display = '';
    showTryIt(suggest);
    // persist this attempt so it survives reloads and shows in the history rail
    try {
      const files = await exportLoraAdapter(ctrl.trainer, { name });
      await store.saveRun(
        { id: runId, name, base, kind, system: system || null, suggest: suggest || '',
          createdAt: Date.now(), steps: total, epochs, durationSec: +dt, finalLoss: lastLoss, rank: 16, alpha: 32 },
        { safetensors: files.safetensors, configJson: files.configJson },
      );
      renderHistory();
    } catch (e) { console.warn('[history] save failed', e); }
    log(`Trained "${name}" in ${dt}s. Saved to your fine-tunes; switch to Inference to try it.`);
  } catch (e) {
    st.loop(['fwd', 'bwd', 'opt'], false);
    trainProgress(0, total, null, 'training error: ' + e.message);
    cap.textContent = 'error: ' + e.message;
    console.error(e);
  } finally {
    stop();
    state.busy = false;
    lockInference(false); gateButtons();
  }
}

// ── BYOD: turn text into short "continue the note" recall examples ────────────
const MAX_CHARS = 12000, MAX_CHUNKS = 24, MIN_WORDS = 12, HEAD_WORDS = 6;
function chunkText(text) {
  text = (text || '').replace(/\r/g, '').slice(0, MAX_CHARS);
  const paras = text.split(/\n{2,}|\.(?=\s)/).map((s) => s.trim()).filter(Boolean);
  const out = [];
  for (const p of paras) {
    const words = p.split(/\s+/).filter(Boolean);
    if (words.length < MIN_WORDS) continue;
    const head = words.slice(0, HEAD_WORDS).join(' ');
    const rest = words.slice(HEAD_WORDS).join(' ');
    out.push({ head, rest, full: p });
    if (out.length >= MAX_CHUNKS) break;
  }
  return out;
}
let _ownChunks = [];
function ownExamples() {
  return _ownChunks.map((c) => ({ messages: [{ role: 'user', content: c.head }], completion: ' ' + c.rest }));
}
function refreshOwn() {
  const text = $('ownText').value;
  _ownChunks = chunkText(text);
  const chars = Math.min(MAX_CHARS, (text || '').length);
  $('ownStats').textContent = _ownChunks.length
    ? `${_ownChunks.length} snippet(s) · ${chars} chars (cap ${MAX_CHARS}) · ready to teach`
    : `paste/drop at least one paragraph (~${MIN_WORDS}+ words). 100% local.`;
  gateButtons();
}

// ── small UI helpers ──────────────────────────────────────────────────────────
function switchTab(which) {
  const infer = which === 'infer';
  $('paneInfer').classList.toggle('active', infer);
  $('paneTrain').classList.toggle('active', !infer);
  $('tabInfer').classList.toggle('on', infer);
  $('tabTrain').classList.toggle('on', !infer);
}
function addAdapterOption(name) {
  const sel = $('adapterSel');
  if (![...sel.options].some((o) => o.value === name)) {
    const o = document.createElement('option');
    o.value = name; o.textContent = name; sel.appendChild(o);
  }
  // reveal the adapter picker only once there's something to pick
  const wrap = $('adapterWrap');
  if (wrap) wrap.hidden = false;
}
function trainProgress(step, total, loss, label) {
  $('trainBar').style.width = (100 * step / Math.max(1, total)).toFixed(1) + '%';
  $('trainLabel').textContent = label;
}
function resetTrainTelemetry() {
  trainLosses = [];
  const box = $('trainMetrics');
  if (box) box.hidden = false;
  for (const [id, v] of [['tmLoss', '—'], ['tmTokps', '—'], ['tmActive', '—'], ['tmOpt', '—']]) {
    const el = $(id);
    if (el) el.textContent = v;
  }
  const line = $('lossLine');
  if (line) line.setAttribute('points', '');
  const preview = $('maskPreview');
  if (preview) preview.hidden = true;
}
function updateTrainTelemetry(step, total, r) {
  trainLosses.push(r.loss);
  $('tmLoss').textContent = r.loss.toFixed(4);
  $('tmTokps').textContent = `${fmtNum(r.trainTokPerSec)} tok/s`;
  $('tmActive').textContent = `${r.numActive || 0} / ${r.tokens || 0}`;
  $('tmOpt').textContent = fmtMs(r.optimizerStepMs);
  drawLossSpark();
}
function drawLossSpark() {
  const line = $('lossLine');
  if (!line || trainLosses.length < 2) return;
  const min = Math.min(...trainLosses);
  const max = Math.max(...trainLosses);
  const span = Math.max(1e-6, max - min);
  const points = trainLosses
    .map((v, i) => {
      const x = (i / Math.max(1, trainLosses.length - 1)) * 300;
      const y = 36 - ((v - min) / span) * 32;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  line.setAttribute('points', points);
}
function renderMaskPreview(ctrl, example) {
  const box = $('maskPreview');
  const rows = $('maskRows');
  if (!box || !rows || !example) return;
  try {
    const preview = ctrl.inspectExample(example);
    $('maskSummary').textContent =
      `${preview.tokens.length} tokens · ${preview.trainPositions} trained next-token labels`;
    const shown = preview.rows.slice(0, 96);
    rows.innerHTML =
      '<div class="hdr">pos</div><div class="hdr">segment</div><div class="hdr">token</div><div class="hdr target">trained target</div>' +
      shown
        .map((r) => {
          const cls = `${r.trainsNext ? 'train' : ''} ${r.segment}`;
          const target = r.trainsNext ? `${r.targetId} ${clip(r.targetText, 24)}` : '';
          return `<div class="${cls}">${r.index}</div><div class="${cls}">${esc(r.segment)}</div><div class="${cls}">${r.id} ${esc(clip(r.text, 28))}</div><div class="${cls} target">${esc(target)}</div>`;
        })
        .join('') +
      (preview.rows.length > shown.length ? `<div class="prompt">…</div><div class="prompt">truncated</div><div class="prompt">${preview.rows.length - shown.length} more rows</div><div class="prompt target"></div>` : '');
    box.hidden = false;
  } catch (e) {
    rows.innerHTML = `<div class="prompt">preview</div><div class="prompt">error</div><div class="prompt">${esc(e.message)}</div><div class="prompt target"></div>`;
    box.hidden = false;
  }
}
function showTryIt(suggest) {
  const t = $('tryIt');
  t.style.display = 'flex';
  $('tryItBtn').onclick = () => {
    switchTab('infer');
    $('adapterSel').value = state.tuned.name; setBadge();
    $('prompt').value = suggest;
    runInference();
  };
}

// ── history rail: every saved fine-tune, persisted across reloads ─────────────
function uniqueName(base) {
  const taken = new Set(store.listRuns().map((r) => r.name));
  if (!taken.has(base)) return base;
  let i = 2;
  while (taken.has(`${base} #${i}`)) i++;
  return `${base} #${i}`;
}
function buildFromMeta(meta) {
  return meta.system
    ? (u) => [{ role: 'system', content: meta.system }, { role: 'user', content: u }]
    : (u) => [{ role: 'user', content: u }];
}
function fmtRunMeta(m) {
  const parts = [];
  if (m.finalLoss != null) parts.push('loss ' + Number(m.finalLoss).toFixed(3));
  if (m.steps) parts.push(m.steps + ' steps');
  if (m.durationSec != null) parts.push(Math.round(m.durationSec) + 's');
  try { parts.push(new Date(m.createdAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })); } catch {}
  return parts.join(' · ');
}
function renderHistory() {
  const runs = store.listRuns();
  $('historyCount').textContent = String(runs.length);
  $('historyEmpty').style.display = runs.length ? 'none' : '';
  const ul = $('historyList');
  ul.innerHTML = '';
  for (const m of runs) {
    const li = document.createElement('li');
    li.className = 'hrun' + (m.id === state.activeRunId ? ' active' : '');
    li.dataset.id = m.id;
    li.dataset.kind = m.kind || 'own';
    li.innerHTML =
      `<span class="hrun__led"></span>` +
      `<div class="hrun__name" title="${esc(m.name)}">${esc(m.name)}</div>` +
      `<div class="hrun__meta">${esc(fmtRunMeta(m))}</div>` +
      `<div class="hrun__acts">` +
      `<button data-act="apply" class="tiny primary">▶ Use</button>` +
      `<button data-act="export" class="tiny secondary" title="Export adapter">⬇</button>` +
      `<button data-act="del" class="tiny danger" title="Delete">✕</button>` +
      `</div>`;
    li.querySelector('[data-act=apply]').onclick = (e) => { e.stopPropagation(); applyRun(m.id); };
    li.querySelector('[data-act=export]').onclick = (e) => { e.stopPropagation(); exportRun(m.id); };
    li.querySelector('[data-act=del]').onclick = (e) => { e.stopPropagation(); delRun(m.id); };
    li.onclick = () => applyRun(m.id);
    ul.appendChild(li);
  }
  renderKnife();
}

// ── the Swiss army knife: trained skills as equippable slots ─────────────────
const SKILL_ICON = { guided: '⚔', own: '📜' };
function skillLevel(m) {
  // Level grows with training amount; XP fills as loss drops (cosmetic, bounded).
  const lv = Math.max(1, Math.min(9, Math.round((m.steps || 12) / 12)));
  const loss = m.finalLoss == null ? 1.5 : Number(m.finalLoss);
  const xp = Math.max(6, Math.min(100, Math.round((100 * (3 - loss)) / 3)));
  return { lv, xp };
}
function renderKnife() {
  const slots = $('knifeSlots');
  if (!slots) return;
  const runs = store.listRuns();
  slots.innerHTML = '';
  // Always advertise the flagship if it hasn't been forged yet.
  const hasInbox = runs.some((r) => (r.base || '').startsWith('inbox-calendar') || /inbox-calendar/.test(r.name));
  if (!hasInbox) {
    const lock = document.createElement('div');
    lock.className = 'kslot kslot--locked';
    lock.title = 'Forge the Inbox & Calendar skill in the Train tab';
    lock.innerHTML = `<span class="kslot__icon">＋</span><span class="kslot__name">Inbox &amp; Calendar</span>` +
      `<span class="kslot__lv">locked · train to forge</span>`;
    lock.onclick = () => { switchTab('train'); };
    slots.appendChild(lock);
  }
  if (!runs.length && hasInbox) return;
  for (const m of runs) {
    const { lv, xp } = skillLevel(m);
    const el = document.createElement('div');
    el.className = 'kslot' + (m.id === state.activeRunId ? ' equipped' : '');
    el.title = `${m.name} — click to equip (hot-swap into inference)`;
    el.innerHTML =
      `<span class="kslot__icon">${SKILL_ICON[m.kind] || '🗡'}</span>` +
      `<span class="kslot__name">${esc(m.name)}</span>` +
      `<span class="kslot__lv">Lv ${lv}${m.id === state.activeRunId ? ' · equipped' : ''}</span>` +
      `<span class="kslot__xp"><i style="width:${xp}%"></i></span>`;
    el.onclick = () => applyRun(m.id);
    slots.appendChild(el);
  }
}
async function applyRun(id) {
  const meta = store.getRun(id);
  if (!meta) return;
  if (!state.loaded) { log('Load VibeThinker-3B first (Step 1), then tap a fine-tune to use it.'); switchTab('infer'); return; }
  if (state.busy) return;
  state.busy = 'apply'; gateButtons();
  try {
    log(`Applying "${meta.name}"…`);
    let adapter = adapters.get(meta.name);
    if (!adapter) {
      const files = await store.loadRunFiles(id);
      adapter = await loadLoraAdapterGPU(session.rt.dev, files, QWEN25_3B);
      adapter.name = meta.name;
      adapters.adapters[meta.name] = adapter;
    }
    addAdapterOption(meta.name);
    state.tuned = { name: meta.name, kind: meta.kind, build: buildFromMeta(meta), suggest: meta.suggest };
    state.activeRunId = id;
    $('adapterSel').value = meta.name;
    setBadge(); renderHistory();
    switchTab('infer');
    if (meta.suggest) $('prompt').value = meta.suggest;
    log(`Now serving fine-tune "${meta.name}". Ask away.`);
  } catch (e) {
    log('Could not apply: ' + e.message); console.error(e);
  } finally {
    state.busy = false; gateButtons();
  }
}
async function exportRun(id) {
  const meta = store.getRun(id);
  if (!meta) return;
  try {
    const { safetensors, configJson } = await store.getRunBlobs(id);
    const stem = (meta.name || 'adapter').replace(/[^\w.-]+/g, '_');
    if (state.dirHandle && (await store.ensurePermission(state.dirHandle))) {
      await store.writeFileToDir(state.dirHandle, stem + '.safetensors', safetensors);
      await store.writeFileToDir(state.dirHandle, stem + '.adapter_config.json', configJson);
      log(`Saved "${meta.name}" to your connected folder.`);
    } else {
      triggerBlob(safetensors, stem + '.safetensors');
      triggerBlob(new Blob([configJson], { type: 'application/json' }), stem + '.adapter_config.json');
      log(`Exported "${meta.name}".`);
    }
  } catch (e) { log('Export failed: ' + e.message); }
}
async function delRun(id) {
  await store.deleteRun(id);
  if (state.activeRunId === id) state.activeRunId = null;
  renderHistory();
}
function triggerBlob(data, filename) {
  const blob = data instanceof Blob ? data : new Blob([data]);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function fmtMs(ms) {
  return Number.isFinite(ms) ? `${ms.toFixed(ms >= 100 ? 0 : 1)}ms` : '—';
}
function fmtNum(n) {
  return Number.isFinite(n) ? (n >= 100 ? n.toFixed(0) : n.toFixed(1)) : '—';
}
function clip(s, n) {
  s = String(s ?? '').replace(/\s+/g, ' ');
  return s.length > n ? s.slice(0, Math.max(0, n - 1)) + '…' : s;
}

// ── layout modes (desktop / mobile / foldable-open) ───────────────────────────
function applyLayout() {
  const mq = (q) => { try { return window.matchMedia(q).matches; } catch { return false; } };
  const fold = mq('(horizontal-viewport-segments: 2)') || mq('(spanning: single-fold-vertical)');
  const mobile = mq('(max-width: 700px)');
  document.body.dataset.layout = fold ? 'foldable' : mobile ? 'mobile' : 'desktop';
}

// ── File System Access: connect a folder for import + export/save ─────────────
async function initFs() {
  if (!store.fsSupported) { $('fsBlock').hidden = true; return; }
  $('fsBlock').hidden = false;
  const setDir = (h) => {
    state.dirHandle = h;
    $('fsForget').hidden = false;
    $('ownImportDir').hidden = false;
    $('fsStatus').textContent = `connected: ${h.name || 'folder'} — adapters can save here; import text below.`;
  };
  try { const saved = await store.savedDirectory(); if (saved) setDir(saved); } catch {}
  $('fsConnect').onclick = async () => {
    try { setDir(await store.connectDirectory()); }
    catch (e) { if (e.name !== 'AbortError') log('folder: ' + e.message); }
  };
  $('fsForget').onclick = async () => {
    await store.forgetDirectory();
    state.dirHandle = null;
    $('fsForget').hidden = true; $('ownImportDir').hidden = true;
    $('fsStatus').textContent = 'not connected — import training text & save adapters straight to a folder you pick.';
  };
  $('ownImportDir').onclick = async () => {
    if (!state.dirHandle) return;
    if (!(await store.ensurePermission(state.dirHandle, 'read'))) { log('permission denied for folder'); return; }
    try {
      const { text, names } = await store.readDirText(state.dirHandle);
      if (!text.trim()) { $('ownStats').textContent = 'no .txt/.md/.json/.csv files found in that folder'; return; }
      $('ownText').value = (text + '\n' + $('ownText').value).slice(0, MAX_CHARS);
      refreshOwn();
      $('ownStats').textContent = `imported ${names.length} file(s) · ` + $('ownStats').textContent;
    } catch (e) { log('import failed: ' + e.message); }
  };
}

// ── wiring ────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  // render guided facts list
  $('guidedList').innerHTML = GUIDED.map(([q, a]) =>
    `<li><span class="skill-req">${esc(q)}</span><pre class="skill-macro">${esc(a)}</pre></li>`).join('');

  $('tabInfer').onclick = () => switchTab('infer');
  $('tabTrain').onclick = () => switchTab('train');
  $('gear').onclick = () => {
    const open = $('settings').hidden;
    $('settings').hidden = !open;
    $('gear').classList.toggle('on', open);
  };
  $('adapterSel').onchange = setBadge;

  $('load').onclick = () => loadWith(urlReader($('modelUrl').value.trim()), $('modelUrl').value.trim());
  $('loadHF').onclick = () => {
    const repo = $('hfRepo').value.trim();
    const token = ($('hfToken')?.value || '').trim();
    if (!repo) return log('enter a Hugging Face repo id, e.g. WeiboAI/VibeThinker-3B');
    loadWith(hfReader(repo, token), 'HF: ' + repo);
  };
  $('modelFiles').onchange = (ev) => {
    const files = [...ev.target.files];
    if (!files.length) return;
    const map = {}; for (const f of files) map[f.name] = f;
    loadWith(fileReader(map), `${files.length} local files`);
  };

  $('run').onclick = runInference;
  $('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runInference(); });

  $('trainGuided').onclick = () => runTraining({
    examples: GUIDED.map(([q, a]) => ({ messages: [{ role: 'system', content: SKILL_SYS }, { role: 'user', content: q }], completion: ' ' + a })),
    lr: 3e-4, epochs: 12, accum: 2, base: 'inbox-calendar', kind: 'guided', system: SKILL_SYS,
    build: (u) => [{ role: 'system', content: SKILL_SYS }, { role: 'user', content: u }],
    suggest: GUIDED_SUGGEST,
  });

  $('ownText').addEventListener('input', refreshOwn);
  $('ownFiles').onchange = async (ev) => {
    const files = [...ev.target.files].slice(0, 5);
    let txt = '';
    for (const f of files) { try { txt += (await f.text()) + '\n\n'; } catch {} }
    $('ownText').value = (txt + '\n' + $('ownText').value).slice(0, MAX_CHARS);
    refreshOwn();
  };
  $('ownFetch').onclick = async () => {
    const url = $('ownUrl').value.trim();
    if (!url) return;
    $('ownStats').textContent = 'fetching readable text via reader proxy…';
    try {
      const r = await fetch('https://r.jina.ai/' + url);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const t = await r.text();
      $('ownText').value = t.slice(0, MAX_CHARS);
      refreshOwn();
    } catch (e) { $('ownStats').textContent = 'could not fetch (CORS/blocked) — paste the text instead. ' + e.message; }
  };
  $('trainOwn').onclick = () => {
    const ex = ownExamples();
    if (!ex.length) return;
    const windows = Math.ceil(ex.length / 2);
    runTraining({
      examples: ex, lr: 3e-4, accum: 2,
      epochs: Math.max(3, Math.min(8, Math.round(50 / windows))),
      base: 'my-notes', kind: 'own', system: null,
      build: (u) => [{ role: 'user', content: u }],
      suggest: _ownChunks[0]?.head || '',
    });
  };

  $('downloadAdapter').onclick = () => { if (state.tuned?.ctrl?.trainer) downloadLoraAdapter(state.tuned.ctrl.trainer, { name: state.tuned.name }); };

  // layout modes (real layout switch, not just CSS breakpoints)
  applyLayout();
  for (const q of ['(max-width: 700px)', '(horizontal-viewport-segments: 2)', '(spanning: single-fold-vertical)']) {
    try { window.matchMedia(q).addEventListener('change', applyLayout); } catch {}
  }
  window.__layout = (m) => { document.body.dataset.layout = m; }; // test/devtools hook
  window.__eg = { store, renderHistory, renderKnife, applyRun, exportRun, delRun, state }; // devtools/test surface

  initFs();
  renderHistory();
  switchTab('infer'); setBadge(); refreshOwn(); gateButtons();
});

function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
