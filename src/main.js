/*
 * Emberglass — browser harness for the custom WebGPU VibeThinker-3B runtime.
 *
 * This file implements the **engine harness** only.
 * Per Saturday review.MD (the active control document):
 *   - The approved product is the three-screen loop: Skillbook/Home, Skill/Train Surface, Job Board.
 *   - The full UI Reset Card is explicitly a later/separate pass.
 *   - Current work focuses on the 4 Recovery Work Cards + Later Fixes, under the Recovery Contract.
 *   - Only real model weights, real declared training sources, real accounts (when supplied), and
 *     real browser runs are acceptable. No local substitutes.
 *
 * All writes are dry-run only (executors_are_dry_run ratchet) until the action layer is designed.
 */
import { QWEN25_3B } from './config.js';
import { urlReader, hfReader, fileReader } from './readers.js';
import { AdapterRegistry } from './services/adapter_registry.js';
import { ModelSession } from './services/model_session.js';
import { TrainingController } from './services/training_controller.js';
import { downloadLoraAdapter, exportLoraAdapter } from './lora_export.js';
import { loadLoraAdapterGPU } from './lora_gpu.js';
import * as store from './services/store.js';
import { SKILLS, POPULAR_2026, verifyMacro, planFor, dryRun } from './skills.js';
import { ICON_THEME_PRESETS, iconTheme, paintSkillIcon, setIconTheme, themedTileColor } from './icon_pipeline.js';

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

// ── action spaces + macro verifier live in ./skills.js (pure, Node-testable) ──
// SKILLS is the trained-surface registry; POPULAR_2026 is the dock catalog of
// account/app roots; verifyMacro is the "does what we say" gate.
const skillByKey = (key) => SKILLS.find((s) => key && (key === s.key || String(key).startsWith(s.key + ' ')));
let selectedSkillKey = SKILLS[0].key;
let trainLosses = [];

// Pick up to `n` examples for a single forge: always keep the OUT_OF_SCOPE pairs
// (so the adapter still learns to bounce), then fill with a deterministic spread
// of the in-scope macros for variety without making each train run enormous.
function sampleExamples(all, n) {
  const oos = all.filter(([, a]) => a === 'OUT_OF_SCOPE');
  const inscope = all.filter(([, a]) => a !== 'OUT_OF_SCOPE');
  const keep = Math.max(0, n - oos.length);
  const stride = Math.max(1, Math.floor(inscope.length / Math.max(1, keep)));
  const picked = [];
  for (let i = 0; i < inscope.length && picked.length < keep; i += stride) picked.push(inscope[i]);
  return [...picked, ...oos];
}

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
    log('Model ready. Train an account surface or equip a chain to execute writes.');
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
  setMacroCheck(null);
  st.active('tok'); cap.textContent = 'Tokenizing your prompt with the VibeThinker tokenizer…';
  const t0 = performance.now();
  let n = 0, first = true, acc = '';
  try {
    const msgs = buildMessages(userText);
    st.done('tok'); st.active('prefill'); cap.textContent = 'Reading the prompt into the KV cache (prefill)…';
    for await (const d of session.generate(msgs, { maxTokens: GEN.maxTokens, temperature: GEN.temperature, topP: GEN.topP, topK: GEN.topK })) {
      if (first) { first = false; st.done('prefill'); st.active('decode'); cap.textContent = 'Generating the answer one token at a time…'; }
      node.appendData(d); acc += d; n++;
      $('tokps').textContent = `${n} tok · ${(n / ((performance.now() - t0) / 1000)).toFixed(1)} tok/s`;
      out.scrollTop = out.scrollHeight;
    }
    const dt = (performance.now() - t0) / 1000;
    $('tokps').textContent = `${n} tok · ${(n / dt).toFixed(1)} tok/s · ${dt.toFixed(1)}s`;
    st.done('prefill'); st.done('decode'); st.done('done');
    cap.textContent = `Done — ${sel === 'none' ? 'base model' : 'tuned adapter "' + sel + '"'}.`;
    // "does what we say" gate: if a service surface is equipped, verify the macro.
    const skill = sel !== 'none' && state.tuned && state.tuned.name === sel ? skillByKey(state.tuned.base) : null;
    if (skill) {
      const res = verifyMacro(acc, skill.spec);
      setMacroCheck(res, skill, acc);
      if (res.status === 'ok') stageMsg(`Write resolved — compiled a ${res.n}-step plan on ${skill.label}.`);
      else if (res.status === 'oos') stageMsg(`That request is outside the ${skill.label} surface. Try one of its writes.`);
      else stageMsg(`The plan didn't validate — adjust the request and try again.`);
      if (state.activeRunId) { bumpUses(state.activeRunId); renderDock(); }
    }
    log(`done (${sel === 'none' ? 'base model' : 'tuned adapter'}).`);
  } catch (e) {
    out.appendData('\n\n[error] ' + e.message); cap.textContent = 'error: ' + e.message; console.error(e);
  } finally {
    stop(); $('inferProc').classList.remove('on'); state.busy = false; gateButtons();
  }
}

// ── training: shared runner ───────────────────────────────────────────────────
async function runTraining({ examples, lr, epochs, accum, base, kind, system, build, suggest }) {
  if (!state.loaded) { log('Boot the engine first, then train a surface.'); closeTrainer(); return; }
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
    state.tuned = { name, kind, base, build, suggest, ctrl };
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
    log(`Trained "${name}" in ${dt}s. Saved to your Atlas; equip it to try the write surface.`);
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
// Single-view world: there are no tabs. Training opens as a menu over the atlas;
// everything else lives on one screen.
function openTrainer() {
  const t = $('trainer'); if (!t) return;
  renderSkillPicker(); selectSkill(selectedSkillKey);
  t.hidden = false; document.body.classList.add('modal-open');
  $('gear')?.classList.remove('on'); $('settings') && ($('settings').hidden = true);
}
function closeTrainer() {
  const t = $('trainer'); if (t) t.hidden = true;
  document.body.classList.remove('modal-open');
}
// kept for back-compat with existing call sites: 'train' opens the training menu,
// anything else returns to the single game view.
function switchTab(which) { which === 'train' ? openTrainer() : closeTrainer(); }
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
  renderEquipPanel();
  if (state.tuned?.name) stageMsg(`New surface trained: “${state.tuned.name}” — it was added to your Atlas. Equip it into a chain to act.`);
}

// ── equipped surface panel (Inference): shows writes + one-tap "drills" ───────
// When a service surface is equipped, the inference pane stops being a blank chat
// box: it surfaces the action space and a few example requests to fire.
function renderEquipPanel() {
  const bar = $('equipBar');
  if (!bar) return;
  const skill = state.tuned ? skillByKey(state.tuned.base) : null;
  if (!skill || !skill.spec) { bar.hidden = true; return; }
  bar.hidden = false;
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  paintIcon($('equipIcon'), dockOf(skill.key), skill.icon, 0.85);
  set('equipName', `${skill.label} surface`);
  set('equipScope', `surface: ${skill.spec.scope}`);
  const ops = $('equipOps');
  if (ops) {
    ops.innerHTML = '';
    for (const op of skill.spec.ops) {
      const c = document.createElement('span');
      c.className = 'equip__op';
      c.textContent = op.name;
      c.title = `${op.name}(${(op.params || []).join(', ')})`;
      ops.appendChild(c);
    }
  }
  const host = $('equipDrills');
  if (host) {
    host.innerHTML = '';
    const inscope = skill.examples.filter(([, a]) => a !== 'OUT_OF_SCOPE');
    const step = Math.max(1, Math.floor(inscope.length / 4));
    const picks = [];
    for (let i = 0; i < inscope.length && picks.length < 4; i += step) picks.push(inscope[i][0]);
    for (const q of picks) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'drill'; b.textContent = q; b.title = 'Fire this drill';
      b.onclick = () => { $('prompt').value = q; runInference(); };
      host.appendChild(b);
    }
  }
}

// Turn an emitted macro into a plain-English "battle plan" (op → key args).
function humanizePlan(text) {
  const out = [];
  for (const raw of String(text).split('\n')) {
    const line = raw.trim();
    if (!line || line === 'OUT_OF_SCOPE') continue;
    const m = line.match(/^(?:[A-Za-z_]\w*\s*=\s*)?([A-Za-z_]\w*)\s*\((.*)\)\s*;?\s*$/);
    if (!m) continue;
    const op = m[1].replace(/_/g, ' ');
    const args = [...m[2].matchAll(/([A-Za-z_]\w*)\s*=\s*"([^"]*)"/g)].map((x) => x[2]).filter(Boolean);
    const summary = args.slice(0, 2).join(' · ');
    out.push(summary ? `${op} — ${summary}` : op);
  }
  return out;
}

// ── atlas rail: every saved fine-tune, persisted across reloads ───────────────
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

  // Saturday review.MD — UI symptoms + "remove or hide complexity":
  // Adapter history competes for first-screen attention.
  // Default to collapsed. User can expand if needed.
  const histContainer = $('history') || $('historyRail') || $('historySection');
  if (histContainer && !histContainer.dataset.saturdayExpanded) {
    histContainer.style.display = 'none';
  }

  // Expose a minimal way to show it (for power users / future three-screen work)
  window.__egShowHistory = () => { if (histContainer) { histContainer.style.display = ''; histContainer.dataset.saturdayExpanded = '1'; } };
  for (const m of runs) {
    const { lv, xp } = skillLevel(m);
    const rar = rarityOf(lv);
    const active = m.id === state.activeRunId;
    const li = document.createElement('li');
    li.className = 'item' + (active ? ' active' : '');
    li.dataset.id = m.id;
    li.dataset.kind = m.kind || 'own';
    li.dataset.rarity = rar.key;
    li.title = `${m.name} — click to equip`;
    li.innerHTML =
      `<div class="item__frame"><span class="item__icon"></span><span class="item__lv">L${lv}</span></div>` +
      `<div class="item__body">` +
      `<div class="item__name">${esc(m.name)}</div>` +
      `<div class="item__rar">${rar.label} · ${esc(itemTypeLabel(m))}</div>` +
      `<div class="item__meta">${esc(fmtRunMeta(m))}</div>` +
      `<div class="item__xp"><i style="width:${xp}%"></i></div>` +
      `</div>` +
      (active ? `<div class="item__tag">EQUIPPED</div>` : '') +
      `<div class="item__acts">` +
      `<button data-act="apply" class="tiny primary">${active ? '✓ Equipped' : '▶ Equip'}</button>` +
      `<button data-act="export" class="tiny secondary" title="Export adapter">⬇</button>` +
      `<button data-act="del" class="tiny danger" title="Scrap">✕</button>` +
      `</div>`;
    paintIcon(li.querySelector('.item__icon'), runTile(m), runIcon(m), 0.76);
    li.querySelector('[data-act=apply]').onclick = (e) => { e.stopPropagation(); applyRun(m.id); };
    li.querySelector('[data-act=export]').onclick = (e) => { e.stopPropagation(); exportRun(m.id); };
    li.querySelector('[data-act=del]').onclick = (e) => { e.stopPropagation(); delRun(m.id); };
    li.onclick = () => applyRun(m.id);
    ul.appendChild(li);
  }
  renderDock();
  renderStage();
}

// ── the surface dock: trained surfaces as equippable slots ───────────────────
const SKILL_ICON = { guided: '⚔', own: '📜' };
const usesByRun = new Map(); // per-session count: how many times each surface was fired
function bumpUses(id) { usesByRun.set(id, (usesByRun.get(id) || 0) + 1); }
function runIcon(m) {
  const sk = skillByKey(m.base);
  return sk ? sk.icon : (SKILL_ICON[m.kind] || '🗡');
}
function runTile(m) {
  const sk = skillByKey(m.base);
  return sk ? dockOf(sk.key) : { ...BYOD_TILE, name: m.name, glyph: SKILL_ICON[m.kind] || '🗡' };
}
function skillLevel(m) {
  // Level grows with training amount; XP fills as loss drops (cosmetic, bounded).
  const lv = Math.max(1, Math.min(9, Math.round((m.steps || 12) / 12)));
  const loss = m.finalLoss == null ? 1.5 : Number(m.finalLoss);
  const xp = Math.max(6, Math.min(100, Math.round((100 * (3 - loss)) / 3)));
  return { lv, xp };
}
// Cosmetic rarity from level — pure flavor for the Atlas.
function rarityOf(lv) {
  if (lv >= 9) return { key: 'legendary', label: 'Legendary' };
  if (lv >= 7) return { key: 'epic', label: 'Epic' };
  if (lv >= 5) return { key: 'rare', label: 'Rare' };
  if (lv >= 3) return { key: 'uncommon', label: 'Uncommon' };
  return { key: 'common', label: 'Common' };
}
function itemTypeLabel(m) {
  const sk = skillByKey(m.base);
  if (sk) return sk.label;
  return m.kind === 'guided' ? 'Surface' : 'Custom surface';
}
// ── the dock: a bottom-anchored tray of account/app roots ─────────────────────
// Some services map to real trainable surfaces; the rest are the vision — "any
// app you're logged into" — shown as dimmed planned tiles. Tiles render from
// tiny glyph fallback metadata first, then upgrade to vendored SVG logos.
const BYOD_TILE = { bg: '#6b6256', fg: '#fff', glyph: '📜', fs: 20 };
// Per Saturday review.MD UI symptoms and "remove surfaces" guidance:
// The broad catalog (most of POPULAR_2026) is future vision and must not dominate the first experience.
// Only the core forgeable skills (those with .skill) are immediately relevant.
// The rest are collapsed behind an explicit "more" affordance.
const ALL_SERVICES = POPULAR_2026;
const CORE_SERVICES = ALL_SERVICES.filter(s => s.skill);
const SERVICES = CORE_SERVICES; // start minimal; full armory is secondary
let showFullDock = false;
let dockRuns = []; // run ids in dock order — the source of truth for number-key equip
let justEquippedId = null; // run id that should play the one-shot equip flourish on next render
// platform-correct label for the quick-switcher chord (Slack-style ⌘/Ctrl-K)
const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
const SWAP_KEY = IS_MAC ? '⌘K' : 'Ctrl+K';
function renderDock() {
  const tray = $('dockSlots');
  if (!tray) return;
  const runs = store.listRuns();
  tray.innerHTML = '';
  dockRuns = [];
  const seen = new Set();

  // Saturday review.MD: first screen must answer "What can I do now?"
  // Keep the dock minimal by default (core forgeable skills only).
  // Full future catalog is secondary and collapsed.
  const servicesToShow = showFullDock ? ALL_SERVICES : SERVICES;

  const addTile = (svc, opts) => {
    const el = document.createElement('div');
    el.className = 'dock__tile';
    el.tabIndex = 0;
    el.setAttribute('role', 'button');
    el.dataset.state = opts.state;
    el.dataset.key = svc.key;
    if (opts.runid) el.dataset.runid = opts.runid;
    if (opts.pop) el.classList.add('dock__tile--pop'); // one-shot equip flourish
    const g = document.createElement('span');
    g.className = 'dock__glyph';
    paintIcon(g, svc, svc.glyph, 1, { state: opts.state });
    el.appendChild(g);
    if (opts.lv != null) { const b = document.createElement('span'); b.className = 'dock__lv'; b.textContent = 'L' + opts.lv; el.appendChild(b); }
    if (opts.keyN != null) { const k = document.createElement('span'); k.className = 'dock__key'; k.textContent = opts.keyN; el.appendChild(k); }
    if (opts.forge) { const f = document.createElement('span'); f.className = 'dock__forge'; f.textContent = '+'; el.appendChild(f); }
    if (opts.lock) { const l = document.createElement('span'); l.className = 'dock__lock'; l.textContent = '🔒'; el.appendChild(l); }
    const t = document.createElement('span'); t.className = 'dock__tip';
    if (opts.tipHtml) { t.classList.add('dock__tip--rich'); t.innerHTML = opts.tipHtml; }
    else t.textContent = opts.tip;
    el.appendChild(t);
    el.setAttribute('aria-label', opts.tip);
    el.onclick = opts.onClick;
    el.onmouseenter = () => sfx.hover();
    el.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); opts.onClick(); } };
    tray.appendChild(el);
  };

  for (const svc of servicesToShow) {
    if (svc.skill) {
      const run = runs.find((r) => skillByKey(r.base)?.key === svc.skill);
      if (run) {
        seen.add(run.id);
        const { lv } = skillLevel(run);
        const equipped = run.id === state.activeRunId;
        dockRuns.push(run.id);
        const keyN = dockRuns.length <= 9 ? dockRuns.length : null;
        const uses = usesByRun.get(run.id) || 0;
        const sk = skillByKey(svc.skill);
        addTile(svc, {
          state: equipped ? 'equipped' : 'owned', runid: run.id, lv, keyN, pop: equipped && justEquippedId === run.id,
          tip: `${svc.name} · Lv ${lv}${equipped ? ' · equipped' : ''}${uses ? ' · ' + uses + '×' : ''}${keyN ? ' · [' + keyN + ']' : ''}`,
          tipHtml: dockTip(svc.name, { lv, rarity: rarityOf(lv), scope: sk?.spec?.scope, opsN: sk?.spec?.ops?.length, uses, keyN, equipped }),
          // the equipped "lead" slot opens the radial quick-swap wheel (BotW-style)
          onClick: () => (equipped ? openWheel(false) : applyRun(run.id)),
        });
      } else {
        addTile(svc, {
          state: 'forge', forge: true, tip: `${svc.name} — train this surface`,
          onClick: () => { selectSkill(svc.skill); openTrainer(); },
        });
      }
    } else {
      addTile(svc, {
        state: 'locked', lock: true, tip: `${svc.name} — locked account root`,
        onClick: () => stageMsg(`“${svc.name}” is locked in this build. Train one of the unlocked account roots first.`),
      });
    }
  }

  // forged BYOD adapters (custom notes) map to no service → append as tiles
  const extra = runs.filter((r) => !seen.has(r.id));
  if (extra.length) { const sep = document.createElement('div'); sep.className = 'dock__sep'; tray.appendChild(sep); }
  for (const r of extra) {
    const { lv } = skillLevel(r);
    const equipped = r.id === state.activeRunId;
    dockRuns.push(r.id);
    const keyN = dockRuns.length <= 9 ? dockRuns.length : null;
    addTile({ key: 'byod-' + r.id, name: r.name, ...BYOD_TILE }, {
      state: equipped ? 'equipped' : 'owned', runid: r.id, lv, keyN, pop: equipped && justEquippedId === r.id,
      tip: `${r.name} · Lv ${lv}${equipped ? ' · equipped' : ''}${keyN ? ' · [' + keyN + ']' : ''}`,
      tipHtml: dockTip(r.name, { lv, rarity: rarityOf(lv), scope: 'your private notes', uses: usesByRun.get(r.id) || 0, keyN, equipped }),
      onClick: () => (equipped ? openWheel(false) : applyRun(r.id)),
    });
  }
  justEquippedId = null; // consumed

  // Saturday review.MD: keep first screen focused.
  // Only show the full future armory if user explicitly asks.
  if (!showFullDock) {
    const more = document.createElement('div');
    more.className = 'dock__tile dock__more';
    more.textContent = '…';
    more.title = 'Show additional planned surfaces (not yet trainable)';
    more.onclick = () => { showFullDock = true; renderDock(); };
    tray.appendChild(more);
  }
}
// GW2-style rich tooltip card for a dock tile.
function dockTip(name, { lv, rarity, scope, opsN, uses, keyN, equipped } = {}) {
  const rows = [`<b class="dock__tipname">${esc(name)}</b>`];
  if (lv != null) rows.push(`<span class="dock__tiprar" data-rar="${(rarity && rarity.key) || 'common'}">Lv ${lv} · ${esc((rarity && rarity.label) || '')}</span>`);
  if (scope) rows.push(`<span class="dock__tipline">⚔ ${esc(scope)}</span>`);
  const bits = [];
  if (opsN != null) bits.push(`${opsN} action${opsN === 1 ? '' : 's'}`);
  if (uses) bits.push(`used ${uses}×`);
  if (bits.length) rows.push(`<span class="dock__tipline dim">${bits.join(' · ')}</span>`);
  rows.push(`<span class="dock__tipkey">${equipped ? `◆ equipped — ${SWAP_KEY} or click to switch` : (keyN ? `press [${keyN}] · ${SWAP_KEY} to switch` : 'tap to equip')}</span>`);
  return rows.join('');
}
// number-key quick-equip (FPS weapon slots): 1..9 → the Nth owned dock tile
let lastEquipIntent = null; // last run id a quick-equip resolved to (test/devtools surface)
function equipByIndex(i) {
  if (i < 0 || i >= dockRuns.length) return;
  lastEquipIntent = dockRuns[i];
  applyRun(dockRuns[i]);
}

// ── sound: tiny synthesized SFX, no asset files (Web Audio) ──────────────────
// Lazily created on first user gesture (browsers suspend audio until then) and
// fully muteable. Tasteful, short, low-gain — game feedback, not a soundtrack.
const sfx = (() => {
  let ctx = null, muted = false;
  try { muted = localStorage.getItem('eg_mute') === '1'; } catch {}
  const ac = () => {
    if (!ctx) { try { ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch {} }
    if (ctx && ctx.state === 'suspended') ctx.resume();
    return ctx;
  };
  const tone = (freq, at, dur, type = 'sine', gain = 0.05, slideTo = null) => {
    const c = ac(); if (!c || muted) return;
    const t = c.currentTime + at, o = c.createOscillator(), g = c.createGain();
    o.type = type; o.frequency.setValueAtTime(freq, t);
    if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, t + dur);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(c.destination); o.start(t); o.stop(t + dur + 0.03);
  };
  return {
    get muted() { return muted; },
    toggle() { muted = !muted; try { localStorage.setItem('eg_mute', muted ? '1' : '0'); } catch {} if (!muted) this.equip(); return muted; },
    hover() { tone(1100, 0, 0.035, 'triangle', 0.018); },
    open() { tone(360, 0, 0.14, 'sawtooth', 0.035, 760); },
    move() { tone(720, 0, 0.03, 'square', 0.02); },
    equip() { tone(523.25, 0, 0.08, 'triangle', 0.05); tone(783.99, 0.06, 0.1, 'triangle', 0.05); tone(1046.5, 0.13, 0.16, 'sine', 0.045); },
    cancel() { tone(380, 0, 0.12, 'sine', 0.035, 240); },
    error() { tone(170, 0, 0.18, 'square', 0.045); },
  };
})();

// ── BotW-style radial quick-swap wheel ───────────────────────────────────────
// Click the equipped lead slot to fan your owned surfaces into a ring;
// flick the pointer / arrow-keys to highlight, release or Enter to equip.
let wheelOn = false, wheelHeld = false, wheelSel = 0, wheelNodes = [];
function ownedRunsInDockOrder() { return dockRuns.map((id) => store.getRun(id)).filter(Boolean); }
function openWheel(held) {
  if (wheelOn) { if (!held) closeWheel(true); return; }
  const el = $('wheel'); if (!el) return;
  const runs = ownedRunsInDockOrder();
  if (!runs.length) { sfx.error(); stageMsg('No surfaces to swap yet — train one first.'); return; }
  wheelOn = true; wheelHeld = !!held; wheelNodes = [];
  const ring = $('wheelRing'); ring.innerHTML = '';
  const N = runs.length, R = Math.min(168, 96 + N * 9);
  runs.forEach((r, i) => {
    const ang = -Math.PI / 2 + i * (2 * Math.PI / N);
    const x = Math.cos(ang) * R, y = Math.sin(ang) * R;
    const sk = skillByKey(r.base), d = dockOf(r.base) || { ...BYOD_TILE, name: r.name };
    const node = document.createElement('button');
    node.type = 'button'; node.className = 'wheel__node';
    node.style.transform = `translate(-50%,-50%) translate(${x}px,${y}px)`;
    const ic = document.createElement('span'); ic.className = 'wheel__nicon'; paintIcon(ic, d, sk?.icon, 1);
    const nm = document.createElement('span'); nm.className = 'wheel__nname'; nm.textContent = r.name;
    const kb = document.createElement('span'); kb.className = 'wheel__nkey'; kb.textContent = i < 9 ? i + 1 : '';
    node.append(ic, nm, kb);
    node.onmouseenter = () => setWheelSel(i, true);
    node.onclick = () => { setWheelSel(i); commitWheel(); };
    ring.appendChild(node); wheelNodes.push({ el: node, run: r });
  });
  const cur = dockRuns.indexOf(state.activeRunId);
  setWheelSel(cur >= 0 ? cur : 0);
  el.hidden = false; el.setAttribute('aria-hidden', 'false');
  document.body.classList.add('wheel-open');
  sfx.open();
}
function setWheelSel(i, quiet) {
  if (!wheelNodes.length) return;
  wheelSel = (i + wheelNodes.length) % wheelNodes.length;
  wheelNodes.forEach((n, j) => n.el.classList.toggle('on', j === wheelSel));
  const hub = $('wheelHub'); if (hub) hub.textContent = wheelNodes[wheelSel].run.name;
  if (!quiet) sfx.move(); else sfx.hover();
}
function commitWheel() {
  const sel = wheelNodes[wheelSel]; const id = sel && sel.run.id;
  closeWheel(false);
  if (id && id !== state.activeRunId) applyRun(id);
}
function closeWheel(silent) {
  if (!wheelOn) return;
  wheelOn = false; wheelHeld = false;
  const el = $('wheel'); if (el) { el.hidden = true; el.setAttribute('aria-hidden', 'true'); }
  document.body.classList.remove('wheel-open');
  if (!silent) sfx.cancel();
}
// flick selection: pointer angle from wheel center → nearest node (outside a deadzone)
function wheelPointerMove(e) {
  if (!wheelOn || wheelNodes.length < 2) return;
  const el = $('wheel'); const r = el.getBoundingClientRect();
  const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  const dx = e.clientX - cx, dy = e.clientY - cy;
  if (Math.hypot(dx, dy) < 34) return; // center deadzone
  const ang = Math.atan2(dy, dx), N = wheelNodes.length, step = 2 * Math.PI / N;
  let best = 0, bd = Infinity;
  for (let i = 0; i < N; i++) {
    const a = -Math.PI / 2 + i * step;
    let diff = Math.abs(((ang - a + Math.PI * 3) % (2 * Math.PI)) - Math.PI);
    if (diff < bd) { bd = diff; best = i; }
  }
  if (best !== wheelSel) setWheelSel(best, true);
}
// risk → accent hue for the dry-run plan readout (matches the gradient-orange theme)
const RISK_HUE = { 'read': '#6b7280', 'read-only': '#6b7280', 'reversible-write': '#c2772a', 'sensitive-write': '#d9480f' };
// ── HUD: the macro verifier readout in the inference pane ─────────────────────
function setMacroCheck(res, skill, text) {
  const el = $('macroCheck');
  if (!el) return;
  if (!res || res.status === 'empty') { el.hidden = true; el.textContent = ''; el.removeAttribute('data-state'); return; }
  el.hidden = false;
  if (res.status === 'ok') {
    el.dataset.state = 'ok';
    const ops = res.calls.map((c) => c.op).join(', ');
    // compile the verified macro into a typed, provider-resolved dry-run plan
    const plan = text ? planFor(skill.key, text) : null;
    const resolved = plan && plan.provider !== 'unknown';
    let planHtml = '';
    if (plan && plan.steps.length) {
      const items = plan.steps.map((s) => {
        const vals = s.args.filter((a) => a.kind === 'string').slice(0, 2).map((a) => esc(a.value)).join(' · ');
        const via = resolved ? ` <span style="opacity:.6">→ ${esc(s.providerMethod)}</span>` : '';
        const tag = s.effect === 'read' ? 'read' : s.risk;
        const risk = ` <span style="font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:${RISK_HUE[tag] || '#6b7280'}">${tag}</span>`;
        return `<li>${esc(s.op.replace(/_/g, ' '))}${vals ? ` — ${vals}` : ''}${via}${risk}</li>`;
      }).join('');
      const caps = resolved && plan.requiredCapabilities.length
        ? `<div style="margin-top:6px;opacity:.7;font-size:11px">would need ${plan.requiredCapabilities.map((c) => `<code>${esc(c)}</code>`).join(' ')} · <b>dry-run</b> — simulated, nothing sent</div>`
        : '';
      planHtml = `<ol class="macrochk__plan">${items}</ol>${caps}`;
    }
    const where = resolved ? ` <span style="opacity:.6">(${esc(plan.provider)})</span>` : '';
    el.innerHTML = `<b>✓ valid write plan</b> · ${res.n} call${res.n === 1 ? '' : 's'} on the ${esc(skill.label)} surface${where} · <code>${esc(ops)}</code>${planHtml}`;
  } else if (res.status === 'oos') {
    el.dataset.state = 'oos';
    el.innerHTML = `<b>⛔ OUT_OF_SCOPE</b> · the ${esc(skill.label)} surface correctly refused — that request is outside its writes`;
  } else {
    el.dataset.state = 'bad';
    el.innerHTML = `<b>✗ invalid macro</b> · ${esc(res.issues.slice(0, 2).join('; '))}`;
  }
}
// ── RPG atlas HUD: turns trained surfaces into a player stat strip ────────────
const RANKS = [[12, 'Grandmaster'], [9, 'Master'], [6, 'Artisan'], [4, 'Adept'], [2, 'Journeyman'], [1, 'Apprentice'], [0, 'Initiate']];
// Paint a service icon into any element. The icon pipeline renders the fallback
// glyph immediately, then upgrades to a vendored SVG and the selected theme.
function paintIcon(el, d, fallbackGlyph, fsScale = 1, opts = {}) {
  if (!el) return;
  paintSkillIcon(el, d || {}, { fallbackGlyph, fsScale, state: opts.state });
}
// the narrator line at the bottom of the stage (classic adventure-game message bar)
function stageMsg(text) { const e = $('stageMsg'); if (e) e.textContent = '» ' + text; }

// ── the stage: a Sierra-style atlas scene — train account surfaces, then equip
//    one and act. Score box + narrator bar.
function renderStage() {
  const stage = $('stage');
  if (!stage) return;
  const runs = store.listRuns();
  const acquired = new Set(runs.map((r) => skillByKey(r.base)?.key).filter(Boolean));
  let maxLv = 0, steps = 0;
  for (const r of runs) { maxLv = Math.max(maxLv, skillLevel(r).lv); steps += (r.steps || 0); }
  const lvl = 1 + Math.floor(steps / 120);
  const xpPct = Math.round(((steps % 120) / 120) * 100);
  const rank = (RANKS.find(([t]) => runs.length >= t) || [0, 'Initiate'])[1];
  const active = runs.find((r) => r.id === state.activeRunId);
  const skill = active ? skillByKey(active.base) : null;
  const d = skill ? dockOf(skill.key) : null;
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  set('stageScore', `${acquired.size} / ${SKILLS.length}`);
  set('stageLv', String(lvl));
  set('stageRank', rank);
  const xp = $('stageXp'); if (xp) xp.style.width = xpPct + '%';
  const scene = $('stageScene');
  const icon = $('stageSignIcon');
  if (active) {
    set('stageSignName', active.name);
    paintIcon(icon, d, skill?.icon, 0.8);
    if (scene) scene.style.setProperty('--scene', themedTileColor(d, iconTheme()));
    stage.dataset.where = 'in';
  } else {
    set('stageSignName', 'Account Atlas');
    if (icon) { icon.classList.remove('hasvg'); icon.textContent = '🌐'; icon.style.background = '#13393f'; icon.style.color = '#cdeeea'; icon.style.fontSize = '17px'; }
    if (scene) scene.style.setProperty('--scene', '#1d6f6a');
    stage.dataset.where = 'out';
  }
}

// ── Train pane: choose which account/app surface to train ────────────────────
const dockOf = (key) => POPULAR_2026.find((s) => s.key === key) || {};
function renderSkillPicker() {
  const host = $('skillPicker');
  if (!host) return;
  const runs = store.listRuns();
  host.innerHTML = '';
  for (const sk of SKILLS) {
    const d = dockOf(sk.key);
    const run = runs.find((r) => skillByKey(r.base)?.key === sk.key);
    const lv = run ? skillLevel(run).lv : 0;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'skillpick__btn' + (sk.key === selectedSkillKey ? ' on' : '') + (lv ? ' forged' : '');
    b.dataset.key = sk.key;
    const icon = document.createElement('span');
    icon.className = 'skillpick__icon';
    paintIcon(icon, d, sk.icon, 0.78);
    const txt = document.createElement('span');
    txt.className = 'skillpick__txt';
    txt.innerHTML = `<b>${esc(sk.label)}</b><i>${sk.spec.ops.length} writes · ${sk.examples.length} drills</i>`;
    b.append(icon, txt);
    if (lv) {
      const badge = document.createElement('span');
      badge.className = 'skillpick__lv';
      badge.textContent = 'L' + lv;
      b.appendChild(badge);
    }
    b.onclick = () => selectSkill(sk.key);
    host.appendChild(b);
  }
}
function renderPairList(host, pairs, { limit = 4, compact = false } = {}) {
  if (!host) return;
  const shown = pairs.slice(0, limit);
  const more = Math.max(0, pairs.length - shown.length);
  host.innerHTML = shown.map(([q, a]) => {
    const macro = compact && a !== 'OUT_OF_SCOPE' ? clip(a, 120) : a;
    return `<li><span class="skill-req">${esc(q)}</span><pre class="skill-macro">${esc(macro)}</pre></li>`;
  }).join('') +
    (more > 0 ? `<li class="skill-more">+ ${more} more ${compact ? 'hidden' : 'spec-valid'} drill${more === 1 ? '' : 's'}</li>` : '');
}
function renderSurfacePlan(sk) {
  const d = dockOf(sk.key);
  paintIcon($('surfacePlanIcon'), d, sk.icon, 0.86);
  const guards = [...(sk.examples || []), ...(sk.eval || [])].filter(([, a]) => a === 'OUT_OF_SCOPE');
  const chips = [
    `${sk.spec.ops.length} writes`,
    `${sk.examples.length} train drills`,
    `${(sk.eval || []).length} held-out evals`,
    `${guards.length} refusal guards`,
    'rank 16 LoRA',
  ];
  const chipHost = $('surfacePlanChips');
  if (chipHost) chipHost.innerHTML = chips.map((c) => `<span class="surfacechip">${esc(c)}</span>`).join('');
  const contract = $('writeContract');
  if (contract) {
    contract.innerHTML = sk.spec.ops.map((op) => {
      const sig = `${op.name}(${(op.params || []).join(', ')})${op.ret ? ' -> ' + op.ret : ''}`;
      const params = (op.params || []).length ? (op.params || []).join(', ') : 'no args';
      return `<div class="contractop"><code>${esc(sig)}</code><span>${esc(params)}</span></div>`;
    }).join('');
  }
  const rules = [];
  if (sk.context) rules.push(['Date anchor', sk.context]);
  rules.push(['Scope', `Only ${sk.spec.scope}; anything else must emit exactly OUT_OF_SCOPE.`]);
  for (const a of sk.contract?.assertions || []) rules.push([a.id, a.describe]);
  for (const f of sk.contract?.forbidden || []) rules.push([f.id, f.describe]);
  const ruleHost = $('surfaceRules');
  if (ruleHost) ruleHost.innerHTML = rules.map(([k, v]) => `<div class="ruleitem"><b>${esc(k)}</b>${esc(v)}</div>`).join('');
  const inscope = (sk.examples || []).filter(([, a]) => a !== 'OUT_OF_SCOPE');
  renderPairList($('guidedList'), inscope, { limit: 5 });
  renderPairList($('evalList'), sk.eval || [], { limit: 4, compact: true });
  renderPairList($('guardList'), guards, { limit: 4, compact: true });
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  set('guidedSummary', `${inscope.length} train`);
  set('evalSummary', `${(sk.eval || []).length} held out`);
  set('guardSummary', `${guards.length} OOS`);
}
function selectSkill(key) {
  const sk = skillByKey(key) || SKILLS[0];
  selectedSkillKey = sk.key;
  document.querySelectorAll('#skillPicker .skillpick__btn').forEach((b) =>
    b.classList.toggle('on', b.dataset.key === sk.key));
  const title = $('skillTitle'); if (title) title.innerHTML = `${sk.icon} ${esc(sk.label)} surface`;
  const desc = $('skillDesc'); if (desc) desc.textContent = sk.desc;
  renderSurfacePlan(sk);
}
async function applyRun(id) {
  const meta = store.getRun(id);
  if (!meta) return;
  if (!state.loaded) { log('Boot the engine first, then equip a surface.'); closeTrainer(); return; }
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
    state.tuned = { name: meta.name, kind: meta.kind, base: meta.base, build: buildFromMeta(meta), suggest: meta.suggest };
    state.activeRunId = id;
    justEquippedId = id; // trigger the one-shot equip flourish on the lead slot
    $('adapterSel').value = meta.name;
    setMacroCheck(null);
    sfx.equip();
    setBadge(); renderHistory(); renderEquipPanel();
    switchTab('infer');
    if (meta.suggest) $('prompt').value = meta.suggest;
    stageMsg(`Equipped “${meta.name}”. Pick a drill or write request.`);
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

function repaintIconSurfaces() {
  renderHistory();
  renderSkillPicker();
  renderEquipPanel();
  renderStage();
}

function initIconTheme() {
  const sel = $('iconTheme');
  if (!sel) return;
  sel.innerHTML = Object.entries(ICON_THEME_PRESETS)
    .filter(([k]) => k !== 'locked')
    .map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`)
    .join('');
  sel.value = iconTheme();
  document.documentElement.dataset.iconTheme = iconTheme();
  sel.onchange = () => {
    setIconTheme(sel.value);
    repaintIconSurfaces();
  };
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
  // render the surface picker + the selected surface's example macros
  renderSkillPicker();
  selectSkill(selectedSkillKey);

  // open the train surface menu from the Atlas header or empty-state CTA
  $('learnBtn')?.addEventListener('click', () => openTrainer());
  $('learnCta')?.addEventListener('click', () => openTrainer());
  $('jobBoardBtn')?.addEventListener('click', () => {
    window.location.href = new URL('wireframes/job-board.html', window.location.href).href;
  });
  $('trainerClose')?.addEventListener('click', () => closeTrainer());
  // click the dimmed backdrop (outside the menu window) to dismiss
  $('trainer')?.addEventListener('click', (e) => { if (e.target.id === 'trainer') closeTrainer(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeTrainer(); });
  // Saturday review.MD: hide secondary controls by default to reduce first-screen noise.
  $('settings').hidden = true;
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

  // Quick-switch surfaces the way workflow software does it: ⌘/Ctrl-K opens the
  // switcher (Slack's quick-switcher convention; also VS Code / Linear / Notion),
  // and 1–9 are direct hotkeys. No game-style hold-to-open.
  document.addEventListener('keydown', (e) => {
    // ⌘K / Ctrl-K — works anywhere, even from a text field (like Slack)
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      wheelOn ? closeWheel(true) : openWheel(false);
      return;
    }
    // switcher navigation takes priority while it's open
    if (wheelOn) {
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); setWheelSel(wheelSel + 1); }
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); setWheelSel(wheelSel - 1); }
      else if (e.key === 'Enter') { e.preventDefault(); commitWheel(); }
      else if (e.key === 'Escape') { e.preventDefault(); closeWheel(false); }
      else if (e.key >= '1' && e.key <= '9') { e.preventDefault(); setWheelSel(+e.key - 1); commitWheel(); }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) return;
    if (e.key >= '1' && e.key <= '9') equipByIndex(+e.key - 1); // direct hotkey
  });
  // flick selection + click-out-to-cancel on the wheel overlay
  const wheelEl = $('wheel');
  if (wheelEl) {
    wheelEl.addEventListener('pointermove', wheelPointerMove);
    wheelEl.addEventListener('pointerdown', (e) => { if (e.target === wheelEl || e.target.id === 'wheelHub') closeWheel(false); });
  }
  // sound mute toggle
  const mute = $('mute');
  if (mute) {
    const paint = () => { mute.textContent = sfx.muted ? '🔇' : '🔊'; mute.classList.toggle('on', !sfx.muted); mute.setAttribute('aria-label', sfx.muted ? 'Unmute sounds' : 'Mute sounds'); };
    paint();
    mute.onclick = () => { sfx.toggle(); paint(); };
  }

  $('trainGuided').onclick = () => {
    const sk = skillByKey(selectedSkillKey) || SKILLS[0];
    // Each surface ships 40-64 spec-valid pairs; sample a balanced subset per run so
    // in-browser training stays responsive, then scale epochs to ~hit a step budget.
    const pool = sampleExamples(sk.examples, 32);
    const ex = pool.map(([q, a]) => ({ messages: [{ role: 'system', content: sk.system }, { role: 'user', content: q }], completion: ' ' + a }));
    const windows = Math.ceil(ex.length / 2);
    runTraining({
      examples: ex,
      lr: 3e-4, epochs: Math.max(6, Math.min(14, Math.round(280 / windows))), accum: 2,
      base: sk.key, kind: 'guided', system: sk.system,
      build: (u) => [{ role: 'system', content: sk.system }, { role: 'user', content: u }],
      suggest: sk.suggest,
    });
  };

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
  window.__eg = { store, renderHistory, renderDock, renderStage, stageMsg, renderEquipPanel, humanizePlan, applyRun, exportRun, delRun, state, // devtools/test surface
    openTrainer, closeTrainer, openWheel, closeWheel, commitWheel, setWheelSel, sfx, SKILLS, POPULAR_2026, selectSkill, renderSkillPicker, verifyMacro, planFor, dryRun, setMacroCheck, equipByIndex, skillByKey, sampleExamples,
    setIconTheme: (theme) => { const t = setIconTheme(theme); const sel = $('iconTheme'); if (sel) sel.value = t; repaintIconSurfaces(); return t; },
    get iconTheme() { return iconTheme(); },
    get selectedSkillKey() { return selectedSkillKey; }, get lastEquipIntent() { return lastEquipIntent; } };

  initFs();
  initIconTheme();
  renderHistory();
  switchTab('infer'); setBadge(); refreshOwn(); gateButtons();
});

function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
