/* LIVE, HEADED, REAL-GPU workflow — opens Chrome Canary with WebGPU and drives
 * the actual Emberglass app (docs/index.html) through the genuine engine:
 *   1. Stream VibeThinker-3B int4 weights from HuggingFace (WeiboAI/VibeThinker-3B)
 *      and assemble the custom WebGPU runtime on the GPU.
 *   2. BASE answer (adapter = none) to the flagship Inbox & Calendar request.
 *   3. Train a real per-surface LoRA in the browser (full backward + AdamW),
 *      then persist it (localStorage index + IndexedDB safetensors).
 *   4. Equip & act with the just-trained adapter; assert the macro changed.
 * This is the real ML — no mocks. It downloads ~GBs and takes several minutes.
 *
 * Run: node test/run_live_gpu.mjs   (Chrome/Canary with WebGPU required)
 *   CHROME_PATH=…  HEADLESS=1  LOAD_MS=…  TRAIN_MS=…  GEN_MS=…
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { GOLD, verifyMacro } from './_gold_calendar.mjs';

const SHOTS = '/tmp/eg_ui_scratch/gpu';
mkdirSync(SHOTS, { recursive: true });

const PORT = Number(process.env.GPU_PORT || 8016);
const URL = `http://localhost:${PORT}/docs/index.html`;
const LOAD_MS = +(process.env.LOAD_MS || 240000);   // weights stream + GPU assemble (fail-fast diagnostic)
const STALL_MS = +(process.env.STALL_MS || 90000);  // give up if weight % never advances
const TRAIN_MS = +(process.env.TRAIN_MS || 600000);
const GEN_MS = +(process.env.GEN_MS || 240000);
const HEADLESS = process.env.HEADLESS === '1';
const FACTS = [/compose_email|schedule_send/i, /create_event/i];

const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linux = '/usr/local/bin/google-chrome';
const CHROME = process.env.CHROME_PATH || (existsSync(macCanary) ? macCanary : existsSync(linux) ? linux : undefined);

const srv = spawn('npx', ['http-server', '.', '-p', String(PORT), '-c-1', '--silent'], { stdio: 'ignore' });
const waitUp = async (u, n = 40) => { for (let i = 0; i < n; i++) { try { if ((await fetch(u)).ok) return; } catch {} await new Promise(r => setTimeout(r, 250)); } throw new Error('no server'); };

let failures = 0;
const ok = (c, label, extra = '') => { console.log(`  ${c ? 'PASS' : 'FAIL'} · ${label}${extra ? ' — ' + extra : ''}`); if (!c) failures++; return c; };
const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();

await waitUp(URL);
const b = await chromium.launch({
  ...(CHROME ? { executablePath: CHROME } : {}),
  headless: HEADLESS,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--no-first-run'],
});
process.on('exit', (c) => console.log(`\n[process exit ${c}]`));
process.on('uncaughtException', (e) => { console.log('UNCAUGHT', e.message); });
process.on('unhandledRejection', (e) => { console.log('UNHANDLED', String(e).slice(0, 200)); });
const p = await b.newPage();
p.on('crash', () => console.log('  PAGE CRASHED (GPU/renderer)'));
b.on('disconnected', () => console.log('  BROWSER DISCONNECTED'));
p.on('pageerror', (e) => console.log('  PAGEERR', String(e).slice(0, 240)));
p.on('crash', () => console.log('  !! PAGE CRASHED (tab killed — likely GPU device-lost / OOM during quantization)'));
let verbose = true; // print all console during load phase, then quiet down
p.on('console', (m) => { const t = m.text(); if (verbose || /GPUERR|uncaught|unhandled|error|fail|load|model|webgpu|adapter|fetch/i.test(t)) console.log('  CON', t.slice(0, 200)); });
// ── network diagnostics: surface any 4xx/5xx or failed model fetch ──
const netbad = [];
p.on('response', (r) => { const s = r.status(); const u = r.url(); if (s >= 400 && /huggingface|hf\.co|\.safetensors|\.json|\.gguf|resolve|cdn|model/i.test(u)) { const line = `${s} ${u}`; netbad.push(line); console.log('  NET', line); } });
p.on('requestfailed', (r) => { const u = r.url(); if (/huggingface|hf\.co|\.safetensors|\.json|\.gguf|resolve|cdn|model/i.test(u)) { const line = `FAILED ${r.failure()?.errorText || '?'} ${u}`; netbad.push(line); console.log('  NET', line); } });

const enabled = (sel) => p.evaluate((s) => { const e = document.querySelector(s); return !!e && !e.disabled; }, sel);
const txt = (sel) => p.evaluate((s) => document.querySelector(s)?.textContent || '', sel);
const val = (sel) => p.evaluate((s) => document.querySelector(s)?.value || '', sel);
async function waitEnabled(sel, ms, label) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { if (await enabled(sel)) return true; process.stdout.write(`\r  …waiting for ${label || sel} (${((Date.now() - t0) / 1000) | 0}s)   `); await p.waitForTimeout(1000); }
  console.log(`\n  …timed out waiting for ${label || sel}`); return false;
}
async function ask(q) { await p.fill('#prompt', q); await p.click('#run'); await p.waitForTimeout(800); await waitEnabled('#run', GEN_MS, 'generation'); return norm(await txt('#out')); }
const shot = (name) => p.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: true }).catch(() => {});

const Q = "Email the design team this week's notes, then put a 30-minute review on my calendar for Monday morning.";
console.log('LIVE GPU workflow @', URL, '\n  Chrome:', CHROME || '(playwright chromium — may lack WebGPU)');

try {
  await p.goto(URL, { waitUntil: 'domcontentloaded' });
  await p.waitForFunction(() => !!window.__eg, null, { timeout: 10000 }).catch(() => {});

  // ── WebGPU availability probe (the most likely silent failure) ──
  const gpu = await p.evaluate(async () => {
    if (!('gpu' in navigator)) return { supported: false, reason: 'navigator.gpu missing' };
    try { const a = await navigator.gpu.requestAdapter(); return { supported: !!a, reason: a ? 'adapter ok' : 'requestAdapter() returned null' }; }
    catch (e) { return { supported: false, reason: 'requestAdapter threw: ' + e.message }; }
  });
  ok(gpu.supported, 'WebGPU available in this browser', gpu.reason);
  console.log('  #loadHF present:', await p.evaluate(() => !!document.querySelector('#loadHF')),
              '| #run present:', await p.evaluate(() => !!document.querySelector('#run')));

  const MODEL_URL = process.env.MODEL_URL || '/model';
  console.log(`\n[1] loading VibeThinker-3B from ${MODEL_URL} + assembling on GPU …`);
  if (MODEL_URL === 'HF') {
    await p.click('#loadHF');                                  // stream from HuggingFace (can stall on cold cache)
  } else {
    await p.evaluate(() => { const s = document.getElementById('settings'); if (s) s.hidden = false; });
    await p.fill('#modelUrl', MODEL_URL);                      // local same-origin weights (reliable path)
    await p.click('#load');
  }
  // poll with REAL progress: rail message + assemble percent + error banner
  const tL0 = Date.now(); let loaded = false; let lastProg = ''; let lastChange = Date.now(); let stalled = false; let crashed = false;
  while (Date.now() - tL0 < LOAD_MS) {
    let rail = '', pct = '';
    try {
      if (await enabled('#run')) { loaded = true; break; }
      rail = norm(await txt('#railMsg')); pct = norm(await txt('.wh-pct'));
    } catch (e) { crashed = true; console.log('   read failed (page gone):', e.message.slice(0, 80)); break; }
    const prog = `${rail} ${pct}`.trim();
    if (prog !== lastProg) { lastProg = prog; lastChange = Date.now(); }
    console.log('   ', `${((Date.now() - tL0) / 1000) | 0}s · ${prog || '(no rail msg)'}`);
    if (Date.now() - lastChange > STALL_MS) { stalled = true; break; }
    try { await p.waitForTimeout(2500); } catch { crashed = true; break; }
  }
  verbose = false;
  if (!loaded) {
    console.log(`\n  DIAGNOSIS: ${crashed ? 'tab crashed during load' : stalled ? `progress stuck at "${lastProg}" for >${STALL_MS / 1000}s` : 'load did not finish within window'}`);
    console.log('  network problems seen:', netbad.length ? '\n    - ' + [...new Set(netbad)].join('\n    - ') : 'none captured');
  }
  ok(loaded, 'model loaded (WebGPU runtime ready)', loaded ? '' : (crashed ? 'tab crashed' : stalled ? 'weight stream stalled at 0%' : 'timeout'));
  await shot('1-loaded').catch(() => {});
  if (!loaded) { await p.waitForTimeout(1500).catch(() => {}); await b.close(); srv.kill('SIGTERM'); process.exit(1); }

  console.log('[2] BASE answer (adapter = none)');
  const before = await ask(Q);
  console.log('\n    BEFORE >>>', before.slice(0, 200));
  await shot('2-base');

  console.log('\n[3] in-browser LoRA training (real backward + AdamW) …');
  await p.click('#trainGuided');
  const seen = new Set(); const t0 = Date.now(); let trained = false;
  while (Date.now() - t0 < TRAIN_MS) {
    const lbl = norm(await txt('#trainLabel'));
    if (lbl && !seen.has(lbl)) { seen.add(lbl); if (/loss|done|warm|step|epoch/i.test(lbl)) console.log('    ', lbl); }
    if (/done in/i.test(lbl)) { trained = true; break; }
    if (/error/i.test(lbl)) break;
    await p.waitForTimeout(500);
  }
  ok(trained, 'training completed', `${((Date.now() - t0) / 1000).toFixed(1)}s`);
  await shot('3-trained');

  const saved = await p.waitForFunction(() => (window.__eg?.store?.listRuns?.().length ? window.__eg.store.listRuns()[0] : null), null, { timeout: 120000 }).then(h => h.jsonValue()).catch(() => null);
  ok(!!saved, 'adapter persisted (export + save)', saved ? `name="${saved.name}" loss=${saved.finalLoss}` : '');

  console.log('\n[4] TUNED answer (equip & act with the trained adapter)');
  await p.click('#tryItBtn');
  await p.waitForTimeout(800);
  await waitEnabled('#run', GEN_MS, 'tuned generation');
  const after = norm(await txt('#out'));
  console.log('\n    AFTER  >>>', after.slice(0, 200));
  await shot('4-tuned');
  ok(FACTS.some(re => re.test(after)), 'tuned answer emits the calendar/email macro ops');
  ok(after !== before, 'tuned answer differs from base');

  // ── [5] L3 GOLD-TARGET eval: prompt -> action -> verify-against-target ────────
  // (adapter stays equipped after "Equip & act"; amortize one model load over N)
  console.log(`\n[5] gold-target verification (${GOLD.length} cases) — op + contract + target args`);
  const results = [];
  for (const g of GOLD) {
    const macro = await ask(g.prompt);
    const v = verifyMacro(macro, g);
    results.push({ ...v, prompt: g.prompt, macro: macro.slice(0, 400) });
    const checks = Object.entries(v.targetChecks).map(([k, val]) => `${val ? '✓' : '✗'}${k}`).join(' ');
    console.log(`  ${v.pass ? 'PASS' : 'FAIL'} · ${g.id} — op:${v.opPresent ? '✓' : '✗'} contract:${v.contractOk ? '✓' : '✗'} target:${v.targetOk ? '✓' : '✗'}  [${checks}]`);
    console.log(`        got: ${v.got || '(no matching op line)'}`);
  }
  const passed = results.filter(r => r.pass).length;
  const opOnly = results.filter(r => r.opPresent).length;
  const rate = (passed / GOLD.length * 100).toFixed(0);
  writeFileSync('/tmp/eg_ui_scratch/gold_results.json', JSON.stringify({ when: new Date().toISOString(), passed, total: GOLD.length, opOnly, results }, null, 2));
  console.log(`\n  L1 (emits the op):        ${opOnly}/${GOLD.length}`);
  console.log(`  L3 (op+contract+target):  ${passed}/${GOLD.length}  =  ${rate}% success rate`);
  ok(true, `gold eval recorded`, `${passed}/${GOLD.length} (${rate}%) — see /tmp/eg_ui_scratch/gold_results.json`);

  console.log('\n=== LIVE GPU SUMMARY ===');
  console.log('BASE :', before.slice(0, 150));
  console.log('TUNED:', after.slice(0, 150));
  console.log(failures === 0 ? '\nALL PASS · LIVE_GPU_DONE' : `\n${failures} FAIL · LIVE_GPU_FAILED`);
} catch (e) {
  failures++; console.error('\nGPU ERROR', e.message);
  await shot('x-error').catch(() => {});
} finally {
  try { await b.close(); } catch {}
  srv.kill('SIGTERM');
}
process.exit(failures === 0 ? 0 : 1);
