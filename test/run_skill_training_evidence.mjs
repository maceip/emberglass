/*
 * Real skill training evidence runner (Saturday review — Real Skill Training Card).
 * Drives docs/runtime.html (real WebGPU app) through base → train → tuned GOLD eval.
 * Writes skill-training-artifact.json — no local substitutes.
 */
import { writeFile, mkdirSync } from 'node:fs';
import { createRangeServer, listen } from './lib/range_server.mjs';
import { chromeExecutable, launchWebGpuBrowser } from './lib/browser_launch.mjs';
import { GOLD, verifyMacro } from './_gold_calendar.mjs';

const SHOTS = 'docs/evidence/ui/post-training';
mkdirSync(SHOTS, { recursive: true });

const LOAD_MS = +(process.env.LOAD_MS || 240000);
const TRAIN_MS = +(process.env.TRAIN_MS || 1800000);
const GEN_MS = +(process.env.GEN_MS || 240000);
const Q = GOLD[0].prompt;

const root = process.cwd();
const server = createRangeServer(root);
await listen(server);
const { port } = server.address();
const URL = `http://127.0.0.1:${port}/docs/index.html`;

const executablePath = chromeExecutable();
const browser = await launchWebGpuBrowser({ headless: false });
const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();

let failures = 0;
try {
  const p = await browser.newPage();
  p.on('pageerror', (e) => console.log('PAGEERR', String(e).slice(0, 240)));
  const enabled = (sel) => p.evaluate((s) => { const e = document.querySelector(s); return !!e && !e.disabled; }, sel);
  const txt = (sel) => p.evaluate((s) => document.querySelector(s)?.textContent || '', sel);
  async function waitEnabled(sel, ms, label) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      if (await enabled(sel)) return true;
      await p.waitForTimeout(1000);
    }
    console.log(`timed out: ${label || sel}`);
    return false;
  }
  async function ask(q) {
    await p.fill('#prompt', q);
    await p.click('#run');
    await p.waitForTimeout(800);
    await waitEnabled('#run', GEN_MS, 'generation');
    return norm(await txt('#out'));
  }

  await p.goto(URL, { waitUntil: 'domcontentloaded' });
  await p.evaluate(() => { const s = document.getElementById('settings'); if (s) s.hidden = false; });
  await p.fill('#modelUrl', '/model');
  await p.click('#load');
  if (!await waitEnabled('#run', LOAD_MS, 'model load')) {
    failures++;
    throw new Error('model load failed');
  }

  console.log('[before] GOLD eval (base adapter)');
  const beforeResults = [];
  for (const g of GOLD) {
    const macro = await ask(g.prompt);
    const v = verifyMacro(macro, g);
    beforeResults.push({ phase: 'before', ...v, macro: macro.slice(0, 400) });
    console.log(`  ${v.pass ? 'PASS' : 'FAIL'} · ${g.id}`);
  }
  const beforePass = beforeResults.filter((r) => r.pass).length;

  console.log('[train] guided LoRA …');
  await p.evaluate(() => {
    window.__eg?.selectSkill?.('inbox-calendar');
    window.__eg?.openTrainer?.();
  });
  await p.waitForSelector('#trainer:not([hidden])', { timeout: 10000 });
  await p.waitForFunction(() => !document.getElementById('trainGuided')?.disabled, null, { timeout: 120000 });
  await p.click('#trainGuided');
  const t0 = Date.now();
  let trained = false;
  let lastLbl = '';
  while (Date.now() - t0 < TRAIN_MS) {
    const lbl = norm(await txt('#trainLabel'));
    if (lbl && lbl !== lastLbl) { lastLbl = lbl; console.log('   ', lbl); }
    if (/done in/i.test(lbl)) { trained = true; break; }
    if (/error/i.test(lbl)) break;
    await p.waitForTimeout(500);
  }
  if (!trained) {
    failures++;
    const artifact = {
      schema: 'emberglass/skill-training-artifact/v1',
      status: 'failed',
      capturedAt: new Date().toISOString(),
      before: { goldPass: beforePass, goldTotal: GOLD.length, results: beforeResults },
      error: 'training did not complete within TRAIN_MS',
      lastTrainLabel: lastLbl,
    };
    await writeFile('skill-training-artifact.json', JSON.stringify(artifact, null, 2));
    throw new Error('training did not complete');
  }

  const saved = await p.waitForFunction(
    () => (window.__eg?.store?.listRuns?.().length ? window.__eg.store.listRuns()[0] : null),
    null,
    { timeout: 120000 },
  ).then((h) => h.jsonValue()).catch(() => null);

  await p.click('#tryItBtn');
  await p.waitForTimeout(800);
  await waitEnabled('#run', GEN_MS, 'tuned generation');

  console.log('[after] GOLD eval (trained adapter)');
  const afterResults = [];
  for (const g of GOLD) {
    const macro = await ask(g.prompt);
    const v = verifyMacro(macro, g);
    afterResults.push({ phase: 'after', ...v, macro: macro.slice(0, 400) });
    console.log(`  ${v.pass ? 'PASS' : 'FAIL'} · ${g.id}`);
  }
  const afterPass = afterResults.filter((r) => r.pass).length;

  await p.screenshot({ path: `${SHOTS}/runtime-trained.png`, fullPage: true }).catch(() => {});

  const artifact = {
    schema: 'emberglass/skill-training-artifact/v1',
    generatedBy: 'npm run evidence:skill-training',
    capturedAt: new Date().toISOString(),
    environment: { userAgent: await p.evaluate(() => navigator.userAgent), executablePath: executablePath || 'playwright chromium' },
    trainingSource: {
      skill: 'inbox-calendar',
      provider: 'google',
      corpus: 'src/skills/inbox-calendar/adapters/google.ts (pinned seed via generateCorpus)',
      method: 'guided in-browser LoRA (QwenLoraTrainer, rank 8)',
    },
    evalPromptSet: GOLD.map((g) => ({ id: g.id, prompt: g.prompt })),
    before: { goldPass: beforePass, goldTotal: GOLD.length, results: beforeResults },
    after: { goldPass: afterPass, goldTotal: GOLD.length, results: afterResults },
    adapter: saved ? { name: saved.name, finalLoss: saved.finalLoss, persisted: true } : { persisted: false },
    runtimeNotes: {
      modelPath: '/model',
      flagshipPrompt: Q,
      lift: `${beforePass}/${GOLD.length} → ${afterPass}/${GOLD.length} GOLD L3 pass`,
    },
  };
  await writeFile('skill-training-artifact.json', JSON.stringify(artifact, null, 2));
  console.log('Wrote skill-training-artifact.json');
  console.log(`GOLD L3: before ${beforePass}/${GOLD.length}, after ${afterPass}/${GOLD.length}`);
} catch (e) {
  failures++;
  console.error('SKILL_TRAINING_ERROR', e.message);
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
process.exit(failures ? 1 : 0);
