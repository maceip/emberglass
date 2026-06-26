/* Headless sanity check for the Swiss-army-knife skill bar + the new
   Inbox & Calendar skill copy. No model load needed. Output: /tmp/ui_knife_*.png */
import { chromium } from 'playwright';

const b = await chromium.launch({ headless: true, args: ['--no-first-run'] });
const p = await b.newPage({ viewport: { width: 1120, height: 900 } });
const errs = [];
p.on('pageerror', (e) => errs.push(String(e).slice(0, 200)));
await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__eg, null, { timeout: 8000 });

// State 1: no skills yet → the flagship slot should render as a locked "train to forge".
let locked = await p.evaluate(() => {
  const slots = document.querySelectorAll('#knifeSlots .kslot');
  return { count: slots.length, lockedFirst: slots[0]?.classList.contains('kslot--locked') };
});

// Seed skills, incl. a forged inbox-calendar, and equip it.
await p.evaluate(async () => {
  const s = window.__eg.store;
  for (const r of s.listRuns()) await s.deleteRun(r.id);
  const blob = new Blob([new Uint8Array(64)]);
  const mk = (name, kind, base, loss, steps) => ({ id: s.newId(), name, kind, base, createdAt: Date.now(), steps, rank: 16, finalLoss: loss });
  const a = mk('inbox-calendar', 'guided', 'inbox-calendar', 0.08, 84);
  const b2 = mk('my-notes', 'own', 'my-notes', 0.31, 36);
  await s.saveRun(a, { safetensors: blob, configJson: '{}' });
  await s.saveRun(b2, { safetensors: blob, configJson: '{}' });
  window.__eg.state.activeRunId = a.id;     // pretend it's equipped
  window.__eg.renderKnife();
});

const forged = await p.evaluate(() => {
  const slots = [...document.querySelectorAll('#knifeSlots .kslot')];
  return {
    count: slots.length,
    anyLocked: slots.some((s) => s.classList.contains('kslot--locked')),
    equipped: slots.filter((s) => s.classList.contains('equipped')).map((s) => s.querySelector('.kslot__name')?.textContent),
    names: slots.map((s) => s.querySelector('.kslot__name')?.textContent),
  };
});

await p.click('#tabTrain'); await p.waitForTimeout(200);
const macroShown = await p.evaluate(() => {
  const m = document.querySelector('#guidedList .skill-macro');
  return { hasMacro: !!m, sample: m?.textContent?.slice(0, 40) };
});
await p.screenshot({ path: '/tmp/ui_knife_train.png', fullPage: true });
await p.click('#tabInfer'); await p.waitForTimeout(150);
await p.screenshot({ path: '/tmp/ui_knife_infer.png', fullPage: true });

await b.close();
console.log('locked-state  :', JSON.stringify(locked));
console.log('forged-state  :', JSON.stringify(forged));
console.log('macro-render  :', JSON.stringify(macroShown));
console.log('pageerrors    :', errs.length ? errs : 'none');
const ok = locked.lockedFirst && forged.count === 2 && !forged.anyLocked &&
  forged.equipped.includes('inbox-calendar') && macroShown.hasMacro && errs.length === 0;
console.log(ok ? 'KNIFE_CHECK_PASS' : 'KNIFE_CHECK_FAIL');
process.exit(ok ? 0 : 1);
