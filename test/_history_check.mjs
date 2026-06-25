/* Verify: full-width tabs, history rail (empty + populated), store round-trip
   (localStorage index + IndexedDB blob), and the 3 layout modes. No model load. */
import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true });
const errs = [];
const shot = async (p, name, sel) => { await (sel ? p.locator(sel) : p).screenshot({ path: `/tmp/${name}.png` }); };

const p = await b.newPage({ viewport: { width: 1040, height: 1180 }, deviceScaleFactor: 2 });
p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
p.on('pageerror', (e) => errs.push('PAGEERROR: ' + e.message));
await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__eg, null, { timeout: 5000 });

// full-width tabs: two tabs should roughly fill the tab strip
const tabFill = await p.evaluate(() => {
  const bar = document.querySelector('.tabbar').getBoundingClientRect().width;
  const t1 = document.getElementById('tabInfer').getBoundingClientRect().width;
  const t2 = document.getElementById('tabTrain').getBoundingClientRect().width;
  return { ratio: +(((t1 + t2) / bar)).toFixed(2) };
});
console.log('tabs fill ratio (want >~0.85):', tabFill.ratio);

console.log('history empty visible:', await p.evaluate(() => getComputedStyle(document.getElementById('historyEmpty')).display !== 'none'));
await shot(p, 'hist_empty', '.win');

// store round-trip: save two runs through the real store module, re-render
const rt = await p.evaluate(async () => {
  const s = window.__eg.store;
  const st = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
  await s.saveRun({ id: 'A', name: 'emberglass-os', kind: 'guided', system: 'sys', suggest: 'who?', createdAt: Date.now() - 9e5, steps: 48, epochs: 12, durationSec: 31.4, finalLoss: 0.132 }, { safetensors: st, configJson: '{"r":16}' });
  await s.saveRun({ id: 'B', name: 'my-notes', kind: 'own', system: null, suggest: 'recall', createdAt: Date.now(), steps: 20, epochs: 5, durationSec: 12.0, finalLoss: 0.401 }, { safetensors: st, configJson: '{"r":16}' });
  window.__eg.renderHistory();
  const files = await s.loadRunFiles('A');
  const buf = new Uint8Array(await files[0].arrayBuffer());
  return { count: s.listRuns().length, lsRaw: localStorage.getItem('emberglass.history.v2')?.length || 0, blobLen: buf.length, fileName: files[0].name, cfg: await files[1].text() };
});
console.log('store round-trip:', JSON.stringify(rt));
console.log('history count badge:', await p.evaluate(() => document.getElementById('historyCount').textContent));
await shot(p, 'hist_full', '.win');

// apply without model should be a graceful no-op (logs to rail)
await p.evaluate(() => window.__eg.applyRun('A'));
await p.waitForTimeout(100);
console.log('apply-without-model rail msg:', await p.evaluate(() => document.getElementById('railMsg').textContent.slice(0, 60)));

// layout modes
await p.evaluate(() => window.__layout('mobile'));
await p.waitForTimeout(120);
await b.newContext; // no-op keepalive
const p2 = await b.newPage({ viewport: { width: 412, height: 915 }, deviceScaleFactor: 2 });
await p2.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p2.waitForFunction(() => !!window.__eg, null, { timeout: 5000 });
await p2.evaluate(async () => { const s = window.__eg.store; await s.saveRun({ id: 'A', name: 'emberglass-os', kind: 'guided', createdAt: Date.now(), steps: 48, durationSec: 31, finalLoss: 0.13 }, { safetensors: new Uint8Array([1, 2]), configJson: '{}' }); window.__eg.renderHistory(); window.__layout('mobile'); });
await p2.waitForTimeout(150);
console.log('mobile layout attr:', await p2.evaluate(() => document.body.dataset.layout));
await shot(p2, 'hist_mobile', '.win');

const p3 = await b.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2 });
await p3.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p3.waitForFunction(() => !!window.__eg, null, { timeout: 5000 });
await p3.evaluate(async () => { const s = window.__eg.store; await s.saveRun({ id: 'A', name: 'emberglass-os', kind: 'guided', createdAt: Date.now(), steps: 48, durationSec: 31, finalLoss: 0.13 }, { safetensors: new Uint8Array([1, 2]), configJson: '{}' }); window.__eg.renderHistory(); window.__layout('foldable'); });
await p3.waitForTimeout(150);
console.log('foldable both panes visible:', await p3.evaluate(() => {
  const a = getComputedStyle(document.getElementById('paneInfer')).display;
  const c = getComputedStyle(document.getElementById('paneTrain')).display;
  return a !== 'none' && c !== 'none';
}));
await shot(p3, 'hist_foldable', '.win');

console.log('CONSOLE ERRORS:', errs.length ? JSON.stringify(errs) : 'none');
await b.close();
console.log('HISTORY_CHECK_DONE');
