/* Visual QA for the design-system pass — captures screenshots of every surface
   (no model load needed). Output: /tmp/ui_*.png */
import { chromium } from 'playwright';
import { existsSync } from 'node:fs';
const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linux = '/usr/local/bin/google-chrome';
const CHROME = process.env.CHROME_PATH || (existsSync(linux) ? linux : existsSync(macCanary) ? macCanary : undefined);
const b = await chromium.launch({ ...(CHROME ? { executablePath: CHROME } : {}), headless: true, args: ['--no-first-run'] });
const p = await b.newPage({ viewport: { width: 1120, height: 900 } });
p.on('pageerror', (e) => console.log('PAGEERR', String(e).slice(0, 200)));
await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__eg, null, { timeout: 8000 });

// seed two synthetic runs so the history rail + button variants render
await p.evaluate(async () => {
  const s = window.__eg.store;
  for (const r of s.listRuns()) await s.deleteRun(r.id);
  const mk = (name, kind, loss) => ({ id: s.newId(), name, kind, createdAt: Date.now(), steps: 48, rank: 16, finalLoss: loss });
  const blob = new Blob([new Uint8Array(64)]);
  await s.saveRun(mk('emberglass-os', 'guided', 0.004), { safetensors: blob, configJson: '{}' });
  await s.saveRun(mk('my-notes', 'own', 0.21), { safetensors: blob, configJson: '{}' });
  window.__eg.renderHistory();
});

const shot = async (name) => { await p.waitForTimeout(250); await p.screenshot({ path: `/tmp/ui_${name}.png`, fullPage: true }); console.log('  ✓ /tmp/ui_' + name + '.png'); };

await shot('desktop_infer');
await p.click('#gear'); await shot('settings_open'); await p.click('#gear');
await p.click('#tabTrain'); await shot('desktop_train');
await p.evaluate(() => { document.body.dataset.layout = 'mobile'; }); await shot('mobile_train');
await p.click('#tabInfer'); await shot('mobile_infer');
await p.evaluate(() => { document.body.dataset.layout = 'foldable'; }); await shot('foldable');
await p.evaluate(() => { document.body.dataset.layout = 'desktop'; });
await b.close();
console.log('UI_SHOTS_DONE');
