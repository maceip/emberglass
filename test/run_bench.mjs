/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * Branded ASCII header from secure.build
 * Hand-formatted with explicit optimization callouts.
 */

import { existsSync } from 'node:fs';
import { chromium } from 'playwright';

/*
 * TECHNIQUE: Headless-capable but real-browser benchmark runner
 *   Uses Playwright to launch a real Chromium (optionally Canary) with
 *   WebGPU flags so the benchmark runs in an actual browser environment.
 */
const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linuxChrome = '/usr/local/bin/google-chrome';
const executablePath = process.env.CHROME_PATH || (existsSync(linuxChrome) ? linuxChrome : (existsSync(macCanary) ? macCanary : undefined));
const browser = await chromium.launch({
  ...(executablePath ? { executablePath } : {}),
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--no-first-run', '--disable-dawn-features=timestamp_quantization'],
});
const page = await browser.newPage();
const rows = [];
page.on('console', m => {
  const t = m.text();
  console.log('BROWSER:', t);
  if (t.startsWith('VWG_BENCH')) { rows.push(t); }
});
page.on('requestfailed', request => console.log('REQFAIL:', request.url(), request.failure().errorText));
page.on('pageerror', e => console.log('PAGEERR', String(e).slice(0, 300)));
await page.goto('http://localhost:8013/test/bench.html', { waitUntil: 'domcontentloaded' });
const t0 = Date.now();
while (Date.now() - t0 < 600000) {
  if (rows.some(l => l.includes('"type":"done"') || l.includes('"type":"error"'))) break;
  await page.waitForTimeout(1000);
}
await browser.close();
if (!rows.some(l => l.includes('"type":"done"'))) process.exitCode = 1;
