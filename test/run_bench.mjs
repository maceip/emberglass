/*
 *   ,;
 *  \@@#\:          :/.        .:;;:
 * _@@@@@@#+\|/!;;!-@@@--;    ,@@@@@;
 * .!_*@@@@@@@@@@@@@@@@@@@;   |@@@@@\
 *     .:!|+@@@@@##@@@@@@@#!  -@@@@@#,
 *         .\@@@*;,\@@@@@@@@+,*@@@@@@+.
 *     :*#@@@@@@@@@@@@@@-+@@@@@@@\@@@@-.
 *     .#@@@@@#@@@@#*@@@+ /@@@@@@;\@@@@+.
 *      ;\/:,  -@@@@;|@@@\ ,+@@@@!.+@@@@*:
 *             ,@@@@#*@@@@@#+__!.  ,*@@@@@/
 *              \##+_@@@@@@@@,      ,+@@@_:
 *                   ;;,,..,:         !;.
 */

import { writeFile } from 'node:fs/promises';
import { createRangeServer, listen } from './lib/range_server.mjs';
import { chromeExecutable, launchWebGpuBrowser, webgpuLaunchArgs } from './lib/browser_launch.mjs';

const root = process.cwd();

const server = createRangeServer(root);
await listen(server);

/*
 * TECHNIQUE: Headless-capable but real-browser benchmark runner
 *   Uses Playwright to launch a real Chromium (optionally Canary) with
 *   WebGPU flags so the benchmark runs in an actual browser environment.
 */
const executablePath = chromeExecutable();
const browser = await launchWebGpuBrowser({ headless: false });
try {
  const page = await browser.newPage();
  const rows = [];
  page.on('console', m => {
    const t = m.text();
    console.log('BROWSER:', t);
    if (t.startsWith('VWG_BENCH')) { rows.push(t); }
  });
  page.on('requestfailed', request => console.log('REQFAIL:', request.url(), request.failure().errorText));
  page.on('pageerror', e => console.log('PAGEERR', String(e).slice(0, 300)));

  const { port } = server.address();
  await page.goto(`http://127.0.0.1:${port}/test/bench.html`, { waitUntil: 'domcontentloaded' });
  const t0 = Date.now();
  while (Date.now() - t0 < 600000) {
    if (rows.some(l => l.includes('"type":"done"') || l.includes('"type":"error"'))) break;
    await page.waitForTimeout(1000);
  }
  if (!rows.some(l => l.includes('"type":"done"'))) process.exitCode = 1;

  const hadLoadSuccess = rows.some(l => l.includes('"type":"load"') && !l.includes('error'));
  const hadMeasurements = rows.some(l => l.includes('"type":"greedy-decode"') || l.includes('"type":"prefill"') || l.includes('"type":"train-step"'));
  const hadError = rows.some(l => l.includes('"type":"error"'));

  // Real artifact emission for the Saturday review benchmark card.
  // Only emit if we had a successful real model load + actual measurements, with no load errors.
  // This enforces "real browser run against real model weights only".
  if (rows.some(l => l.includes('"type":"done"')) && hadLoadSuccess && hadMeasurements && !hadError) {
    const parsedRows = [];
    for (const line of rows) {
      const jsonPart = line.replace(/^VWG_BENCH\s*/, '');
      try { parsedRows.push(JSON.parse(jsonPart)); } catch { parsedRows.push({ raw: line }); }
    }

    let userAgent = 'unknown';
    try {
      userAgent = await page.evaluate(() => navigator.userAgent);
    } catch {}

    const artifact = {
      schema: 'emberglass/real-browser-benchmark-artifact/v1',
      generatedBy: 'npm run bench:wgpu (real Chromium + WebGPU subgroups)',
      capturedAt: new Date().toISOString(),
      environment: {
        userAgent,
        executablePath: executablePath || 'system playwright chromium',
        webgpuFlags: webgpuLaunchArgs,
      },
      model: {
        path: '/model (or ?model=... override)',
        note: 'Must be real safetensors from /model — no substitutes allowed for published numbers',
      },
      rows: parsedRows,
      acceptance: 'This artifact is the single source of truth. Copy tok/s, prefill, load times etc. from a committed version of this file only after a clean run on real weights. Device, Chrome version, and model path must be visible here.',
    };

    await writeFile('benchmark-artifact.json', JSON.stringify(artifact, null, 2));
    console.log('Wrote benchmark-artifact.json — this is the raw evidence artifact for the review card.');
  }
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
