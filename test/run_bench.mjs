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

import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { extname, isAbsolute, relative, resolve } from 'node:path';
import { chromium } from 'playwright';

const root = process.cwd();
const types = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
]);

const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://127.0.0.1').pathname);
    const file = resolve(root, pathname === '/' ? 'index.html' : `.${pathname}`);
    const rel = relative(root, file);
    if (rel.startsWith('..') || isAbsolute(rel)) {
      res.writeHead(403);
      res.end('forbidden');
      return;
    }
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': types.get(extname(file)) ?? 'application/octet-stream' });
    res.end(body);
  } catch (err) {
    res.writeHead(err.code === 'ENOENT' ? 404 : 500);
    res.end(String(err.message ?? err));
  }
});

await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

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

  // Real artifact emission for the Saturday review benchmark card.
  // When this runs against a real /model in a real browser, we write the raw evidence
  // that can be committed. README numbers must come from this artifact.
  if (rows.some(l => l.includes('"type":"done"'))) {
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
        webgpuFlags: ['--enable-unsafe-webgpu', '--enable-features=WebGPU'],
      },
      model: {
        path: '/model (or ?model=... override)',
        note: 'Must be real safetensors from /model — no substitutes allowed for published numbers',
      },
      rows: parsedRows,
      acceptance: 'This artifact is the single source of truth. Copy tok/s, prefill, load times etc. from a committed version of this file only after a clean run on real weights. Device, Chrome version, and model path must be visible here.',
    };

    const { writeFile } = await import('node:fs/promises');
    await writeFile('benchmark-artifact.json', JSON.stringify(artifact, null, 2));
    console.log('Wrote benchmark-artifact.json — this is the raw evidence artifact for the review card.');
  }
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
