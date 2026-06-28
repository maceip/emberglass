import { writeFile } from 'node:fs/promises';
import { createRangeServer, listen } from './lib/range_server.mjs';
import { chromeExecutable, launchWebGpuBrowser, webgpuLaunchArgs } from './lib/browser_launch.mjs';

const root = process.cwd();
const server = createRangeServer(root);
await listen(server);

const executablePath = chromeExecutable();
const browser = await launchWebGpuBrowser({ headless: false });
try {
  const page = await browser.newPage();
  const rows = [];
  page.on('console', (m) => {
    const t = m.text();
    console.log('BROWSER:', t);
    if (t.startsWith('VWG_F16')) rows.push(t);
  });
  page.on('pageerror', (e) => console.log('PAGEERR', String(e).slice(0, 300)));

  const { port } = server.address();
  await page.goto(`http://127.0.0.1:${port}/test/f16_parity.html`, { waitUntil: 'domcontentloaded' });
  const t0 = Date.now();
  while (Date.now() - t0 < 600000) {
    if (rows.some((l) => l.includes('"type":"done"') || l.includes('"type":"error"'))) break;
    await page.waitForTimeout(1000);
  }
  if (!rows.some((l) => l.includes('"type":"done"'))) process.exitCode = 1;

  if (rows.some((l) => l.includes('"type":"done"'))) {
    const parsedRows = [];
    for (const line of rows) {
      const jsonPart = line.replace(/^VWG_F16\s*/, '');
      try { parsedRows.push(JSON.parse(jsonPart)); } catch { parsedRows.push({ raw: line }); }
    }
    let userAgent = 'unknown';
    try { userAgent = await page.evaluate(() => navigator.userAgent); } catch {}

    const artifact = {
      schema: 'emberglass/f16-parity-artifact/v1',
      generatedBy: 'npm run test:f16-parity',
      capturedAt: new Date().toISOString(),
      environment: { userAgent, executablePath: executablePath || 'system playwright chromium', webgpuFlags: webgpuLaunchArgs },
      model: { path: '/model', note: 'Real WeiboAI/VibeThinker-3B weights; greedy decode vs ref.json gen_ids' },
      rows: parsedRows,
    };
    await writeFile('f16-parity-artifact.json', JSON.stringify(artifact, null, 2));
    console.log('Wrote f16-parity-artifact.json');
  }
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
