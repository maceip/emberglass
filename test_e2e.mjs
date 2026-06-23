// End-to-end: load the full Qwen2.5-3B in Canary via the harness, generate triage
// output, capture it. Validates the in-browser tf.js WebGPU forward pass works.
import { chromium } from 'playwright';
const CANARY = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const URL = 'http://localhost:8013/';

const browser = await chromium.launch({
  executablePath: CANARY, headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--no-first-run', '--no-default-browser-check'],
});
const page = await browser.newPage();
page.on('console', m => { const t = m.text(); if (/error|fail|exception|ready|webgpu|weights|nan|undefined/i.test(t)) console.log('[c]', m.type(), t.slice(0, 200)); });
page.on('pageerror', e => console.log('[pageerror]', String(e).slice(0, 300)));

await page.goto(URL, { waitUntil: 'domcontentloaded' });
console.log('page loaded; clicking Load model…');
await page.locator('#load').click();

// wait for READY (6GB fetch + bf16 decode + GPU upload) — up to 8 min
const t0 = Date.now();
let status = '', ready = false, failed = false;
while (Date.now() - t0 < 480000) {
  status = ((await page.locator('#status').textContent().catch(() => '')) || '').trim();
  if (/^READY/.test(status)) { ready = true; break; }
  if (/ERROR/i.test(status)) { failed = true; break; }
  await page.waitForTimeout(3000);
}
console.log(`LOAD [${((Date.now() - t0) / 1000).toFixed(0)}s]: ${status}`);
if (!ready) { console.log('RESULT: load did NOT reach READY.'); await browser.close(); process.exit(1); }

// run triage, stream output, cap at ~4 min / enough text to judge coherence
await page.locator('#go').click({ noWaitAfter: true });
console.log('generating…');
const g0 = Date.now(); let out = '', stable = 0;
while (Date.now() - g0 < 240000) {
  const cur = ((await page.locator('#out').textContent().catch(() => '')) || '');
  if (cur === out && cur.length > 0) { if (++stable >= 6) break; } else stable = 0;
  out = cur;
  if (out.length > 1200) break; // enough to judge
  await page.waitForTimeout(2000);
}
console.log(`GEN [${((Date.now() - g0) / 1000).toFixed(0)}s] OUTPUT_LEN=${out.length}`);
console.log('STATUS:', ((await page.locator('#status').textContent().catch(() => '')) || '').trim());
console.log('OUTPUT_START>>>'); console.log(out.slice(0, 1500)); console.log('<<<OUTPUT_END');
const hasJson = /"disposition"/.test(out);
const coherent = out.length > 40 && /[a-zA-Z]{3,}/.test(out);
console.log(`\nRESULT: ${coherent ? 'PASS' : 'FAIL'} — loaded + generated ${out.length} chars in-browser. json_verdict=${hasJson} coherent=${coherent}`);
await browser.close();
process.exit(coherent ? 0 : 1);
