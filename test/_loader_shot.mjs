/* Screenshot the warehouse/assembly loader at several progress states for review. */
import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true });
const p = await b.newPage({ viewport: { width: 760, height: 720 }, deviceScaleFactor: 2 });
await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(300);
async function shot(statusText, file) {
  await p.evaluate((t) => { document.getElementById('status').textContent = t; }, statusText);
  await p.waitForTimeout(450);
  await p.locator('#warehouse').screenshot({ path: file });
  console.log('shot', file);
}
await shot('weights: model.embed_tokens 6%', '/tmp/loader_06.png');
await shot('weights: model.layers.12.self_attn streaming + quantizing 47%', '/tmp/loader_47.png');
await shot('weights: model.layers.27.mlp 88%', '/tmp/loader_88.png');
await shot('READY in 18.2s', '/tmp/loader_done.png');
await b.close();
console.log('LOADER_SHOTS_DONE');
