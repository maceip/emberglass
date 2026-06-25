/* Screenshot the step strips + processing shimmer in active states (orange). */
import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true });
const p = await b.newPage({ viewport: { width: 760, height: 900 }, deviceScaleFactor: 2 });
await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(300);
// inference: mark decode active + proc on
await p.evaluate(() => {
  const s = document.getElementById('inferSteps');
  s.querySelector('[data-s=tok]').className = 'step done';
  s.querySelector('[data-s=prefill]').className = 'step done';
  s.querySelector('[data-s=decode]').className = 'step active';
  document.getElementById('inferProc').classList.add('on');
  document.getElementById('inferCap').textContent = 'Generating the answer one token at a time…';
});
await p.waitForTimeout(300);
await p.locator('#inferSteps').screenshot({ path: '/tmp/steps_infer.png' });
// training: prep done, fwd/bwd/opt looping
await p.evaluate(() => {
  document.getElementById('tabTrain').click();
  const s = document.getElementById('trainSteps');
  s.querySelector('[data-s=prep]').className = 'step done';
  ['fwd', 'bwd', 'opt'].forEach((k) => (s.querySelector('[data-s=' + k + ']').className = 'step loop'));
  document.getElementById('trainCap').textContent = 'Step 22/48 — forward → backward → AdamW · loss 0.649';
  document.getElementById('trainWidget').style.display = '';
  document.getElementById('trainBar').style.width = '46%';
});
await p.waitForTimeout(300);
await p.locator('#paneTrain').screenshot({ path: '/tmp/steps_train.png' });
await b.close();
console.log('STEPS_SHOTS_DONE');
