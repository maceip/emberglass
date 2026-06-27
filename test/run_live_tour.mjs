/* LIVE, HEADED product tour — opens a real browser window and walks the full
 * Emberglass user workflow on screen, narrating each step, including the
 * login/session precondition (the capability the architecture review missed):
 *
 *   1. Logged-out Google surface -> Emberglass blocks capture/train/cast and
 *      shows a "Sign in to Google" gate (the doc only modeled tokens, not this).
 *   2. Mock Google account-chooser popup (stand-in for chrome.identity SSO).
 *   3. Signed in -> surface reachable -> Cast builds a verified dry-run plan.
 *   4. Inspect a weak skill (Notes), open the Trial Page.
 *   5. Train -> forge fills, the weak trial flips to pass, Reliable -> Mastered,
 *      Equip unlocks.
 *   6. Equip carries across screens; back on Home the equipped skill casts.
 *   7. Quest Board, then resized to the mobile priority queue.
 *
 * All client-only / dry-run: no model, no network, no account or token touched.
 * Run: node test/run_live_tour.mjs   (set HEADLESS=1 for CI; SLOWMO ms optional)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';

const PORT = Number(process.env.TOUR_PORT || 8048);
const BASE = `http://localhost:${PORT}/docs/wireframes`;
const HEADLESS = process.env.HEADLESS === '1';
const SLOWMO = Number(process.env.SLOWMO || 220);
const SHOTS = '/tmp/eg_ui_scratch/tour';
mkdirSync(SHOTS, { recursive: true });

const srv = spawn('npx', ['http-server', '.', '-p', String(PORT), '-c-1', '--silent'], { stdio: 'ignore' });
const waitUp = async (u, n = 40) => { for (let i = 0; i < n; i++) { try { if ((await fetch(u)).ok) return; } catch {} await new Promise(r => setTimeout(r, 250)); } throw new Error('no server'); };

let step = 0;
const beat = (ms = 1500) => new Promise(r => setTimeout(r, ms));

async function caption(p, title, body) {
  step++;
  await p.evaluate(({ title, body, step }) => {
    let el = document.getElementById('__cap');
    if (!el) { el = document.createElement('div'); el.id = '__cap'; document.body.appendChild(el); }
    el.style.cssText = 'position:fixed;left:12px;right:12px;bottom:12px;z-index:9999;pointer-events:none;' +
      'background:#0a1416f2;color:#eafffb;border:1px solid #1d6f6a;border-radius:12px;padding:12px 16px;' +
      'font:14px/1.45 Verdana,system-ui,sans-serif;box-shadow:0 10px 28px #0009;max-width:760px;margin:0 auto;';
    el.innerHTML = `<b style="color:#ffd24a">Step ${step} · ${title}</b><br><span style="color:#cfeee9">${body}</span>`;
  }, { title, body, step });
}
async function shot(p, name) { await p.screenshot({ path: `${SHOTS}/${String(step).padStart(2, '0')}-${name}.png`, fullPage: true }); }

await waitUp(`${BASE}/home.html`);
const b = await chromium.launch({ headless: HEADLESS, slowMo: SLOWMO, args: ['--no-first-run'] });
const p = await b.newPage({ viewport: { width: 1180, height: 900 }, deviceScaleFactor: 2 });
p.on('pageerror', e => console.log('PAGEERR', String(e).slice(0, 200)));
const log = (m) => console.log(`  ${m}`);

try {
  // 1 — logged-out gate
  await p.goto(`${BASE}/home.html?session=out`, { waitUntil: 'networkidle' });
  await caption(p, 'Logged-out surface → sign-in gate',
    'Calendar’s Google tab is signed out. Emberglass refuses to capture/train/cast and shows a gate — the precondition the review only half-covered (it modeled OAuth tokens, not session state).');
  await p.waitForSelector('.gate'); await shot(p, 'gate'); log('gate shown'); await beat(2600);

  // 2 — open the mock account chooser
  await p.click('.gate .btn--hero');
  await caption(p, 'Mock Google account chooser',
    'Clicking “Sign in to Google” opens a stand-in for the chrome.identity / Google SSO popup. No credentials, no token — we only need to know the surface is reachable.');
  await p.waitForSelector('.authoverlay'); await shot(p, 'chooser'); log('chooser open'); await beat(2600);

  // 3 — choose account -> signed in -> cast
  await p.click('.acct'); // first account
  await p.waitForSelector('.cast input');
  await caption(p, 'Signed in → surface reachable → Cast',
    'Session flips to logged-in; the gate is replaced by the Cast box. Casting builds a verified, contract-checked dry-run plan.');
  await shot(p, 'signedin'); await beat(1800);
  await p.click('.cast .btn--hero');                 // Cast
  await p.waitForSelector('.seal .stamp--ok', { timeout: 4000 });
  await caption(p, 'Verified plan seal',
    'create_event(…) · contract passed · “no account changed · simulated”. The plan is ready; nothing was executed.');
  await shot(p, 'plan'); log('plan sealed'); await beat(2600);

  // 4 — inspect a weak skill
  await p.evaluate(() => window.__sel('notes'));
  await caption(p, 'Inspect a weak skill (Notes)',
    'Notes is Learning (68%) with a “title missing” failure. Open its Trial Page to train it.');
  await shot(p, 'inspect'); await beat(2200);

  // 5 — Trial Page: train
  await p.goto(`${BASE}/skill.html`, { waitUntil: 'networkidle' });
  await caption(p, 'Trial Page — before training',
    'Calendar at Reliable (93%), 3/4 trials pass, the Boundary Check is weak, Equip is locked until it passes.');
  await shot(p, 'skill-before'); await beat(2400);
  await p.click('#train');
  await caption(p, 'Forge running…',
    'Train Starter Skill fills the forge meter and re-runs the weak trial.');
  await beat(1300);
  await p.waitForSelector('#equip:not([disabled])', { timeout: 5000 }); // enabled only once Mastered
  await caption(p, 'Mastered → Equip unlocked',
    'The Boundary Check flipped to pass, score climbed to Mastered (97%), before/after updated, and Equip is now enabled.');
  await shot(p, 'skill-after'); log('trained'); await beat(2600);

  // 6 — Equip -> carries back to Home
  await p.click('#equip');
  await p.waitForURL('**/home.html', { timeout: 5000 });
  await p.waitForSelector('.heroslot');
  await caption(p, 'Equipped — carried across screens',
    'Equipping wrote to sessionStorage, so Calendar is the equipped skill back on Home, ready to cast.');
  await shot(p, 'equipped'); await beat(2400);

  // 7 — Quest Board, then mobile queue
  await p.goto(`${BASE}/job-board.html`, { waitUntil: 'networkidle' });
  await caption(p, 'Quest Board (desktop)',
    'Failures become tasks: what’s reliable, what failed, what to train next.');
  await shot(p, 'board-desk'); await beat(2200);
  await p.setViewportSize({ width: 390, height: 880 });
  await p.goto(`${BASE}/job-board.html`, { waitUntil: 'networkidle' });
  await caption(p, 'Quest Board (mobile priority queue)',
    'On mobile the comparison table becomes a prioritized card queue — train-me-next first, locked promises last.');
  await p.waitForSelector('.board-cards .qcard'); await shot(p, 'board-mob'); log('mobile queue'); await beat(2600);

  console.log('\nLIVE_TOUR_DONE — frames in', SHOTS);
} catch (e) {
  console.error('TOUR ERROR', e.message);
  await shot(p, 'error');
} finally {
  await beat(800);
  await b.close();
  srv.kill('SIGTERM');
}
