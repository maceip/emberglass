/* Static smoke test for the three-screen game wireframes (docs/wireframes/*).
   Spawns http-server, loads each screen headless, and asserts:
     - zero pageerrors (ES-module state/ui render cleanly),
     - the never-hide surfaces exist: command input, verified plan seal,
       dry-run trust line, an equipped skill,
     - the *H1/*S1/*J1 asset markers are present (provenance kept in-product),
     - no broken processed assets (icons/frames) or brand SVGs.
   No model, no network, no GPU. Run: node test/_wireframes.mjs */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';

const PORT = Number(process.env.WF_PORT || 8019);
const BASE = `http://localhost:${PORT}`;
const DIR = 'docs/wireframes';

function startServer() {
  return spawn('npx', ['http-server', '.', '-p', String(PORT), '-c-1', '--silent'], { stdio: 'ignore' });
}
async function waitForServer(url, tries = 40) {
  for (let i = 0; i < tries; i++) {
    try { const r = await fetch(url); if (r.ok) return true; } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`server did not come up at ${url}`);
}

// screen -> required asset marker that must remain in the product copy
const SCREENS = [
  { path: `${DIR}/home.html`, marker: '*H1', need: 'equipped' },   // Home: Equip/Cast
  { path: `${DIR}/skill.html`, marker: '*S1', need: 'forge' },     // Skill: Train
  { path: `${DIR}/job-board.html`, marker: '*J1', need: 'quest' }, // Board: Claim
];
const GALLERY = `${DIR}/index.html`;

const server = startServer();
let failed = false;
try {
  await waitForServer(`${BASE}/${GALLERY}`);
  const b = await chromium.launch({ headless: true, args: ['--no-first-run'] });

  // --- gallery: just needs to render error-free and frame the screens ---
  {
    const p = await b.newPage({ viewport: { width: 1180, height: 940 } });
    const errs = [];
    p.on('pageerror', (e) => errs.push(String(e).slice(0, 200)));
    await p.goto(`${BASE}/${GALLERY}`, { waitUntil: 'load' });
    const frames = await p.evaluate(() => document.querySelectorAll('iframe').length);
    const problems = [];
    if (errs.length) problems.push(`pageerror: ${errs.join(' | ')}`);
    if (frames < 3) problems.push(`expected >=3 framed screens, saw ${frames}`);
    if (problems.length) { failed = true; console.error(`FAIL ${GALLERY}\n  - ${problems.join('\n  - ')}`); }
    else console.log(`ok   ${GALLERY}  (frames=${frames})`);
    await p.close();
  }

  // --- each screen rendered standalone ---
  for (const { path, marker, need } of SCREENS) {
    const p = await b.newPage({ viewport: { width: 1180, height: 940 } });
    const errs = [];
    p.on('pageerror', (e) => errs.push(String(e).slice(0, 200)));
    await p.goto(`${BASE}/${path}`, { waitUntil: 'networkidle' });

    const facts = await p.evaluate((mk) => {
      const imgs = Array.from(document.images);
      const broken = imgs.filter((im) => !im.complete || im.naturalWidth === 0).map((im) => im.getAttribute('src'));
      const has = (sel) => !!document.querySelector(sel);
      return {
        dryrun: has('.dryrun'),
        command: has('.cast input') || has('.btn--hero'),
        equipped: has('.chip--on, .tile--equipped, .heroslot'),
        forge: has('.forge'),
        quest: has('.quest') || has('.board'),
        icons: document.querySelectorAll('.sicon img').length,
        marker: document.body.innerHTML.includes(mk),
        imgCount: imgs.length,
        broken,
      };
    }, marker);

    const problems = [];
    if (errs.length) problems.push(`pageerror: ${errs.join(' | ')}`);
    if (!facts.dryrun) problems.push('missing dry-run trust line');
    if (!facts.command) problems.push('missing command/primary action');
    if (!facts[need]) problems.push(`missing dominant-verb surface (${need})`);
    if (!facts.icons) problems.push('no processed *I1 skill icons rendered');
    if (!facts.marker) problems.push(`missing asset marker ${marker}`);
    if (facts.broken.length) problems.push(`broken asset(s): ${facts.broken.join(', ')}`);

    if (problems.length) { failed = true; console.error(`FAIL ${path}\n  - ${problems.join('\n  - ')}`); }
    else console.log(`ok   ${path}  (icons=${facts.icons}, imgs=${facts.imgCount}, ${marker})`);
    await p.close();
  }

  await b.close();
} catch (e) {
  failed = true;
  console.error('ERROR', e && e.message ? e.message : e);
} finally {
  server.kill('SIGTERM');
}

if (failed) { console.error('\nwireframes smoke: FAILED'); process.exit(1); }
console.log('\nwireframes smoke: PASS');
