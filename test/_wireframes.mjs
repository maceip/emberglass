/* Static smoke test for the form-factor wireframes (wireframes/*.html).
   Spawns http-server on a local port, loads each page headless, and asserts:
     - zero pageerrors
     - the never-hide surfaces exist: command input, verified-plan card,
       dry-run/action status, an equipped skill.
   No model, no network, no GPU. Run: node test/_wireframes.mjs */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';

const PORT = Number(process.env.WF_PORT || 8019);
const BASE = `http://localhost:${PORT}`;
const DIR = 'docs/wireframes';

function startServer() {
  const cp = spawn('npx', ['http-server', '.', '-p', String(PORT), '-c-1', '--silent'], {
    stdio: 'ignore',
  });
  return cp;
}
async function waitForServer(url, tries = 40) {
  for (let i = 0; i < tries; i++) {
    try {
      const r = await fetch(url);
      if (r.ok) return true;
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`server did not come up at ${url}`);
}

const PAGES = [`${DIR}/index.html`, `${DIR}/desktop.html`, `${DIR}/foldable.html`, `${DIR}/mobile.html`];

const server = startServer();
let failed = false;
try {
  await waitForServer(`${BASE}/${DIR}/index.html`);
  const b = await chromium.launch({ headless: true, args: ['--no-first-run'] });

  for (const path of PAGES) {
    const p = await b.newPage({ viewport: { width: 1180, height: 940 } });
    const errs = [];
    p.on('pageerror', (e) => errs.push(String(e).slice(0, 200)));
    await p.goto(`${BASE}/${path}`, { waitUntil: 'networkidle' });

    const facts = await p.evaluate(() => {
      // the gallery frames each screen in iframes; standalone pages embed the surfaces directly
      const framed = document.querySelectorAll('iframe').length > 0;
      return {
        keep: document.querySelectorAll('.keep').length,
        dryrun: !!document.querySelector('.dryrun') || framed,
        command: !!document.querySelector('.cmd input') || framed,
        plan: !!document.querySelector('.plan') || framed,
        equipped: !!document.querySelector('.chip--on, .tile--equipped') || framed,
      };
    });

    const problems = [];
    if (errs.length) problems.push(`pageerror: ${errs.join(' | ')}`);
    if (!facts.dryrun) problems.push('missing dry-run/action status');
    if (!facts.command) problems.push('missing command input');
    if (!facts.plan) problems.push('missing verified-plan card');
    if (!facts.equipped) problems.push('missing equipped skill');

    if (problems.length) {
      failed = true;
      console.error(`FAIL ${path}\n  - ${problems.join('\n  - ')}`);
    } else {
      console.log(`ok   ${path}  (keep=${facts.keep})`);
    }
    await p.close();
  }

  await b.close();
} catch (e) {
  failed = true;
  console.error('ERROR', e && e.message ? e.message : e);
} finally {
  server.kill('SIGTERM');
}

if (failed) {
  console.error('\nwireframes smoke: FAILED');
  process.exit(1);
}
console.log('\nwireframes smoke: PASS');
