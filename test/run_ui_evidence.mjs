/*
 * UI evidence capture (Saturday review — UI Reset Card).
 * Screenshots the approved three-screen wireframes at desktop / mobile / fold viewports.
 * First-run = default state. Post-training = skill.html with scores from skill-training-artifact.json when present.
 */
import { readFile, mkdirSync } from 'node:fs';
import { access, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';

const OUT = 'docs/evidence/ui';
mkdirSync(OUT, { recursive: true });

const PORT = Number(process.env.UI_EVIDENCE_PORT || 8020);
const BASE = `http://127.0.0.1:${PORT}`;
const DIR = 'docs/wireframes';

const VIEWPORTS = [
  { name: 'desktop', width: 1280, height: 900 },
  { name: 'mobile', width: 390, height: 844 },
  { name: 'foldable-unfolded', width: 840, height: 900 },
];

const SCREENS = [
  { id: 'home', path: `${DIR}/home.html` },
  { id: 'skill', path: `${DIR}/skill.html` },
  { id: 'job-board', path: `${DIR}/job-board.html` },
];

function startServer() {
  return spawn('npx', ['http-server', '.', '-p', String(PORT), '-c-1', '--silent'], { stdio: 'ignore' });
}

async function waitForServer(url, tries = 40) {
  for (let i = 0; i < tries; i++) {
    try { if ((await fetch(url)).ok) return; } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`server did not start at ${url}`);
}

let postTrainingQuery = '';
try {
  await access('skill-training-artifact.json');
  const art = JSON.parse(await readFile('skill-training-artifact.json', 'utf8'));
  const after = art.after?.goldPass ?? 0;
  const total = art.after?.goldTotal ?? 4;
  const score = Math.round((after / total) * 100);
  postTrainingQuery = `?trained=1&score=${score}&oos=${Math.max(score - 5, 0)}`;
} catch {
  postTrainingQuery = '';
}

const server = startServer();
const manifest = { schema: 'emberglass/ui-evidence-artifact/v1', capturedAt: new Date().toISOString(), shots: [] };

try {
  await waitForServer(`${BASE}/${DIR}/home.html`);
  const browser = await chromium.launch({ headless: true, args: ['--no-first-run'] });

  for (const vp of VIEWPORTS) {
    for (const phase of ['first-run', 'post-training']) {
      for (const screen of SCREENS) {
        const page = await browser.newPage({ viewport: { width: vp.width, height: vp.height } });
        let url = `${BASE}/${screen.path}`;
        if (phase === 'post-training' && screen.id === 'skill' && postTrainingQuery) {
          url += postTrainingQuery;
        }
        await page.goto(url, { waitUntil: 'networkidle' });
        const file = `${OUT}/${phase}-${screen.id}-${vp.name}.png`;
        await page.screenshot({ path: file, fullPage: true });
        manifest.shots.push({ phase, screen: screen.id, viewport: vp.name, path: file });
        await page.close();
        console.log('shot', file);
      }
    }
  }
  await browser.close();
  await writeFile(`${OUT}/manifest.json`, JSON.stringify(manifest, null, 2));
  console.log(`Wrote ${OUT}/manifest.json (${manifest.shots.length} screenshots)`);
} finally {
  server.kill('SIGTERM');
}
