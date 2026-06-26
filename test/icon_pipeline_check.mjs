import { readFileSync } from 'fs';
import {
  ICON_THEME_PRESETS,
  createLogoIndex,
  logoCandidates,
  resolveLogoFromIndex,
} from '../src/icon_pipeline.js';
import { POPULAR_2026 } from '../src/skills.js';

const logos = JSON.parse(readFileSync(new URL('../vendor/logos/logos.json', import.meta.url), 'utf8'));
const index = createLogoIndex(logos, '/vendor/logos');
const failures = [];

const resolve = (key) => resolveLogoFromIndex(index, POPULAR_2026.find((s) => s.key === key));
const expectFile = (key, pattern) => {
  const got = resolve(key);
  if (!got) failures.push(`${key}: no logo resolved`);
  else if (!pattern.test(got.file)) failures.push(`${key}: resolved ${got.file}, expected ${pattern}`);
};

if (logos.length < 1000) failures.push(`catalog too small: ${logos.length}`);
for (const name of ['brand', 'gold', 'cyan', 'pixel', 'pixelGold']) {
  if (!ICON_THEME_PRESETS[name]) failures.push(`missing theme preset: ${name}`);
}

expectFile('inbox-calendar', /google-calendar\.svg$/);
expectFile('music', /spotify.*\.svg$/);
expectFile('github', /github.*\.svg$/);
expectFile('slack', /slack.*\.svg$/);
expectFile('youtube', /youtube.*\.svg$/);
expectFile('maps', /google-maps\.svg$/);
expectFile('chatgpt', /openai.*\.svg$/);

const fallbackOnly = POPULAR_2026.filter((s) => s.logo && !resolveLogoFromIndex(index, s));
if (fallbackOnly.length) failures.push(`logo ids missing from catalog: ${fallbackOnly.map((s) => s.key).join(', ')}`);

const amazon = POPULAR_2026.find((s) => s.key === 'amazon');
if (!logoCandidates(amazon).includes('amazon')) failures.push('candidate list should include tile key fallback');

console.log(`logo catalog: ${logos.length} entries`);
console.log(`themes: ${Object.keys(ICON_THEME_PRESETS).join(', ')}`);
console.log(`mapped service logos: ${POPULAR_2026.filter((s) => s.logo).length}`);
console.log(failures.length ? 'ICON_PIPELINE_FAIL' : 'ICON_PIPELINE_PASS');
if (failures.length) {
  for (const f of failures) console.log(' - ' + f);
  process.exit(1);
}
