/* Build (or --check) CATALOG.json — the discoverable registry of skill packages.
 *   node scripts/build_catalog.mjs           # write CATALOG.json
 *   node scripts/build_catalog.mjs --check   # fail if the checked-in file is stale (CI/drift guard) */
import { readFileSync, writeFileSync } from 'node:fs';
import { catalog } from '../src/skills.js';

const path = new URL('../CATALOG.json', import.meta.url);
const json = JSON.stringify(catalog(), null, 2) + '\n';
const check = process.argv.includes('--check');

if (check) {
  let current = '';
  try { current = readFileSync(path, 'utf8'); } catch { /* missing → drift */ }
  if (current !== json) {
    console.error('CATALOG.json is stale — run `npm run catalog:build`.');
    process.exit(1);
  }
  console.log('CATALOG.json in sync.');
} else {
  writeFileSync(path, json);
  console.log('CATALOG.json written.');
}
