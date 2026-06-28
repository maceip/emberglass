#!/usr/bin/env node
/*
 * Real helper for Saturday review item #1 (Real Browser Benchmark Card).
 *
 * On YOUR LOCAL MACHINE (where your HF cache lives), this script prepares
 * a ./model directory containing the exact sharded safetensors files that
 * the benchmark and runtime expect.
 *
 * Usage (from the emberglass checkout on your machine):
 *   node scripts/prepare-model-from-hf-cache.mjs
 *   # or point at a specific snapshot:
 *   node scripts/prepare-model-from-hf-cache.mjs --snapshot /path/to/snapshots/<hash>
 *
 * After this, you can run:
 *   npm run bench:wgpu
 *
 * The runner will produce benchmark-artifact.json. That artifact (committed)
 * is the only acceptable source for any published numbers per the review.
 *
 * The script uses symlinks by default (fast, no data duplication). Use --copy
 * if you prefer real copies.
 *
 * It looks for the standard HF hub cache layout for WeiboAI/VibeThinker-3B.
 */

import { existsSync, readdirSync, mkdirSync, symlinkSync, copyFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { homedir } from 'node:os';

const REPO = 'WeiboAI/VibeThinker-3B';
const NEEDED = [
  'model.safetensors.index.json',
  'model-00001-of-00002.safetensors',
  'model-00002-of-00002.safetensors',
];

function log(m) { console.log('[prepare-model]', m); }

function findHfHubDir() {
  const env = process.env.HF_HOME || process.env.HUGGINGFACE_HUB_CACHE;
  if (env && existsSync(env)) return env;

  const candidates = [
    join(homedir(), '.cache', 'huggingface', 'hub'),
    join(homedir(), 'Library', 'Caches', 'huggingface', 'hub'), // macOS
    join(homedir(), '.cache', 'huggingface_hub'),               // older layout
  ];
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return null;
}

function findSnapshotDir(hubDir, explicit) {
  if (explicit) {
    if (!existsSync(explicit)) throw new Error(`--snapshot not found: ${explicit}`);
    return resolve(explicit);
  }
  // models--WeiboAI--VibeThinker-3B
  const repoDir = join(hubDir, `models--${REPO.replace('/', '--')}`);
  const snaps = join(repoDir, 'snapshots');
  if (!existsSync(snaps)) throw new Error(`No snapshots dir for ${REPO} in ${hubDir}`);

  const hashes = readdirSync(snaps).filter(h => existsSync(join(snaps, h)));
  if (!hashes.length) throw new Error(`No snapshots found under ${snaps}`);

  // Prefer the most recent by mtime of the dir
  hashes.sort((a, b) => {
    const ma = existsSync(join(snaps, a)) ? (statSync(join(snaps, a)).mtimeMs || 0) : 0;
    const mb = existsSync(join(snaps, b)) ? (statSync(join(snaps, b)).mtimeMs || 0) : 0;
    return mb - ma;
  });

  const chosen = join(snaps, hashes[0]);
  log(`using snapshot ${hashes[0]}`);
  return chosen;
}

function main() {
  const args = process.argv.slice(2);
  const copy = args.includes('--copy');
  const snapIdx = args.indexOf('--snapshot');
  const explicitSnap = snapIdx !== -1 ? args[snapIdx + 1] : null;

  const hubDir = findHfHubDir();
  if (!hubDir) {
    console.error('Could not find Hugging Face hub cache.');
    console.error('Set HF_HOME or place the snapshot files manually into ./model');
    process.exit(1);
  }
  log(`HF hub cache: ${hubDir}`);

  const snapshot = findSnapshotDir(hubDir, explicitSnap);

  // Verify the needed files exist in the snapshot
  const missing = NEEDED.filter(f => !existsSync(join(snapshot, f)));
  if (missing.length) {
    console.error('Missing required files in snapshot:', missing);
    console.error('The WeiboAI/VibeThinker-3B repo uses sharded safetensors.');
    process.exit(1);
  }

  mkdirSync('model', { recursive: true });

  for (const f of NEEDED) {
    const src = join(snapshot, f);
    const dst = join('model', f);
    if (existsSync(dst)) {
      log(`already present: ${dst}`);
      continue;
    }
    if (copy) {
      copyFileSync(src, dst);
      log(`copied ${f}`);
    } else {
      try {
        symlinkSync(src, dst);
        log(`symlinked ${f}`);
      } catch (e) {
        // On Windows without dev mode, symlink may fail; fall back to copy
        copyFileSync(src, dst);
        log(`symlink failed, copied ${f} instead`);
      }
    }
  }

  log('Done. ./model now points at the real weights from your HF cache.');
  log('Run: npm run bench:wgpu');
  log('The resulting benchmark-artifact.json is the raw evidence for the Real Browser Benchmark Card (Saturday review.MD).');
  log('Per the document: numbers may only be published from a committed artifact produced against real weights in a real browser.');
}

main();
