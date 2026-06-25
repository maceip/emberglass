/*
 * Emberglass — local persistence for in-browser fine-tunes.
 *
 * Two tiers so nothing a user trains is ever lost on reload:
 *   • localStorage  — a small, synchronous INDEX of every training attempt
 *                     (name, kind, loss, steps, timing, the bits needed to
 *                     rebuild the chat template). Cheap to read at startup so
 *                     the history rail paints instantly, even before WebGPU.
 *   • IndexedDB     — the heavy bytes: each adapter's .safetensors blob +
 *                     adapter_config.json. (localStorage can't hold MBs.)
 *
 * Plus an optional File System Access directory handle (also kept in IDB) so a
 * user can wire up a real folder once and import/export against it thereafter.
 */

const LS_KEY = 'emberglass.history.v2';
const DB_NAME = 'emberglass';
const DB_VERSION = 1;
const BLOB_STORE = 'adapters';
const HANDLE_STORE = 'handles';

// ── IndexedDB (tiny promise wrapper) ──────────────────────────────────────────
let _dbp = null;
function db() {
  if (_dbp) return _dbp;
  _dbp = new Promise((resolve, reject) => {
    const r = indexedDB.open(DB_NAME, DB_VERSION);
    r.onupgradeneeded = () => {
      const d = r.result;
      if (!d.objectStoreNames.contains(BLOB_STORE)) d.createObjectStore(BLOB_STORE);
      if (!d.objectStoreNames.contains(HANDLE_STORE)) d.createObjectStore(HANDLE_STORE);
    };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
  return _dbp;
}
async function idbPut(store, key, val) {
  const d = await db();
  return new Promise((res, rej) => {
    const tx = d.transaction(store, 'readwrite');
    tx.objectStore(store).put(val, key);
    tx.oncomplete = () => res();
    tx.onerror = () => rej(tx.error);
  });
}
async function idbGet(store, key) {
  const d = await db();
  return new Promise((res, rej) => {
    const tx = d.transaction(store, 'readonly');
    const rq = tx.objectStore(store).get(key);
    rq.onsuccess = () => res(rq.result);
    rq.onerror = () => rej(rq.error);
  });
}
async function idbDel(store, key) {
  const d = await db();
  return new Promise((res, rej) => {
    const tx = d.transaction(store, 'readwrite');
    tx.objectStore(store).delete(key);
    tx.oncomplete = () => res();
    tx.onerror = () => rej(tx.error);
  });
}

// ── history index (localStorage) ──────────────────────────────────────────────
export function listRuns() {
  try {
    const a = JSON.parse(localStorage.getItem(LS_KEY) || '[]');
    return Array.isArray(a) ? a : [];
  } catch {
    return [];
  }
}
function writeIndex(arr) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(arr));
  } catch (e) {
    console.warn('[store] localStorage write failed', e);
  }
}
export function getRun(id) {
  return listRuns().find((r) => r.id === id) || null;
}

export function newId() {
  return 'run_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 7);
}

/*
 * Persist a finished training attempt.
 *   meta  — small, JSON-safe descriptor (see main.js for fields).
 *   files — { safetensors: Uint8Array, configJson: string }.
 * The blob lands in IDB keyed by meta.id; the meta is unshifted into the LS
 * index (newest first). Returns the stored meta.
 */
export async function saveRun(meta, files) {
  const stBytes = files.safetensors instanceof Uint8Array ? files.safetensors : new Uint8Array(files.safetensors);
  await idbPut(BLOB_STORE, meta.id, {
    safetensors: new Blob([stBytes], { type: 'application/octet-stream' }),
    configJson: files.configJson || '{}',
  });
  const idx = listRuns().filter((r) => r.id !== meta.id);
  idx.unshift(meta);
  writeIndex(idx);
  return meta;
}

export async function deleteRun(id) {
  writeIndex(listRuns().filter((r) => r.id !== id));
  try {
    await idbDel(BLOB_STORE, id);
  } catch {}
}

// Reconstruct File objects (name/.arrayBuffer()/.text()) so the saved adapter can
// be fed straight back into loadLoraAdapterGPU() — the same path used for uploads.
export async function loadRunFiles(id) {
  const rec = await idbGet(BLOB_STORE, id);
  if (!rec) throw new Error('adapter blob missing for ' + id);
  const meta = getRun(id);
  const stem = (meta?.name || id).replace(/[^\w.-]+/g, '_');
  return [
    new File([rec.safetensors], `${stem}.safetensors`, { type: 'application/octet-stream' }),
    new File([rec.configJson], 'adapter_config.json', { type: 'application/json' }),
  ];
}

export async function getRunBlobs(id) {
  const rec = await idbGet(BLOB_STORE, id);
  if (!rec) throw new Error('adapter blob missing for ' + id);
  return { safetensors: rec.safetensors, configJson: rec.configJson };
}

// ── File System Access directory handle ───────────────────────────────────────
export const fsSupported = typeof window !== 'undefined' && 'showDirectoryPicker' in window;

export async function connectDirectory() {
  if (!fsSupported) throw new Error('File System Access API not available in this browser');
  const handle = await window.showDirectoryPicker({ id: 'emberglass', mode: 'readwrite' });
  await idbPut(HANDLE_STORE, 'dir', handle);
  return handle;
}

export async function savedDirectory() {
  if (!fsSupported) return null;
  try {
    return (await idbGet(HANDLE_STORE, 'dir')) || null;
  } catch {
    return null;
  }
}

export async function forgetDirectory() {
  try {
    await idbDel(HANDLE_STORE, 'dir');
  } catch {}
}

export async function ensurePermission(handle, mode = 'readwrite') {
  if (!handle) return false;
  const opts = { mode };
  if ((await handle.queryPermission(opts)) === 'granted') return true;
  return (await handle.requestPermission(opts)) === 'granted';
}

// Read all text-ish files in a directory handle (depth 1) -> concatenated text.
export async function readDirText(handle, { exts = ['txt', 'md', 'json', 'csv'], maxChars = 200000 } = {}) {
  let out = '';
  const names = [];
  for await (const [name, h] of handle.entries()) {
    if (h.kind !== 'file') continue;
    const ext = name.split('.').pop().toLowerCase();
    if (!exts.includes(ext)) continue;
    try {
      const f = await h.getFile();
      out += `\n\n# ${name}\n` + (await f.text());
      names.push(name);
      if (out.length > maxChars) break;
    } catch {}
  }
  return { text: out.slice(0, maxChars), names };
}

// Write a file into a directory handle.
export async function writeFileToDir(handle, name, data) {
  const fh = await handle.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(data);
  await w.close();
}
