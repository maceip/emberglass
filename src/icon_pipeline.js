/*
 * icon_pipeline.js
 *
 * Runtime icon renderer for skill tiles. The app keeps tiny glyph/color metadata
 * in `skills.js` for instant fallback, then upgrades to vendored SVG logos from
 * `vendor/logos` when the catalog is available.
 */

const DEFAULT_BASE_PATHS = ['/vendor/logos', './vendor/logos', '../vendor/logos'];

export const ICON_THEME_PRESETS = {
  brand: { mode: 'brand', label: 'Brand', bg: null, fg: null },
  gold: { mode: 'mono', label: 'Gold monochrome', bg: '#2b220b', fg: '#ffd24a' },
  cyan: { mode: 'mono', label: 'Cyan monochrome', bg: '#082a2e', fg: '#61f2ff' },
  pixel: { mode: 'pixel', label: '8-bit brand', bg: null, fg: null, pixelSize: 18 },
  pixelGold: { mode: 'pixel-mono', label: '8-bit gold', bg: '#201806', fg: '#ffd24a', pixelSize: 18 },
  locked: { mode: 'mono', label: 'Locked', bg: '#d7d2c2', fg: '#7d7768' },
};

export const LOGO_ALIASES = {
  'inbox-calendar': ['google-calendar', 'google-gmail'],
  music: ['spotify'],
  github: ['github'],
  youtube: ['youtube'],
  instagram: ['instagram'],
  x: ['twitter'],
  slack: ['slack'],
  notion: ['notion'],
  maps: ['google-maps'],
  reddit: ['reddit'],
  linkedin: ['linkedin'],
  google: ['google'],
  whatsapp: ['whatsapp'],
  tiktok: ['tiktok'],
  facebook: ['facebook'],
  messenger: ['messenger'],
  discord: ['discord'],
  telegram: ['telegram'],
  netflix: ['netflix'],
  twitch: ['twitch'],
  spotify: ['spotify'],
  pinterest: ['pinterest'],
  threads: ['threads'],
  airbnb: ['airbnb'],
  paypal: ['paypal'],
  chatgpt: ['openai'],
  gemini: ['google-gemini'],
  perplexity: ['perplexity'],
};

let catalogPromise = null;
let activeTheme = safeStorageGet('eg_icon_theme') || 'brand';
const paintVersions = new WeakMap();
const rasterCache = new Map();

function safeStorageGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function safeStorageSet(key, value) {
  try { localStorage.setItem(key, value); } catch {}
}

function slug(s) {
  return String(s || '')
    .toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function trimSlash(s) {
  return String(s || '').replace(/\/+$/, '');
}

function cssUrl(src) {
  return `url("${String(src).replace(/"/g, '\\"')}")`;
}

function svgDataUrl(svg) {
  return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(String(svg));
}

function firstColor(bg) {
  if (!bg) return null;
  const m = String(bg).match(/#[0-9a-f]{3,8}/i);
  return m ? m[0] : (String(bg).startsWith('#') ? bg : null);
}

export function iconTheme() {
  return activeTheme in ICON_THEME_PRESETS ? activeTheme : 'brand';
}

export function iconThemePreset(theme = iconTheme()) {
  return ICON_THEME_PRESETS[theme] || ICON_THEME_PRESETS.brand;
}

export function setIconTheme(theme) {
  activeTheme = theme in ICON_THEME_PRESETS ? theme : 'brand';
  safeStorageSet('eg_icon_theme', activeTheme);
  try { document.documentElement.dataset.iconTheme = activeTheme; } catch {}
  try { window.dispatchEvent(new CustomEvent('eg-icon-theme', { detail: { theme: activeTheme } })); } catch {}
  return activeTheme;
}

export function createLogoIndex(entries, basePath = '/vendor/logos') {
  const byShortname = new Map();
  const byFileStem = new Map();
  const byName = new Map();
  for (const entry of entries || []) {
    const record = { ...entry, basePath: trimSlash(basePath) };
    byShortname.set(slug(entry.shortname), record);
    byName.set(slug(entry.name), record);
    for (const f of entry.files || []) byFileStem.set(slug(f.replace(/\.svg$/i, '')), record);
  }
  return { basePath: trimSlash(basePath), entries: entries || [], byShortname, byFileStem, byName };
}

async function fetchCatalog(basePath) {
  const base = trimSlash(basePath);
  const resp = await fetch(`${base}/logos.json`, { cache: 'force-cache' });
  if (!resp.ok) throw new Error(`logos catalog not found at ${base}`);
  const entries = await resp.json();
  return createLogoIndex(entries, base);
}

export async function loadLogoCatalog(basePaths = DEFAULT_BASE_PATHS) {
  if (!catalogPromise) {
    catalogPromise = (async () => {
      let lastErr = null;
      for (const base of basePaths) {
        try { return await fetchCatalog(base); }
        catch (e) { lastErr = e; }
      }
      throw lastErr || new Error('logo catalog unavailable');
    })();
  }
  return catalogPromise;
}

export function logoCandidates(tile = {}) {
  const keys = [];
  const add = (v) => {
    if (!v) return;
    if (Array.isArray(v)) { for (const x of v) add(x); return; }
    const k = slug(v);
    if (k && !keys.includes(k)) keys.push(k);
  };
  add(tile.logo);
  add(LOGO_ALIASES[slug(tile.key)]);
  add(tile.key);
  add(tile.shortname);
  add(tile.name);
  add(tile.label);
  return keys;
}

function chooseFile(entry, preferred) {
  const files = entry?.files || [];
  if (!files.length) return null;
  if (preferred) {
    const exact = files.find((f) => f === preferred || slug(f.replace(/\.svg$/i, '')) === slug(preferred));
    if (exact) return exact;
  }
  return files.find((f) => /-icon\.svg$/i.test(f) && !/monochrome/i.test(f))
    || files.find((f) => !/monochrome/i.test(f))
    || files[0];
}

export function resolveLogoFromIndex(index, tile = {}) {
  if (!index) return null;
  for (const c of logoCandidates(tile)) {
    const entry = index.byShortname.get(c) || index.byFileStem.get(c) || index.byName.get(c);
    if (!entry) continue;
    const file = chooseFile(entry, tile.logoFile);
    if (!file) continue;
    return {
      name: entry.name,
      shortname: entry.shortname,
      file,
      src: `${entry.basePath}/logos/${file}`,
    };
  }
  return null;
}

async function resolveLogo(tile) {
  if (tile?.svg) return { name: tile.name || tile.key, shortname: tile.key, file: 'inline.svg', src: svgDataUrl(tile.svg), inline: tile.svg };
  const index = await loadLogoCatalog();
  return resolveLogoFromIndex(index, tile);
}

function prepareTile(el, tile, preset, fallbackGlyph, fsScale, state) {
  el.classList.remove('hasvg', 'skill-icon--svg', 'skill-icon--mask', 'skill-icon--pixel', 'skill-icon--chip', 'skill-icon--locked');
  el.classList.add('skill-icon');
  el.classList.toggle('skill-icon--chip', state === 'chip');
  el.classList.toggle('skill-icon--locked', state === 'soon' || state === 'locked');
  el.dataset.iconTheme = preset.mode;
  const tileBg = preset.bg || tile?.bg || '#6b6256';
  const tileFg = preset.fg || tile?.fg || '#fff';
  el.style.background = tileBg;
  el.style.color = tileFg;
  el.style.setProperty('--skill-icon-bg', tileBg);
  el.style.setProperty('--skill-icon-fg', tileFg);
  el.style.fontSize = Math.round(((tile && tile.fs) || 18) * fsScale) + 'px';
  el.textContent = '';
  const fallback = document.createElement('span');
  fallback.className = 'skill-icon__fallback';
  fallback.textContent = tile?.glyph || fallbackGlyph || '◆';
  el.appendChild(fallback);
}

function installBrand(el, logo, tile) {
  el.classList.add('hasvg', 'skill-icon--svg');
  el.textContent = '';
  if (logo.inline) {
    el.innerHTML = logo.inline;
  } else {
    const img = document.createElement('img');
    img.className = 'skill-icon__img';
    img.alt = '';
    img.decoding = 'async';
    img.loading = 'lazy';
    img.src = logo.src;
    el.appendChild(img);
  }
  el.style.background = tile?.logoBg || tile?.bg || '#fff';
}

function installMask(el, logo, preset) {
  el.classList.add('hasvg', 'skill-icon--mask');
  el.textContent = '';
  const mark = document.createElement('span');
  mark.className = 'skill-icon__mask';
  mark.style.background = preset.fg || '#ffd24a';
  mark.style.webkitMask = `${cssUrl(logo.src)} center / contain no-repeat`;
  mark.style.mask = `${cssUrl(logo.src)} center / contain no-repeat`;
  el.appendChild(mark);
}

async function rasterizeLogo(logo, preset) {
  const key = `${logo.src}|${preset.mode}|${preset.fg || ''}|${preset.pixelSize || 18}`;
  if (rasterCache.has(key)) return rasterCache.get(key);
  const p = new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => {
      const s = Math.max(8, Math.min(32, preset.pixelSize || 18));
      const c = document.createElement('canvas');
      c.width = s; c.height = s;
      const ctx = c.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, s, s);
      ctx.drawImage(img, 0, 0, s, s);
      if (preset.mode === 'pixel-mono') {
        ctx.globalCompositeOperation = 'source-in';
        ctx.fillStyle = preset.fg || '#ffd24a';
        ctx.fillRect(0, 0, s, s);
      }
      resolve(c.toDataURL('image/png'));
    };
    img.onerror = () => reject(new Error(`could not rasterize icon ${logo.src}`));
    img.src = logo.src;
  });
  rasterCache.set(key, p);
  return p;
}

async function installPixel(el, logo, preset) {
  el.classList.add('hasvg', 'skill-icon--pixel');
  el.textContent = '';
  const img = document.createElement('img');
  img.className = 'skill-icon__img skill-icon__img--pixel';
  img.alt = '';
  img.decoding = 'async';
  img.src = await rasterizeLogo(logo, preset);
  el.appendChild(img);
}

export function paintSkillIcon(el, tile = {}, options = {}) {
  if (!el) return;
  const preset = iconThemePreset(options.theme || iconTheme());
  const version = (paintVersions.get(el) || 0) + 1;
  paintVersions.set(el, version);
  prepareTile(el, tile, preset, options.fallbackGlyph, options.fsScale || 1, options.state);

  resolveLogo(tile).then(async (logo) => {
    if (!logo || paintVersions.get(el) !== version) return;
    if (preset.mode === 'mono') installMask(el, logo, preset);
    else if (preset.mode === 'pixel' || preset.mode === 'pixel-mono') await installPixel(el, logo, preset);
    else installBrand(el, logo, tile);
  }).catch(() => {
    // Fallback glyph is already rendered. Missing upstream logos should not break UI.
  });
}

export function themedTileColor(tile, theme = iconTheme()) {
  const preset = iconThemePreset(theme);
  return preset.bg || firstColor(tile?.bg) || tile?.bg || '#6b6256';
}
