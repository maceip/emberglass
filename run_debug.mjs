import { chromium } from 'playwright';
const CANARY = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const browser = await chromium.launch({ executablePath: CANARY, headless: false,
  args: ['--enable-unsafe-webgpu','--enable-features=WebGPU','--use-angle=metal','--no-first-run','--no-default-browser-check'] });
const page = await browser.newPage();
const lines = [];
page.on('console', m => { const t = m.text(); if (t.startsWith('DBG')) { lines.push(t); console.log(t); } });
page.on('pageerror', e => console.log('PAGEERR', String(e).slice(0,200)));
await page.goto('http://localhost:8013/debug.html', { waitUntil: 'domcontentloaded' });
const t0 = Date.now();
while (Date.now() - t0 < 300000) { if (lines.some(l => l.startsWith('DBG DONE') || l.startsWith('DBG ERROR'))) break; await page.waitForTimeout(2000); }
await browser.close();
