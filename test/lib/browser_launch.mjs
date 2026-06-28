import { existsSync } from 'node:fs';
import { chromium } from 'playwright';

const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linuxChrome = '/usr/local/bin/google-chrome';

export function chromeExecutable() {
  return process.env.CHROME_PATH || (existsSync(linuxChrome) ? linuxChrome : (existsSync(macCanary) ? macCanary : undefined));
}

export async function launchWebGpuBrowser({ headless = false } = {}) {
  const executablePath = chromeExecutable();
  return chromium.launch({
    ...(executablePath ? { executablePath } : {}),
    headless,
    args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--no-first-run', '--disable-dawn-features=timestamp_quantization'],
  });
}

export const webgpuLaunchArgs = ['--enable-unsafe-webgpu', '--enable-features=WebGPU'];
