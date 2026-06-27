// detect.ts — origin -> provider detection.
//
// Given a logged-in tab's origin/URL from the extension capture bridge, resolve which
// calendar provider profile to use. This starts as a checked-in host map and keeps
// the same two-function boundary for richer detectors such as account hints or
// manifest sniffing.
import type { ProviderProfile } from '../types.ts';
import { PROVIDERS } from './providers/index.ts';

// host (or parent domain) -> provider id in the PROVIDERS registry
export const ORIGIN_PROVIDERS: Record<string, string> = {
  'mail.google.com': 'google',
  'calendar.google.com': 'google',
  'gmail.com': 'google',
  'outlook.office.com': 'microsoft',
  'outlook.office365.com': 'microsoft',
  'outlook.live.com': 'microsoft',
  'mail.zoho.com': 'zoho',
  'calendar.zoho.com': 'zoho',
  'zoho.com': 'zoho',
};

function hostOf(input: string): string {
  try {
    return new URL(input).hostname.toLowerCase();
  } catch {
    return String(input).toLowerCase().replace(/^https?:\/\//, '').split('/')[0].split(':')[0];
  }
}

// returns a provider id ('google' | 'microsoft' | 'zoho') or null if the origin is unknown
export function originToProvider(origin: string): string | null {
  if (!origin) return null;
  const host = hostOf(origin);
  if (host in ORIGIN_PROVIDERS) return ORIGIN_PROVIDERS[host];
  for (const domain of Object.keys(ORIGIN_PROVIDERS)) {
    if (host === domain || host.endsWith('.' + domain)) return ORIGIN_PROVIDERS[domain];
  }
  return null;
}

export function providerForOrigin(origin: string): ProviderProfile | null {
  const id = originToProvider(origin);
  return id ? (PROVIDERS[id] ?? null) : null;
}
