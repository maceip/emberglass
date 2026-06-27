// providers/index.ts — the registry of calendar providers behind the canonical port.
//
// Keyed by provider id so the future origin->provider detector can resolve a logged-in tab
// (mail.google.com / outlook.office.com / mail.zoho.com) to the right executor profile.
import type { ProviderProfile } from '../../types.ts';
import { GOOGLE_PROFILE } from './google.ts';
import { OUTLOOK_PROFILE } from './outlook.ts';
import { ZOHO_PROFILE } from './zoho.ts';

export const PROVIDERS: Record<string, ProviderProfile> = {
  google: GOOGLE_PROFILE,
  microsoft: OUTLOOK_PROFILE,
  zoho: ZOHO_PROFILE,
};

export const DEFAULT_PROVIDER = 'google';
