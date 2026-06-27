/* Detector gate. origin/URL -> provider id, and the id resolves to a real ProviderProfile.
 * Closes the loop with the providers brick: anything we detect must be executable. */
import { originToProvider, providerForOrigin } from '../src/skills.js';
import { PROVIDERS } from '../src/skills/inbox-calendar/providers/index.ts';

const fails = [];
const cases = [
  ['https://mail.google.com/mail/u/0/#inbox', 'google'],
  ['calendar.google.com', 'google'],
  ['https://outlook.office.com/calendar/view/week', 'microsoft'],
  ['outlook.office365.com', 'microsoft'],
  ['https://outlook.live.com/owa/', 'microsoft'],
  ['https://mail.zoho.com', 'zoho'],
  ['us2.mail.zoho.com', 'zoho'],
  ['https://example.com/whatever', null],
  ['', null],
];

for (const [origin, want] of cases) {
  const got = originToProvider(origin);
  if (got !== want) fails.push(`originToProvider(${JSON.stringify(origin)}) -> ${got} (want ${want})`);
}

// every detected provider id must resolve to a real, registered profile
for (const [origin, want] of cases) {
  if (!want) continue;
  const prof = providerForOrigin(origin);
  if (!prof || prof.provider !== want) fails.push(`providerForOrigin(${JSON.stringify(origin)}) did not resolve to ${want}`);
  if (!(want in PROVIDERS)) fails.push(`detected id '${want}' missing from PROVIDERS registry`);
}

console.log(`detector: ${cases.length} cases, ${Object.keys(PROVIDERS).length} providers wired`);
if (fails.length) { console.log('\nFAILURES:'); for (const f of fails) console.log('  - ' + f); }
console.log(fails.length ? 'DETECT_FAIL' : 'DETECT_PASS');
process.exit(fails.length ? 1 : 0);
