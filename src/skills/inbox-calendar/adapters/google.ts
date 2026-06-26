// adapters/google.ts — the "graduated" Google adapter.
//
// Now a thin binding: it wires the Google ProviderProfile (vocab + conventions, lifted from
// the checked-in Discovery snapshot) to the family's declarative INTENTS and runs the generic
// generator. The emitted corpus is byte-identical to the previous hand-rolled generator — the
// seed is pinned so swapping in this data-driven path changes structure, not output.
import type { GenResult } from '../../types.ts';
import { GOOGLE_PROFILE } from '../providers/google.ts';
import { INTENTS, OOS } from '../intents.ts';
import { generateCorpus } from '../generate.ts';

const SEED = 'inbox-calendar:v2-iso';

export function genCalendar(): GenResult {
  return generateCorpus(SEED, GOOGLE_PROFILE, INTENTS, OOS);
}
