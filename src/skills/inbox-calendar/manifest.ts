// manifest.ts — block.json-shaped metadata for the Inbox & Calendar block.
//
// Declared now so the transfer/CATALOG seam (later brick) has something to package and the
// detector can map an origin -> provider grade. Not yet wired into buildSkill's output.
import type { BlockManifest } from '../types.ts';

export const MANIFEST: BlockManifest = {
  block: 'inbox-calendar',
  portVersion: '0.1.0',
  grades: [
    {
      grade: 'graduated',
      adapter: 'google',
      capabilities: ['iso-times', 'balanced-op-coverage', 'phrasing-variety', 'held-out-eval', 'oos-near-misses'],
    },
    {
      grade: 'elementary',
      adapter: 'microsoft',
      capabilities: ['canonical-port', 'full-op-coverage', 'graph-dateTime', 'patternedRecurrence'],
    },
    {
      grade: 'elementary',
      adapter: 'zoho',
      capabilities: ['canonical-port', 'full-op-coverage', 'rfc3339', 'rrule'],
    },
  ],
  gates: [],
  consumers: ['runtime macro compiler', 'LoRA guided trainer', 'contract verifier'],
};
