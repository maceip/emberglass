// types.ts — shared shapes for substrate-style skill BLOCKS.
//
// A skill block is: a PORT (canonical action space), graded ADAPTERS (corpus generators
// per provider), a declarative CONTRACT (verification criteria as data), and a MANIFEST
// (block.json-shaped metadata). These types are the seam the rest of the system speaks.

export interface Op {
  name: string;
  params: string[];
  ret?: string;
}

export interface Spec {
  scope: string;
  ops: Op[];
}

// A training/eval pair: [request, macro]. macro === 'OUT_OF_SCOPE' marks a bounce.
export type Example = [string, string];

// What an adapter's generator returns: the trainable corpus + a held-out eval split.
export interface GenResult {
  examples: Example[];
  eval: Example[];
}

// CONTRACT pieces (substrate S7): predicates over an emitted macro, evaluated by checkContract.
export interface Assertion {
  id: string;
  describe: string;
  holds: (macro: string, spec?: Spec) => boolean;
}
export interface Forbidden {
  id: string;
  describe: string;
  violatedBy: (macro: string, spec?: Spec) => boolean;
}
export interface ContractExtra {
  assertions?: Assertion[];
  forbidden?: Forbidden[];
}

// The definition skills.js#buildSkill consumes. Identical surface to the old inline DEF.
export interface SkillDef {
  key: string;
  label: string;
  icon: string;
  domain: string;
  scope: string;
  desc: string;
  suggest: string;
  ops: Op[];
  context?: string;
  examplesFn?: () => GenResult;
  contract?: ContractExtra;
}

// ── provider profiles + data-driven intents ─────────────────────────────────
// An adapter = a PORT (canonical ops) + a ProviderProfile (provider conventions + vocab) +
// a set of INTENTS (declarative recipes). The generator draws vocab from the profile and
// renders each intent into a [request, macro] pair, so swapping providers is data, not code.

export type SlotName = 'person' | 'topic' | 'when' | 'window' | 'label' | 'dur' | 'rsvp';

export interface WhenSlot { nat: string; iso: string }
export interface WindowSlot { nat: string; after: string; before: string }
export interface RsvpSlot { resp: string; verb: string }

export interface ProviderPools {
  people: string[];
  topics: string[];
  whens: WhenSlot[];
  windows: WindowSlot[];
  labels: string[];
  durations: number[];
  rsvps: RsvpSlot[];
}

export interface ProviderProfile {
  provider: string; // 'google'
  label: string; // 'Google (Gmail + Calendar)'
  // grounding: where the conventions were lifted from (the checked-in Discovery snapshot)
  discovery: { source: string[]; revision: string; note: string };
  conventions: { timeFormat: string; searchSyntax: string };
  // canonical PORT op -> provider Discovery method id (for the future write-layer)
  opMap: Record<string, string>;
  pools: ProviderPools;
}

// A declarative recipe over the PORT. `draw` lists the slots to sample (in RNG order),
// `phrasings` are natural-language variants, `macro` is the templated, spec-valid output.
// Templates use ${slot} / ${slot.field} / ${end} tokens resolved by the generator.
export interface Intent {
  n: number;
  draw: SlotName[];
  phrasings: string[];
  macro: string;
}

// block.json-shaped manifest (consumed by the future transfer/CATALOG seam, not buildSkill yet).
export interface BlockGrade {
  grade: 'nursery' | 'elementary' | 'graduated';
  adapter: string;
  capabilities: string[];
}
export interface BlockManifest {
  block: string;
  portVersion: string;
  grades: BlockGrade[];
  gates: unknown[];
  consumers: string[];
}
