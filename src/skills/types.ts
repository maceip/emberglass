// types.ts — shared shapes for substrate-style skill BLOCKS.
//
// A skill block is: a PORT (canonical action space), graded ADAPTERS (corpus generators
// per provider), a declarative CONTRACT (verification criteria as data), and a MANIFEST
// (block.json-shaped metadata). These types are the seam the rest of the system speaks.

// Risk ladder for a write, lowest to highest. Used to classify ops and whole plans.
export type RiskLevel = 'read-only' | 'reversible-write' | 'sensitive-write';

export interface Op {
  name: string;
  params: string[];
  ret?: string;
  // canonical action semantics (provider-independent; specSig ignores these so the corpus is
  // unaffected). A read has no side effect; a write does. capability is the canonical scope a
  // provider executor would need; idempotent writes can be safely retried/deduped.
  effect?: 'read' | 'write';
  capability?: string;
  idempotent?: boolean;
  risk?: RiskLevel;
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
  conventions: {
    timeFormat: string; // how the provider represents instants (RFC3339, dateTime+timeZone, …)
    searchSyntax: string; // mailbox query dialect (gmail-q, kql, zoho-search, …)
    recurrence?: string; // repeat-rule encoding (RRULE / patternedRecurrence / …)
  };
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

// FoT (forest-of-thoughts) lesson: a distilled, reusable insight keyed by skill family.
// `evidence` links a lesson back to the contract id (or signal) that motivated it, so the
// thing we ENFORCE and the thing we EXPLAIN never drift apart.
export interface Lesson {
  id: string;
  family: string;
  origin: 'seed' | 'learned';
  text: string;
  evidence?: string;
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

// ── transfer seam: shareable, content-addressed skill packages ───────────────
// A SkillPackage is the verifiable surface of a skill, minus the LoRA weights (which ship
// alongside via lora_export). Contract predicates can't be serialized, so we carry their
// DESCRIPTORS and re-bind the live implementation by family on import.
export interface ContractDescriptor { id: string; describe: string }
export interface ProviderExport {
  id: string;
  label: string;
  conventions: ProviderProfile['conventions'];
  opMap: Record<string, string>;
}
export interface SkillPackage {
  format: 'eg-skill/1';
  block: string;
  label: string;
  portVersion: string;
  port: { scope: string; ops: Op[] };
  system: string;
  contract: { assertions: ContractDescriptor[]; forbidden: ContractDescriptor[] };
  providers: ProviderExport[];
  eval: Example[];
  lessons: Lesson[];
  fingerprint: string; // sha256 over the package with this field removed
}

// Attestor stub: today just binds a fingerprint under a scheme tag. The seam is shaped so a
// real attestation (e.g. a TEE unified-quote) can drop in behind verifyAttestation later.
export interface Attestation {
  scheme: 'sha256-fingerprint/stub';
  block: string;
  fingerprint: string;
  issuedAt: string;
}

export interface CatalogEntry {
  block: string;
  label: string;
  portVersion: string;
  ops: number;
  evalCount: number;
  providers: string[];
  fingerprint: string;
}
export interface Catalog {
  format: 'eg-catalog/1';
  version: number;
  skills: CatalogEntry[];
}

// ── action layer (planner / dry-run only) ───────────────────────────────────
// A verified macro compiles to an ActionPlan: a typed, provider-resolved program that is
// SHOWN, never executed by the model. The only executor today is a dry-run that performs no
// I/O. Real provider/DOM executors are a separate, approval-gated milestone.
export interface ActionArg {
  key: string;
  kind: 'string' | 'number' | 'ref';
  value: string;
  refBase?: string; // for kind 'ref': the binding var this consumes (e.g. 't' from 's.start')
}
export interface ActionStep {
  index: number;
  op: string;
  effect: 'read' | 'write';
  binds: string | null; // LHS variable this step produces, if any
  args: ActionArg[];
  provider: string;
  providerMethod: string; // resolved from the ProviderProfile opMap (or '(unmapped)')
  capability: string;
  idempotent: boolean;
  risk: RiskLevel;
  idempotencyKey: string;
  dependsOn: number[]; // indices of prior steps whose binding this step references
}
export interface ActionPlan {
  block: string;
  provider: string;
  contractOk: boolean; // gate: an executor must refuse a plan whose contract didn't pass
  risk: RiskLevel; // max over steps
  requiredCapabilities: string[]; // disclosure of scopes a real executor would need
  steps: ActionStep[];
  summary: string[]; // human-readable battle plan
  fingerprint: string; // content hash of the plan (sans this field)
}

// Receipts are produced by an Executor. In dry-run they are always 'simulated'.
export interface Receipt {
  step: number;
  op: string;
  provider: string;
  method: string;
  status: 'simulated';
  idempotencyKey: string;
  at: string;
  detail: string;
}
export interface Executor {
  id: string;
  canExecute: (plan: ActionPlan) => boolean;
  execute: (plan: ActionPlan) => Receipt[];
}
