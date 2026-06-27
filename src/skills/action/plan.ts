// action/plan.ts — compile a VERIFIED macro into a typed, provider-resolved ActionPlan.
//
// Pure and deterministic. The model is not involved here: it already emitted the macro. This
// step parses the calls (passed in), looks up canonical op semantics, resolves each op to the
// provider's method via the opMap, wires read->write data flow, classifies risk, and discloses
// the capabilities a real executor would need. It NEVER executes anything.
import type { ActionArg, ActionPlan, ActionStep, Op, ProviderProfile, RiskLevel } from '../types.ts';
import { fingerprint, sha256Hex } from '../fingerprint.ts';

// a parsed macro call as produced by skills.js#parseMacroCalls
export interface ParsedCall {
  op: string;
  keys: string[];
  binds: string | null;
  args: ActionArg[];
}

export interface CompileInput {
  block: string;
  calls: ParsedCall[];
  ops: Op[];
  profile: ProviderProfile | null;
  contractOk: boolean;
}

const RISK_RANK: Record<RiskLevel, number> = { 'read-only': 0, 'reversible-write': 1, 'sensitive-write': 2 };
const RISK_AT: RiskLevel[] = ['read-only', 'reversible-write', 'sensitive-write'];

function maxRisk(a: RiskLevel, b: RiskLevel): RiskLevel {
  return RISK_AT[Math.max(RISK_RANK[a], RISK_RANK[b])];
}

function renderArg(a: ActionArg): string {
  if (a.kind === 'string') return `${a.key}="${a.value}"`;
  if (a.kind === 'ref') return `${a.key}=$${a.value}`;
  return `${a.key}=${a.value}`;
}

export function compilePlan(input: CompileInput): ActionPlan {
  const { block, calls, ops, profile, contractOk } = input;
  const opByName = new Map(ops.map((o) => [o.name, o]));
  const provider = profile ? profile.provider : 'unknown';
  const opMap = profile ? profile.opMap : {};

  const bindToIndex = new Map<string, number>();
  const steps: ActionStep[] = [];
  let risk: RiskLevel = 'read-only';
  const caps = new Set<string>();

  calls.forEach((c, index) => {
    const meta = opByName.get(c.op);
    const effect = (meta?.effect ?? 'write') as 'read' | 'write';
    const capability = meta?.capability ?? 'unknown';
    const idempotent = meta?.idempotent ?? false;
    const stepRisk: RiskLevel = meta?.risk ?? 'sensitive-write'; // unknown op → treat as most dangerous
    const providerMethod = opMap[c.op] ?? '(unmapped)';

    const dependsOn: number[] = [];
    for (const a of c.args) {
      if (a.kind === 'ref' && a.refBase && bindToIndex.has(a.refBase)) {
        const dep = bindToIndex.get(a.refBase)!;
        if (!dependsOn.includes(dep)) dependsOn.push(dep);
      }
    }

    const argSig = c.args.map(renderArg).join(', ');
    const idempotencyKey = sha256Hex(`${provider}|${c.op}|${argSig}`).slice(0, 12);

    steps.push({
      index, op: c.op, effect, binds: c.binds, args: c.args,
      provider, providerMethod, capability, idempotent, risk: stepRisk,
      idempotencyKey, dependsOn,
    });

    risk = maxRisk(risk, stepRisk);
    caps.add(capability);
    if (c.binds) bindToIndex.set(c.binds, index);
  });

  const summary = steps.map((s) => {
    const tag = s.effect === 'read' ? 'read' : s.risk;
    const dep = s.dependsOn.length ? ` ⟵ step ${s.dependsOn.map((d) => d + 1).join(',')}` : '';
    const lhs = s.binds ? `${s.binds} = ` : '';
    return `${s.index + 1}. ${lhs}${s.op}(${s.args.map(renderArg).join(', ')}) → ${s.providerMethod} [${tag}]${dep}`;
  });

  const base = {
    block,
    provider,
    contractOk,
    risk,
    requiredCapabilities: [...caps].filter((c) => c !== 'unknown').sort(),
    steps,
    summary,
  };
  return { ...base, fingerprint: fingerprint(base) };
}
