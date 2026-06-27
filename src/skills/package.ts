// package.ts — the transfer seam: export/import/attest a skill, and build a CATALOG.
//
// Pure functions that take their inputs (a built skill, its providers, its lessons) so this
// module has no dependency on the registry and stays trivially testable. skills.js provides
// thin wrappers that bind these to the live SKILLS / PROVIDERS / lessons store.
import type {
  Attestation, Catalog, CatalogEntry, Example, Lesson, ProviderProfile, SkillPackage,
} from './types.ts';
import { fingerprint } from './fingerprint.ts';

// the structural shape we need off a built skill (see skills.js#buildSkill output)
export interface BuiltSkillLike {
  key: string;
  label: string;
  system: string;
  spec: { scope: string; ops: { name: string; params: string[]; ret?: string }[] };
  contract: { assertions?: { id: string; describe: string }[]; forbidden?: { id: string; describe: string }[] };
  eval?: Example[];
}

export interface ExportOpts {
  portVersion?: string;
  providers?: ProviderProfile[];
  lessons?: Lesson[];
}

export function exportSkillPackage(skill: BuiltSkillLike, opts: ExportOpts = {}): SkillPackage {
  const base = {
    format: 'eg-skill/1' as const,
    block: skill.key,
    label: skill.label,
    portVersion: opts.portVersion ?? '0.1.0',
    port: { scope: skill.spec.scope, ops: skill.spec.ops },
    system: skill.system,
    contract: {
      assertions: (skill.contract.assertions || []).map((a) => ({ id: a.id, describe: a.describe })),
      forbidden: (skill.contract.forbidden || []).map((f) => ({ id: f.id, describe: f.describe })),
    },
    providers: (opts.providers || []).map((p) => ({
      id: p.provider, label: p.label, conventions: p.conventions, opMap: p.opMap,
    })),
    eval: skill.eval || [],
    lessons: opts.lessons || [],
  };
  return { ...base, fingerprint: fingerprint(base) };
}

export interface ImportResult {
  ok: boolean;
  violations: string[];
  recomputed: string;
  contract: unknown; // live, re-bound contract (with predicates) if the family is known here
  pkg: SkillPackage | null;
}

// resolveContract re-binds the live (executable) contract by family — descriptors alone can't run.
export function importSkillPackage(
  pkg: SkillPackage,
  resolveContract?: (block: string) => unknown,
): ImportResult {
  const violations: string[] = [];
  if (!pkg || pkg.format !== 'eg-skill/1') violations.push('bad-format');
  let recomputed = '';
  if (pkg) {
    const { fingerprint: _fp, ...rest } = pkg;
    recomputed = fingerprint(rest);
    if (pkg.fingerprint !== recomputed) violations.push('fingerprint-mismatch');
  } else {
    violations.push('empty');
  }
  const contract = pkg && resolveContract ? resolveContract(pkg.block) : undefined;
  return { ok: violations.length === 0, violations, recomputed, contract, pkg: pkg ?? null };
}

// Attestor stub. issuedAt is metadata only (not part of the verified payload), so leaving it
// fixed keeps attestations deterministic for tests; callers may pass a real timestamp.
export function attest(pkg: SkillPackage, issuedAt = '1970-01-01T00:00:00Z'): Attestation {
  return { scheme: 'sha256-fingerprint/stub', block: pkg.block, fingerprint: pkg.fingerprint, issuedAt };
}

export function verifyAttestation(pkg: SkillPackage, att: Attestation): boolean {
  if (!att || att.scheme !== 'sha256-fingerprint/stub') return false;
  const { fingerprint: fp, ...rest } = pkg;
  return att.block === pkg.block && att.fingerprint === fp && fp === fingerprint(rest);
}

export function buildCatalog(
  skills: BuiltSkillLike[],
  exportFn: (skill: BuiltSkillLike) => SkillPackage,
  version = 1,
): Catalog {
  const entries: CatalogEntry[] = skills.map((s) => {
    const pkg = exportFn(s);
    return {
      block: s.key,
      label: s.label,
      portVersion: pkg.portVersion,
      ops: s.spec.ops.length,
      evalCount: (s.eval || []).length,
      providers: pkg.providers.map((p) => p.id),
      fingerprint: pkg.fingerprint,
    };
  });
  return { format: 'eg-catalog/1', version, skills: entries };
}
