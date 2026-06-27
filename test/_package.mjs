/* Transfer-seam gate.
 * Proves a skill can be PACKAGED, fingerprinted, attested, shared, and independently VERIFIED:
 *   - sha256 matches the FIPS-180-4 "abc" vector (our hash is correct)
 *   - export is deterministic; fingerprint covers the payload
 *   - attest -> verify round-trips; any tamper flips verification + import to a mismatch
 *   - import re-binds the live contract and the held-out eval still passes it
 *   - CATALOG.json is in sync with the registry */
import { readFileSync } from 'node:fs';
import { SKILLS, checkContract, exportSkill, importSkill, attest, verifyAttestation, catalog } from '../src/skills.js';
import { sha256Hex } from '../src/skills/fingerprint.ts';

const fails = [];
const ok = (cond, msg) => { if (!cond) fails.push(msg); };

// 0) hash correctness
ok(sha256Hex('abc') === 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad', 'sha256("abc") wrong');
ok(sha256Hex('') === 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'sha256("") wrong');

// 1) export the calendar package
const pkg = exportSkill('inbox-calendar');
ok(pkg && pkg.format === 'eg-skill/1', 'bad package format');
ok(pkg.providers.length === 3, `expected 3 providers, got ${pkg.providers?.length}`);
ok(pkg.eval.length >= 8, `expected held-out eval, got ${pkg.eval?.length}`);
ok(pkg.lessons.length >= 5, `expected lessons, got ${pkg.lessons?.length}`);
ok(/^[0-9a-f]{64}$/.test(pkg.fingerprint), 'fingerprint not 64 hex');

// 2) export is deterministic
ok(exportSkill('inbox-calendar').fingerprint === pkg.fingerprint, 'export not deterministic');

// 3) attest -> verify round-trip
const att = attest(pkg);
ok(verifyAttestation(pkg, att), 'attestation should verify');

// 4) tamper detection
const tampered = JSON.parse(JSON.stringify(pkg));
tampered.system = tampered.system + ' (sneaky edit)';
ok(!verifyAttestation(tampered, att), 'tampered package must fail attestation');
ok(importSkill(tampered).violations.includes('fingerprint-mismatch'), 'import must flag fingerprint-mismatch');

// 5) import re-binds the live contract; held-out eval still satisfies it (verify a tune w/o the app)
const imp = importSkill(pkg);
ok(imp.ok, `clean import should be ok: ${imp.violations.join(',')}`);
ok(imp.contract && Array.isArray(imp.contract.assertions), 'import did not re-bind a live contract');
if (imp.contract) {
  const spec = { scope: pkg.port.scope, ops: pkg.port.ops };
  for (const [req, macro] of pkg.eval) {
    if (!checkContract(imp.contract, macro, spec).ok) fails.push(`re-bound contract rejects eval pair → "${req}"`);
  }
}

// 6) CATALOG.json in sync
const cat = catalog();
ok(cat.skills.length === SKILLS.length, `catalog skills ${cat.skills.length} != ${SKILLS.length}`);
const calEntry = cat.skills.find((e) => e.block === 'inbox-calendar');
ok(calEntry && calEntry.providers.join(',') === 'google,microsoft,zoho', 'calendar catalog providers wrong');
let onDisk = '';
try { onDisk = readFileSync(new URL('../CATALOG.json', import.meta.url), 'utf8'); } catch { /* missing */ }
ok(onDisk === JSON.stringify(cat, null, 2) + '\n', 'CATALOG.json stale — run `npm run catalog:build`');

console.log(`package: ${pkg.block}  fp ${pkg.fingerprint.slice(0, 12)}…  providers ${pkg.providers.length}  eval ${pkg.eval.length}  lessons ${pkg.lessons.length}`);
console.log(`catalog: ${cat.skills.length} skills`);
if (fails.length) { console.log('\nFAILURES:'); for (const f of fails) console.log('  - ' + f); }
console.log(fails.length ? 'PACKAGE_FAIL' : 'PACKAGE_PASS');
process.exit(fails.length ? 1 : 0);
