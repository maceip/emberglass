/* Headless sanity check for the Swiss-army-knife HUD: the skills registry,
   the Train-pane skill picker, the macro verifier readout, number-key quick
   equip, and the locked/forged slot states. No model load needed.
   Output screenshots: /tmp/ui_knife_*.png  Server: http://localhost:8016 */
import { chromium } from 'playwright';

const URL = process.env.BASE_URL || 'http://localhost:8016';
const b = await chromium.launch({ headless: true, args: ['--no-first-run'] });
const p = await b.newPage({ viewport: { width: 1180, height: 940 } });
const errs = [];
p.on('pageerror', (e) => errs.push(String(e).slice(0, 200)));
await p.goto(URL + '/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__eg && Array.isArray(window.__eg.SKILLS), null, { timeout: 8000 });

const checks = {};

// 0) registry has ≥2 service skills, each with a spec.
checks.registry = await p.evaluate(() => {
  const S = window.__eg.SKILLS;
  return { n: S.length, keys: S.map((s) => s.key), allHaveSpec: S.every((s) => s.spec && s.spec.ops.length) };
});

// 1) Fresh state → one locked slot per untrained service skill.
checks.locked = await p.evaluate(() => {
  const slots = [...document.querySelectorAll('#knifeSlots .kslot')];
  return { count: slots.length, allLocked: slots.length > 0 && slots.every((s) => s.classList.contains('kslot--locked')) };
});

// 2) Skill picker renders a button per skill; default = first; clicking Music swaps examples.
await p.click('#tabTrain'); await p.waitForTimeout(150);
checks.picker = await p.evaluate(() => {
  const btns = [...document.querySelectorAll('#skillPicker .skillpick__btn')];
  return { btnCount: btns.length, onKey: btns.find((x) => x.classList.contains('on'))?.dataset.key };
});
const macroBefore = await p.evaluate(() => document.querySelector('#guidedList .skill-macro')?.textContent || '');
await p.evaluate(() => window.__eg.selectSkill('music'));
await p.waitForTimeout(60);
checks.musicSelect = await p.evaluate(() => {
  const title = document.querySelector('#skillTitle')?.textContent || '';
  const macros = [...document.querySelectorAll('#guidedList .skill-macro')].map((m) => m.textContent).join('\n');
  return { title, hasMusicOp: /find_track|play_track|set_volume/.test(macros) };
});
checks.macroSwapped = (await p.evaluate(() => document.querySelector('#guidedList .skill-macro')?.textContent || '')) !== macroBefore;
await p.evaluate(() => window.__eg.selectSkill('inbox-calendar'));

// 3) Macro verifier: ok / oos / bad against each skill's spec.
checks.verify = await p.evaluate(() => {
  const eg = window.__eg;
  const inbox = eg.skillByKey('inbox-calendar').spec;
  const music = eg.skillByKey('music').spec;
  const ok = eg.verifyMacro('compose_email(to="x", subject="y", body="z")\ncreate_event(title="a", start="b", end="c", remind_min=10)', inbox);
  const oos = eg.verifyMacro('OUT_OF_SCOPE', inbox);
  const badArg = eg.verifyMacro('compose_email(to="x", nope="y")', inbox);
  const badOp = eg.verifyMacro('send_text(to="x")', inbox);
  const musicOk = eg.verifyMacro('set_volume(level=30)', music);
  return {
    ok: ok.status === 'ok' && ok.n === 2,
    oos: oos.status === 'oos',
    badArg: badArg.status === 'bad',
    badOp: badOp.status === 'bad',
    musicOk: musicOk.status === 'ok',
  };
});

// 4) Seed two forged service skills + one BYOD skill → no locked slots remain,
//    key hints appear, equipped reflects state.
await p.evaluate(async () => {
  const s = window.__eg.store;
  for (const r of s.listRuns()) await s.deleteRun(r.id);
  const blob = new Blob([new Uint8Array(64)]);
  const mk = (name, kind, base, loss, steps) => ({ id: s.newId(), name, kind, base, createdAt: Date.now(), steps, rank: 16, finalLoss: loss });
  const a = mk('inbox-calendar', 'guided', 'inbox-calendar', 0.08, 98);
  const m = mk('music', 'guided', 'music', 0.12, 70);
  const n = mk('my-notes', 'own', 'my-notes', 0.31, 36);
  await s.saveRun(a, { safetensors: blob, configJson: '{}' });
  await s.saveRun(m, { safetensors: blob, configJson: '{}' });
  await s.saveRun(n, { safetensors: blob, configJson: '{}' });
  window.__eg.state.activeRunId = a.id;
  window.__eg.renderHistory(); // renders the inventory list + (via tail call) the knife bar
});
checks.forged = await p.evaluate(() => {
  const slots = [...document.querySelectorAll('#knifeSlots .kslot')];
  return {
    count: slots.length,
    anyLocked: slots.some((s) => s.classList.contains('kslot--locked')),
    keys: slots.map((s) => s.querySelector('.kslot__key')?.textContent || null),
    equipped: slots.filter((s) => s.classList.contains('equipped')).map((s) => s.querySelector('.kslot__name')?.textContent),
    icons: slots.map((s) => s.querySelector('.kslot__icon')?.textContent),
  };
});

// 4b) Inventory (left rail): each forged knife renders as a loot item with a
//     rarity tier, level badge, and an EQUIPPED tag on the active one.
checks.inventory = await p.evaluate(() => {
  const items = [...document.querySelectorAll('#historyList .item')];
  return {
    count: items.length,
    countBadge: document.getElementById('historyCount')?.textContent,
    rarities: items.map((i) => i.dataset.rarity),
    levels: items.map((i) => i.querySelector('.item__lv')?.textContent),
    equippedTag: items.filter((i) => i.classList.contains('active') && i.querySelector('.item__tag'))
      .map((i) => i.querySelector('.item__name')?.textContent),
    allHaveFrame: items.every((i) => i.querySelector('.item__frame .item__icon')),
  };
});

// 5) Number-key quick-equip (FPS slots): press "2" → resolves the 2nd forged
//    knife. (Completing the hot-swap needs a loaded model/GPU, which headless
//    lacks; we assert the keybind→index→run resolution via lastEquipIntent.)
const secondId = await p.evaluate(() => window.__eg.store.listRuns()[1]?.id);
await p.evaluate(() => document.body.focus());
await p.keyboard.press('2');
await p.waitForTimeout(80);
const resolved = await p.evaluate(() => window.__eg.lastEquipIntent);
// a keydown originating inside a text field must NOT trigger a quick-equip
const ignoredInInput = await p.evaluate(() => {
  const before = window.__eg.lastEquipIntent;
  const ta = document.getElementById('prompt');
  ta.dispatchEvent(new KeyboardEvent('keydown', { key: '1', bubbles: true }));
  return window.__eg.lastEquipIntent === before;
});
checks.keybind = { secondId, resolved, ignoredInInput, switched: resolved === secondId };

await p.screenshot({ path: '/tmp/ui_knife_train.png', fullPage: true });
await p.click('#tabInfer'); await p.waitForTimeout(120);
await p.screenshot({ path: '/tmp/ui_knife_infer.png', fullPage: true });
await b.close();

const pass =
  checks.registry.n >= 2 && checks.registry.allHaveSpec &&
  checks.locked.allLocked && checks.locked.count === checks.registry.n &&
  checks.picker.btnCount === checks.registry.n && checks.picker.onKey === 'inbox-calendar' &&
  checks.musicSelect.hasMusicOp && checks.macroSwapped &&
  checks.verify.ok && checks.verify.oos && checks.verify.badArg && checks.verify.badOp && checks.verify.musicOk &&
  checks.forged.count === 3 && !checks.forged.anyLocked && checks.forged.keys[0] === '1' &&
  checks.forged.equipped.includes('inbox-calendar') &&
  checks.inventory.count === 3 && checks.inventory.countBadge === '3' &&
  checks.inventory.allHaveFrame && checks.inventory.equippedTag.includes('inbox-calendar') &&
  checks.inventory.rarities.every(Boolean) &&
  checks.keybind.switched && checks.keybind.ignoredInInput &&
  errs.length === 0;

for (const [k, v] of Object.entries(checks)) console.log(k.padEnd(12), JSON.stringify(v));
console.log('pageerrors  ', errs.length ? errs : 'none');
console.log(pass ? 'KNIFE_CHECK_PASS' : 'KNIFE_CHECK_FAIL');
process.exit(pass ? 0 : 1);
