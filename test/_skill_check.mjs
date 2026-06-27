/* Headless sanity check for the skill HUD: the skills registry,
   the Train-pane skill picker, the macro verifier readout, number-key quick
   equip, and the locked/forged slot states. No model load needed.
   Output screenshots: /tmp/ui_skill_*.png  Server: http://localhost:8016 */
import { chromium } from 'playwright';

const URL = process.env.BASE_URL || 'http://localhost:8016';
const b = await chromium.launch({ headless: true, args: ['--no-first-run'] });
const p = await b.newPage({ viewport: { width: 1180, height: 940 } });
const errs = [];
p.on('pageerror', (e) => errs.push(String(e).slice(0, 200)));
await p.goto(URL + '/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__eg && Array.isArray(window.__eg.SKILLS), null, { timeout: 8000 });

const checks = {};

// 0) registry has the service skills, each with a spec; corpus is 500+ examples.
checks.registry = await p.evaluate(() => {
  const S = window.__eg.SKILLS;
  let total = 0; for (const s of S) total += s.examples.length;
  return {
    n: S.length, keys: S.map((s) => s.key), allHaveSpec: S.every((s) => s.spec && s.spec.ops.length),
    corpus: total, hasGithub: S.some((s) => s.key === 'github'),
    dockTiles: window.__eg.POPULAR_2026.length,
  };
});

// 1) Fresh dock → the two functional skills show as "forge" tiles; popular
//    services render as dimmed locked tiles with generated brand icons.
checks.dockFresh = await p.evaluate(() => {
  const tiles = [...document.querySelectorAll('#dockSlots .dock__tile')];
  const by = (s) => tiles.filter((t) => t.dataset.state === s).length;
  const google = tiles.find((t) => t.dataset.key === 'google');
  return {
    total: tiles.length, forge: by('forge'), locked: by('locked'), owned: by('owned'), equipped: by('equipped'),
    inboxForge: tiles.find((t) => t.dataset.key === 'inbox-calendar')?.dataset.state,
    iconBg: google?.querySelector('.dock__glyph')?.style.background || '',
  };
});

// 1b) Single-view RPG: no tabs; the inventory "Learn" button opens the training menu.
checks.singleView = await p.evaluate(() => ({
  noTabs: !document.getElementById('tabInfer') && !document.getElementById('tabTrain'),
  trainerHiddenAtStart: !!document.getElementById('trainer')?.hidden,
  hasLearnBtn: !!document.getElementById('learnBtn'),
  hasLearnCta: !!document.getElementById('learnCta'),
}));

// 2) Open Learn menu → skill picker renders a button per skill; default = first; Music swaps examples.
await p.click('#learnBtn'); await p.waitForTimeout(150);
checks.singleView.trainerOpens = await p.evaluate(() => !document.getElementById('trainer')?.hidden);
checks.picker = await p.evaluate(() => {
  const btns = [...document.querySelectorAll('#skillPicker .skillpick__btn')];
  return {
    btnCount: btns.length, onKey: btns.find((x) => x.classList.contains('on'))?.dataset.key,
    iconStyled: (btns[0]?.querySelector('.skillpick__icon')?.style.background || '').length > 0,
    hasMeta: !!btns[0]?.querySelector('.skillpick__txt i'),
  };
});
checks.calendarTrain = await p.evaluate(() => {
  const rules = document.getElementById('surfaceRules')?.textContent || '';
  const chips = document.getElementById('surfacePlanChips')?.textContent || '';
  const contract = [...document.querySelectorAll('#writeContract .contractop code')].map((x) => x.textContent || '');
  const evalRows = document.querySelectorAll('#evalList li').length;
  const guardRows = document.querySelectorAll('#guardList li').length;
  return {
    title: document.getElementById('skillTitle')?.textContent || '',
    hasCompose: contract.some((x) => x.includes('compose_email')),
    hasCreateEvent: contract.some((x) => x.includes('create_event')),
    hasDateAnchor: /2026-06-29/.test(rules),
    hasIsoRule: /ISO 8601/.test(rules),
    hasHeldOutChip: /held-out eval/.test(chips),
    evalRows,
    guardRows,
  };
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
  window.__eg.renderHistory(); // renders the inventory list + (via tail call) the dock
});
checks.dockForged = await p.evaluate(() => {
  const tiles = [...document.querySelectorAll('#dockSlots .dock__tile')];
  const st = (k) => tiles.find((t) => t.dataset.key === k)?.dataset.state;
  return {
    inbox: st('inbox-calendar'), music: st('music'),
    owned: tiles.filter((t) => t.dataset.state === 'owned').length,
    equipped: tiles.filter((t) => t.dataset.state === 'equipped').length,
    locked: tiles.filter((t) => t.dataset.state === 'locked').length,
    keys: tiles.filter((t) => t.querySelector('.dock__key')).map((t) => t.querySelector('.dock__key').textContent),
    hasByodSep: !!document.querySelector('#dockSlots .dock__sep'),
    byodTile: tiles.some((t) => t.dataset.key && t.dataset.key.startsWith('byod-')),
  };
});

// 4b) Inventory (left rail): each forged skill renders as a loot item with a
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

// 4c) Adventure stage: Sierra-style scene reflects the equipped service as the
//     current "location", a score box (skills acquired / total), and a narrator.
checks.stage = await p.evaluate(() => {
  const stage = document.getElementById('stage');
  return {
    present: !!stage,
    where: stage?.dataset.where,
    score: document.getElementById('stageScore')?.textContent || '',
    place: document.getElementById('stageSignName')?.textContent || '',
    iconBg: document.getElementById('stageSignIcon')?.style.background || '',
  };
});

// 5) Number-key quick-equip (FPS slots): press "2" → resolves the 2nd owned
//    dock tile (the [2] tile). Completing the hot-swap needs a loaded model/GPU,
//    which headless lacks; we assert keybind→index→run resolution via lastEquipIntent.
const secondId = await p.evaluate(() =>
  document.querySelector('#dockSlots .dock__tile .dock__key')?.parentElement && // ensure keys exist
  [...document.querySelectorAll('#dockSlots .dock__tile')].find((t) => t.querySelector('.dock__key')?.textContent === '2')?.dataset.runid);
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

// 6) Radial quick-swap wheel: opens with one node per owned skill, one highlighted; Esc/cancel closes.
checks.wheel = await p.evaluate(() => {
  window.__eg.openWheel(false);
  const open = !document.getElementById('wheel').hidden;
  const nodes = document.querySelectorAll('#wheelRing .wheel__node').length;
  const onSel = document.querySelectorAll('#wheelRing .wheel__node.on').length;
  const hub = document.getElementById('wheelHub')?.textContent || '';
  window.__eg.closeWheel(true);
  return { open, nodes, onSel, hub, closed: !!document.getElementById('wheel').hidden };
});
// Slack-style quick switcher: Ctrl/⌘-K opens it; Esc closes. (Workflow-software binding.)
await p.keyboard.press('Control+k'); await p.waitForTimeout(80);
checks.wheel.kbdOpen = await p.evaluate(() => !document.getElementById('wheel').hidden);
await p.keyboard.press('Escape'); await p.waitForTimeout(80);
checks.wheel.kbdClosed = await p.evaluate(() => !!document.getElementById('wheel').hidden);

await p.evaluate(() => window.__eg.openTrainer()); await p.waitForTimeout(80);
await p.screenshot({ path: '/tmp/ui_skill_train.png', fullPage: true });
await p.click('#trainerClose'); await p.waitForTimeout(120);
checks.singleView.trainerCloses = await p.evaluate(() => !!document.getElementById('trainer')?.hidden);
await p.screenshot({ path: '/tmp/ui_skill_infer.png', fullPage: true });
await b.close();

const pass =
  checks.registry.n >= 12 && checks.registry.allHaveSpec &&
  checks.registry.corpus >= 500 && checks.registry.hasGithub && checks.registry.dockTiles >= 30 &&
  checks.dockFresh.forge === checks.registry.n && checks.dockFresh.locked >= 5 && checks.dockFresh.owned === 0 &&
  checks.dockFresh.inboxForge === 'forge' && checks.dockFresh.iconBg.length > 0 &&
  checks.picker.btnCount === checks.registry.n && checks.picker.onKey === 'inbox-calendar' &&
  checks.picker.iconStyled && checks.picker.hasMeta &&
  /Inbox & Calendar surface/.test(checks.calendarTrain.title) &&
  checks.calendarTrain.hasCompose && checks.calendarTrain.hasCreateEvent &&
  checks.calendarTrain.hasDateAnchor && checks.calendarTrain.hasIsoRule &&
  checks.calendarTrain.hasHeldOutChip && checks.calendarTrain.evalRows >= 4 &&
  checks.calendarTrain.guardRows >= 4 &&
  checks.musicSelect.hasMusicOp && checks.macroSwapped &&
  checks.verify.ok && checks.verify.oos && checks.verify.badArg && checks.verify.badOp && checks.verify.musicOk &&
  checks.dockForged.inbox === 'equipped' && checks.dockForged.music === 'owned' &&
  checks.dockForged.equipped === 1 && checks.dockForged.owned >= 2 && checks.dockForged.locked >= 5 &&
  checks.dockForged.keys.includes('1') && checks.dockForged.keys.includes('2') &&
  checks.dockForged.hasByodSep && checks.dockForged.byodTile &&
  checks.inventory.count === 3 && checks.inventory.countBadge === '3' &&
  checks.inventory.allHaveFrame && checks.inventory.equippedTag.includes('inbox-calendar') &&
  checks.inventory.rarities.every(Boolean) &&
  checks.stage.present && checks.stage.where === 'in' && checks.stage.score === '2 / 12' &&
  checks.stage.place === 'inbox-calendar' && checks.stage.iconBg.length > 0 &&
  checks.singleView.noTabs && checks.singleView.trainerHiddenAtStart && checks.singleView.hasLearnBtn &&
  checks.singleView.hasLearnCta && checks.singleView.trainerOpens && checks.singleView.trainerCloses &&
  checks.keybind.switched && checks.keybind.ignoredInInput &&
  checks.wheel.open && checks.wheel.nodes === 3 && checks.wheel.onSel === 1 && checks.wheel.closed &&
  checks.wheel.kbdOpen && checks.wheel.kbdClosed &&
  errs.length === 0;

for (const [k, v] of Object.entries(checks)) console.log(k.padEnd(12), JSON.stringify(v));
console.log('pageerrors  ', errs.length ? errs : 'none');
console.log(pass ? 'SKILL_CHECK_PASS' : 'SKILL_CHECK_FAIL');
process.exit(pass ? 0 : 1);
