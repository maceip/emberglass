/* Emberglass wireframes — shared UI component builders.
 * Pure functions returning HTML strings, driven entirely by state.js so the
 * three screens stay consistent. Client-only; no framework. */
import { stateLabel, STATE_HUE, iconUrl, logoUrl, skillById } from './state.js';

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// Every write that renders on the three screens (calendar / gmail / notes) gets
// its OWN distinct *I1 tile — no two writes in a row share a glyph, and none
// reuse a skill tile. Locked-skill writes never render here, so they alias an
// existing distinct glyph rather than shipping unused icon files.
const WRITE_ICON = {
  create_event: 'act-event', find_slot: 'act-slot', set_reminder: 'act-reminder', rsvp: 'act-rsvp',
  draft_reply: 'act-reply', label: 'act-label', archive: 'act-archive', extract_meeting_request: 'act-meeting',
  create_note: 'act-note', append_note: 'act-append', add_checklist: 'act-checklist',
  // aliases for never-rendered locked-skill writes (no extra files generated)
  update_deal: 'act-event', log_activity: 'act-meeting', summarize_call: 'act-reply', create_followup: 'act-reminder',
  open_issue: 'act-event', comment: 'act-reply', label_pr: 'act-label', find_email: 'act-slot',
};

const RISK_HUE = { 'read-only': 'idle', 'reversible-write': 'learning', 'sensitive-write': 'review' };

/** Top brand bar + the three-screen game nav. active in {home,skill,board}. */
export function nav(active) {
  const tab = (id, href, label, feel) =>
    `<a class="navtab${active === id ? ' navtab--on' : ''}" href="${href}"><b>${label}</b><span>${feel}</span></a>`;
  return `
  <header class="topbar">
    <span class="brand"><span class="brand__logo">🜂</span> Emberglass</span>
    <nav class="screens">
      ${tab('home', 'home.html', 'Skillbook', 'Home / Inventory')}
      ${tab('skill', 'skill.html', 'Trial Page', 'Skill / Train')}
      ${tab('board', 'job-board.html', 'Quest Board', 'Job Board')}
    </nav>
    <span class="dryrun" title="No account is ever changed in these wireframes">⛬ dry-run only</span>
  </header>`;
}

/** Skill icon tile (processed *I1), with optional pixel @2x and locked variant. */
export function skillIcon(skill, { size = 'md', locked = false } = {}) {
  const key = skill.iconKey;
  const src = locked ? iconUrl(key).replace('.png', '-locked.png') : iconUrl(key);
  // wire the @2x pixel derivative so retina/upscaled renders stay crisp
  const srcset = locked ? '' : ` srcset="${src} 1x, ${src.replace('.png', '@2x.png')} 2x"`;
  const cls = `sicon sicon--${size}` + (locked ? ' sicon--locked' : '');
  const brand = skill.logo ? `<img class="sicon__brand" src="${logoUrl(skill.logo)}" alt="" onerror="this.remove()">` : '';
  return `<span class="${cls}"><img src="${src}"${srcset} alt="" onerror="this.style.opacity=0">${brand}</span>`;
}

/** Earned-state badge: label first, number as proof. */
export function stateBadge(skill, { showScore = true } = {}) {
  const label = stateLabel(skill);
  const hue = STATE_HUE[label] || 'idle';
  const score = showScore && typeof skill.evalScore === 'number' ? `<span class="proof">${skill.evalScore}%</span>` : '';
  return `<span class="state state--${hue}"><span class="led"></span>${label}</span>${score}`;
}

/** Allowed-write contract as a row of capability icons + labels. */
export function writeIcons(writes, { labels = true } = {}) {
  return `<span class="writes">` + writes.map((w) => {
    const k = WRITE_ICON[w] || 'act-event';
    return `<span class="write" title="${esc(w)}"><img src="${iconUrl(k)}" alt="" onerror="this.style.opacity=0">${labels ? `<code>${esc(w)}</code>` : ''}</span>`;
  }).join('') + `</span>`;
}

/** Inventory tile: icon, name, level chip, state. Locked shows the promise. */
export function skillTile(skill, { selected = false, onclick = '' } = {}) {
  const locked = skill.status === 'locked';
  const equipped = skill.status === 'equipped';
  const cls = ['tile', selected ? 'tile--sel' : '', equipped ? 'tile--equipped' : '', locked ? 'tile--locked' : ''].filter(Boolean).join(' ');
  const lv = locked ? `<span class="lv lv--lock">🔒</span>` : `<span class="lv">L${skill.level}</span>`;
  const sub = locked
    ? `<div class="promise"><span class="req">needs ${esc(skill.requiresText || '')}</span><span class="unl">unlocks ${esc(skill.unlocks || '')}</span></div>`
    : `<div class="tilestate">${stateBadge(skill)}</div>`;
  return `<button class="${cls}" ${onclick ? `onclick="${onclick}"` : ''} type="button">
    ${skillIcon(skill, { size: 'md', locked })}
    <span class="tile__body"><b>${esc(skill.displayName)}</b>${sub}</span>
    ${lv}
  </button>`;
}

/** Equipped chain row (1–4 skills). */
export function equippedChain(ids) {
  const chips = ids.map((id) => {
    const s = skillById(id); if (!s) return '';
    return `<span class="chip chip--on">${skillIcon(s, { size: 'sm' })} ${esc(s.displayName)}</span>`;
  });
  return `<div class="chain">${chips.join('<span class="arrow">→</span>')}</div>`;
}

/** Verified plan "seal": steps + contract stamp + dry-run trust line. */
export function planSeal(cast, { title = 'Plan Seal' } = {}) {
  const steps = cast.steps.map((st, i) => {
    const hue = RISK_HUE[st.risk] || 'idle';
    return `<div class="seal__step">
      <span class="num">${i + 1}</span>
      <code class="macro"><b>${esc(st.op)}</b>(${esc(st.args)})</code>
      <span class="risk risk--${hue}">${esc(st.risk)}</span>
    </div>`;
  }).join('');
  const stamp = cast.contractOk
    ? `<span class="stamp stamp--ok">✦ contract passed</span>`
    : `<span class="stamp stamp--warn">✕ contract failed</span>`;
  return `<div class="seal">
    <div class="seal__title">${esc(title)}</div>
    ${steps}
    <div class="seal__foot">${stamp}<span class="dryrun">⛬ no account changed · simulated</span>
      <span class="cap">needs ${esc(cast.capability)}</span></div>
  </div>`;
}

/** Cast box: command input styled as "using the equipped skill". */
export function castBox(command, { verb = 'Cast Request', btn = 'Cast' } = {}) {
  return `<div class="cast">
    <label class="cast__label">${esc(verb)}</label>
    <div class="cast__row"><input value="${esc(command)}" aria-label="${esc(verb)}"><button class="btn btn--hero" type="button">${esc(btn)}</button></div>
  </div>`;
}

/** Pinned quest task card: weakness + action + reward, one verb. */
export function questCard(task) {
  const s = skillById(task.skill);
  return `<div class="quest">
    <div class="quest__head">${s ? skillIcon(s, { size: 'sm' }) : ''}<b>${esc(task.title)}</b></div>
    <ul class="quest__meta">
      <li><span class="k">weakness</span> ${esc(task.weakness)}</li>
      <li><span class="k">reward</span> ${esc(task.reward)}</li>
    </ul>
    <a class="btn btn--hero btn--block" href="skill.html">${esc(task.action)}</a>
  </div>`;
}

/** Training progress meter (uses the processed *P1 console strip as a track). */
export function forgeMeter(passed, total) {
  const pct = Math.round((passed / total) * 100);
  return `<div class="forge">
    <div class="forge__bar"><span style="width:${pct}%"></span></div>
    <div class="forge__lbl">learned ${passed}/${total} trials · ${pct}%</div>
  </div>`;
}
