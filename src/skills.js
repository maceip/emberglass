/*
 * skills.js — the action-space library that powers the "skills" you forge.
 *
 * A skill teaches the model to compile a plain request into a MACRO over a small,
 * typed action space (the operations). This is constrained codegen — the model's
 * strength — not open-domain recall. We SUPPLY the action space (system prompt +
 * LoRA) so the model never invents an API; out-of-scope requests must bounce with
 * OUT_OF_SCOPE. Every `spec` is the machine-readable contract used to BOTH generate
 * the system prompt AND verify the emitted macro afterwards (the "does what we say"
 * gate). Examples are part hand-seeded (`fixed`) and part programmatically expanded
 * from `templates` × `vocab`, giving 500+ ready-to-train pairs across services.
 *
 * Pure ESM, no DOM — so it can be unit-tested in Node (test/_skills_data.mjs).
 */

// Flagship block, carved into a typed substrate-shaped module (port/contract/adapter/manifest).
// buildSkill consumes calendarDef exactly like the old inline DEF — output is byte-identical.
import { calendarDef } from './skills/inbox-calendar/index.ts';

// ── spec → system prompt ─────────────────────────────────────────────────────
export function specSig(spec) {
  return spec.ops.map((o) => `${o.name}(${(o.params || []).join(', ')})${o.ret ? ' -> ' + o.ret : ''}`).join('; ');
}
export function skillSystem(domain, spec, context) {
  return `You are ${domain}. Convert the request into a macro using ONLY these operations:\n` +
    specSig(spec) + '.\n' +
    (context ? context + '\n' : '') +
    `Output ONLY the macro, one call per line, no prose. If the request is outside ${spec.scope}, output exactly: OUT_OF_SCOPE.`;
}

// ── macro verifier (shared by the UI HUD and the Node data test) ─────────────
export function parseMacroCalls(text) {
  const out = [];
  for (const raw of String(text).split('\n')) {
    const line = raw.trim();
    if (!line || line === 'OUT_OF_SCOPE') continue;
    const m = line.match(/^(?:[A-Za-z_]\w*\s*=\s*)?([A-Za-z_]\w*)\s*\((.*)\)\s*;?\s*$/);
    if (!m) continue;
    const keys = [...m[2].matchAll(/(?:^|,)\s*([A-Za-z_]\w*)\s*=/g)].map((k) => k[1]);
    out.push({ op: m[1], keys });
  }
  return out;
}
export function verifyMacro(text, spec) {
  const t = String(text);
  const calls = parseMacroCalls(t);
  const bounced = /(^|\n)\s*OUT_OF_SCOPE\s*($|\n)/.test(t) && calls.length === 0;
  if (bounced) return { status: 'oos', calls: [], issues: [], n: 0 };
  if (!calls.length) return { status: 'empty', calls: [], issues: [], n: 0 };
  const byName = new Map(spec.ops.map((o) => [o.name, o]));
  const issues = [];
  const detail = [];
  for (const c of calls) {
    const op = byName.get(c.op);
    if (!op) { issues.push(`unknown op: ${c.op}`); detail.push({ op: c.op, ok: false }); continue; }
    const allowed = new Set(op.params || []);
    const bad = c.keys.filter((k) => !allowed.has(k));
    if (bad.length) { issues.push(`${c.op}: unexpected arg ${bad.join(', ')}`); detail.push({ op: c.op, ok: false }); }
    else detail.push({ op: c.op, ok: true });
  }
  return { status: issues.length ? 'bad' : 'ok', calls: detail, issues, n: calls.length };
}

// ── declarative port CONTRACT (substrate S7, reimplemented dependency-free) ──
// A contract is DATA: assertions that must hold and forbidden patterns that must never
// match, evaluated against an emitted macro under a spec. Same "verification criteria as
// data" idea as substrate/_kernel/contract.ts, but no zod and no imports so it runs in the
// browser bundle. This is the gate that makes a SHARED skill verifiable by anyone.
export function checkContract(contract, macro, spec) {
  const violations = [];
  for (const a of contract.assertions || []) {
    let ok = false; try { ok = a.holds(macro, spec); } catch { ok = false; }
    if (!ok) violations.push({ kind: 'assertion', id: a.id, detail: a.describe });
  }
  for (const f of contract.forbidden || []) {
    let bad = true; try { bad = f.violatedBy(macro, spec); } catch { bad = true; } // fail closed
    if (bad) violations.push({ kind: 'forbidden', id: f.id, detail: f.describe });
  }
  return { ok: violations.length === 0, violations };
}
// Base assertion every skill shares: the macro is spec-valid (ops + args) or a clean bounce.
// Family-specific invariants (e.g. calendar ISO times) live in each block's contract.ts.
const BASE_ASSERTIONS = [{
  id: 'spec-valid',
  describe: 'every call uses a spec op with only that op\u2019s params, or the macro is a clean OUT_OF_SCOPE bounce',
  holds: (m, spec) => { const r = verifyMacro(m, spec); return r.status === 'ok' || r.status === 'oos'; },
}];
function buildContract(def) {
  const extra = def.contract || {};
  return {
    block: def.key,
    assertions: [...BASE_ASSERTIONS, ...(extra.assertions || [])],
    forbidden: [...(extra.forbidden || [])],
  };
}

// ── deterministic example generator ──────────────────────────────────────────
function hashStr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function fill(tpl, choice) { return tpl.replace(/\{(\w+)\}/g, (_, k) => (k in choice ? choice[k] : '{' + k + '}')); }
function expand(def, perTemplate) {
  const rnd = mulberry32(hashStr(def.key));
  const out = [];
  const seen = new Set();
  for (const t of def.templates || []) {
    const slots = [...new Set([...t.req.matchAll(/\{(\w+)\}/g)].map((m) => m[1]))];
    let made = 0, tries = 0;
    const cap = perTemplate * 8;
    while (made < perTemplate && tries < cap) {
      tries++;
      const choice = {};
      for (const s of slots) { const arr = def.vocab[s] || ['x']; choice[s] = arr[Math.floor(rnd() * arr.length)]; }
      const req = fill(t.req, choice);
      if (seen.has(req)) continue;
      seen.add(req);
      out.push([req, fill(t.macro, choice)]);
      made++;
    }
  }
  return out;
}
function buildSkill(def, perTemplate = 6) {
  const spec = { scope: def.scope, ops: def.ops };
  // A skill can either hand us a bespoke generator (examplesFn) — used by the
  // flagship calendar skill for ISO-correct, balanced, eval-split data — or rely
  // on the generic fixed + templates×vocab expansion.
  let examples, evals = [];
  if (typeof def.examplesFn === 'function') {
    const g = def.examplesFn();
    examples = g.examples;
    evals = g.eval || [];
  } else {
    examples = [
      ...(def.fixed || []),
      ...expand(def, perTemplate),
      ...(def.oos || []).map((r) => [r, 'OUT_OF_SCOPE']),
    ];
  }
  return {
    key: def.key, label: def.label, icon: def.icon, desc: def.desc, domain: def.domain,
    spec, context: def.context || '', system: skillSystem(def.domain, spec, def.context), suggest: def.suggest, examples, eval: evals,
    contract: buildContract(def),
  };
}

// ── shared vocab pools (kept quote/comma-free so macros stay valid) ──────────
const PEOPLE = ['mom', 'Sarah', 'Alex', 'the design team', 'my manager', 'Priya', 'John', 'the landlord', 'accounting', 'Dana'];
const TOPICS = ['the Q3 roadmap', 'the launch', 'the budget', 'onboarding', 'the API redesign', 'the offsite', 'the bug report', 'the contract'];
const WHENS = ['today 17:00', 'tomorrow 09:00', 'Friday 14:00', 'next Monday 10:00', 'Thursday 16:30', 'tonight 19:00'];

// ── service definitions ──────────────────────────────────────────────────────
const DEFS = [
  calendarDef, // flagship block: src/skills/inbox-calendar/{port,contract,adapters/google,manifest}.ts
  {
    key: 'music', label: 'Music', icon: '♪', domain: 'a music player operator', scope: 'music playback',
    desc: 'Turns “play some lo-fi and turn it down” into a macro over a music action space — find/play/queue/volume/playlist — and bounces non-music asks.',
    suggest: 'Play something upbeat for cooking and add it to a new playlist called Dinner.',
    ops: [
      { name: 'find_track', params: ['query'], ret: 'track' },
      { name: 'play_track', params: ['track'] },
      { name: 'queue_track', params: ['track'] },
      { name: 'pause', params: [] },
      { name: 'skip', params: [] },
      { name: 'previous', params: [] },
      { name: 'set_volume', params: ['level'] },
      { name: 'create_playlist', params: ['name'] },
      { name: 'add_to_playlist', params: ['playlist', 'track'] },
      { name: 'shuffle', params: ['on'] },
      { name: 'repeat', params: ['mode'] },
    ],
    fixed: [
      ['skip this song', 'skip()'],
      ['pause the music', 'pause()'],
      ['go back to the previous song', 'previous()'],
    ],
    templates: [
      { req: 'play some {genre}', macro: 't = find_track(query="{genre}")\nplay_track(track=t)' },
      { req: 'queue up {artist} after this', macro: 't = find_track(query="{artist}")\nqueue_track(track=t)' },
      { req: 'set the volume to {vol}', macro: 'set_volume(level={vol})' },
      { req: 'make a playlist called {name}', macro: 'create_playlist(name="{name}")' },
      { req: 'add {artist} to my {name} playlist', macro: 't = find_track(query="{artist}")\nadd_to_playlist(playlist="{name}", track=t)' },
      { req: 'shuffle my {name} playlist', macro: 'shuffle(on=true)\nt = find_track(query="playlist:{name}")\nplay_track(track=t)' },
      { req: 'put on {artist} and turn it up', macro: 't = find_track(query="{artist}")\nplay_track(track=t)\nset_volume(level=80)' },
      { req: 'repeat this {mode}', macro: 'repeat(mode="{mode}")' },
    ],
    vocab: {
      genre: ['lo-fi beats', 'deep house', 'classic jazz', 'pop hits', 'ambient', 'classical', '90s hip hop', 'indie rock'],
      artist: ['Taylor Swift', 'The Beatles', 'Daft Punk', 'Miles Davis', 'Radiohead', 'Bad Bunny', 'Fleetwood Mac'],
      name: ['Focus', 'Workout', 'Dinner', 'Chill', 'Road Trip', 'Sleep'],
      vol: ['10', '25', '40', '60', '75', '90'], mode: ['one', 'all'],
    },
    oos: ['email my boss', 'what is the weather today?', 'open an issue on the repo'],
  },
  {
    key: 'github', label: 'GitHub', icon: '🐙', domain: 'a GitHub operator',
    scope: 'GitHub repositories, issues, and pull requests',
    desc: 'Compiles dev requests into a macro over issues, pull requests, and repos; bounces anything that isn’t GitHub.',
    suggest: 'Open an issue on the api repo titled "fix login redirect", then assign it to Dana.',
    ops: [
      { name: 'find_issue', params: ['query'], ret: 'issue' },
      { name: 'create_issue', params: ['repo', 'title', 'body'] },
      { name: 'comment_issue', params: ['issue', 'body'] },
      { name: 'close_issue', params: ['issue'] },
      { name: 'assign_issue', params: ['issue', 'assignee'] },
      { name: 'label_issue', params: ['issue', 'label'] },
      { name: 'find_pr', params: ['query'], ret: 'pr' },
      { name: 'open_pr', params: ['repo', 'title', 'branch'] },
      { name: 'review_pr', params: ['pr', 'verdict'] },
      { name: 'merge_pr', params: ['pr'] },
      { name: 'create_repo', params: ['name', 'visibility'] },
      { name: 'star_repo', params: ['repo'] },
    ],
    fixed: [
      ['open an issue on the api repo titled fix login redirect and assign it to Dana',
        'i = create_issue(repo="api", title="fix login redirect", body="The login flow redirects to the wrong page.")\nassign_issue(issue=i, assignee="Dana")'],
    ],
    templates: [
      { req: 'open an issue on {repo} titled {title}', macro: 'create_issue(repo="{repo}", title="{title}", body="{title}.")' },
      { req: 'close the {topic} issue', macro: 'i = find_issue(query="{topic}")\nclose_issue(issue=i)' },
      { req: 'comment {comment} on the {topic} issue', macro: 'i = find_issue(query="{topic}")\ncomment_issue(issue=i, body="{comment}")' },
      { req: 'assign the {topic} issue to {user}', macro: 'i = find_issue(query="{topic}")\nassign_issue(issue=i, assignee="{user}")' },
      { req: 'label the {topic} issue as {label}', macro: 'i = find_issue(query="{topic}")\nlabel_issue(issue=i, label="{label}")' },
      { req: 'open a pull request on {repo} from {branch} titled {title}', macro: 'open_pr(repo="{repo}", title="{title}", branch="{branch}")' },
      { req: 'approve the {topic} pull request', macro: 'p = find_pr(query="{topic}")\nreview_pr(pr=p, verdict="approve")' },
      { req: 'merge the {topic} PR', macro: 'p = find_pr(query="{topic}")\nmerge_pr(pr=p)' },
      { req: 'create a private repo called {repo}', macro: 'create_repo(name="{repo}", visibility="private")' },
      { req: 'star the {repo} repo', macro: 'star_repo(repo="{repo}")' },
    ],
    vocab: {
      repo: ['api', 'frontend', 'docs', 'infra', 'mobile-app', 'design-system'],
      title: ['fix login redirect', 'add dark mode', 'update README', 'flaky test fix', 'bump dependencies', 'improve error logs'],
      topic: ['login', 'dark mode', 'flaky test', 'memory leak', 'rate limiting', 'docs typo'],
      comment: ['looks good to me', 'can you add a test?', 'I will pick this up', 'reproduced on main', 'duplicate of #42'],
      user: ['Dana', 'Alex', 'Priya', 'the on-call', 'Sam'],
      label: ['bug', 'enhancement', 'good first issue', 'p1', 'docs', 'wontfix'],
      branch: ['feature/auth', 'fix/cache', 'chore/deps', 'feat/ui', 'hotfix/crash'],
    },
    oos: ['play some music', 'email my mom', 'what is 2 + 2?'],
  },
  {
    key: 'slack', label: 'Slack', icon: '💬', domain: 'a Slack operator', scope: 'Slack messaging',
    desc: 'Compiles team-chat requests into a macro over channels, DMs, threads, and reminders; bounces non-Slack asks.',
    suggest: 'Post the release notes in #launch and DM Dana to review them.',
    ops: [
      { name: 'find_message', params: ['query'], ret: 'message' },
      { name: 'send_message', params: ['channel', 'text'] },
      { name: 'dm', params: ['user', 'text'] },
      { name: 'reply_thread', params: ['message', 'text'] },
      { name: 'react', params: ['message', 'emoji'] },
      { name: 'set_status', params: ['text', 'emoji'] },
      { name: 'create_channel', params: ['name'] },
      { name: 'invite', params: ['user', 'channel'] },
      { name: 'remind', params: ['text', 'when'] },
      { name: 'pin', params: ['message'] },
    ],
    fixed: [
      ['post the release notes in #launch and dm Dana to review them',
        'send_message(channel="launch", text="Release notes are up — please review.")\ndm(user="Dana", text="Can you review the release notes I posted in #launch?")'],
    ],
    templates: [
      { req: 'post {text} in #{channel}', macro: 'send_message(channel="{channel}", text="{text}")' },
      { req: 'dm {user} {text}', macro: 'dm(user="{user}", text="{text}")' },
      { req: 'reply {text} to the {topic} thread', macro: 'm = find_message(query="{topic}")\nreply_thread(message=m, text="{text}")' },
      { req: 'react {emoji} to the {topic} message', macro: 'm = find_message(query="{topic}")\nreact(message=m, emoji="{emoji}")' },
      { req: 'set my status to {text}', macro: 'set_status(text="{text}", emoji="{emoji}")' },
      { req: 'create a channel called {channel}', macro: 'create_channel(name="{channel}")' },
      { req: 'invite {user} to #{channel}', macro: 'invite(user="{user}", channel="{channel}")' },
      { req: 'remind the team to {task} {when}', macro: 'remind(text="{task}", when="{when}")' },
      { req: 'pin the {topic} message', macro: 'm = find_message(query="{topic}")\npin(message=m)' },
    ],
    vocab: {
      channel: ['launch', 'general', 'engineering', 'design', 'random', 'incidents'],
      user: ['Dana', 'Alex', 'Priya', 'Sam', 'the team lead'],
      text: ['standup in 5', 'PR is ready for review', 'deploy is green', 'lunch at noon?', 'great work today'],
      topic: ['deploy', 'incident', 'roadmap', 'lunch', 'release'],
      emoji: [':eyes:', ':white_check_mark:', ':tada:', ':fire:', ':+1:'],
      task: ['submit timesheets', 'join the retro', 'review the doc', 'update the board'],
      when: WHENS,
    },
    oos: ['play a song', 'order groceries', 'what time is it in Tokyo?'],
  },
  {
    key: 'notion', label: 'Notion', icon: '📝', domain: 'a Notion operator', scope: 'Notion pages, notes, and tasks',
    desc: 'Compiles note-taking requests into a macro over pages, blocks, tasks, and databases; bounces anything else.',
    suggest: 'Create a page titled "Trip plan" and add a task to book flights due Friday.',
    ops: [
      { name: 'find_page', params: ['query'], ret: 'page' },
      { name: 'create_page', params: ['title', 'body'] },
      { name: 'append_block', params: ['page', 'text'] },
      { name: 'create_task', params: ['title', 'due'] },
      { name: 'complete_task', params: ['task'] },
      { name: 'find_task', params: ['query'], ret: 'task' },
      { name: 'add_to_database', params: ['database', 'name'] },
      { name: 'set_property', params: ['page', 'key', 'value'] },
      { name: 'create_database', params: ['name'] },
    ],
    fixed: [
      ['create a page titled Trip plan and add a task to book flights due Friday',
        'create_page(title="Trip plan", body="Planning notes.")\ncreate_task(title="Book flights", due="Friday")'],
    ],
    templates: [
      { req: 'create a page titled {title}', macro: 'create_page(title="{title}", body="{title} — notes.")' },
      { req: 'add a note {text} to the {topic} page', macro: 'p = find_page(query="{topic}")\nappend_block(page=p, text="{text}")' },
      { req: 'add a task to {task} due {when}', macro: 'create_task(title="{task}", due="{when}")' },
      { req: 'mark the {task} task done', macro: 't = find_task(query="{task}")\ncomplete_task(task=t)' },
      { req: 'add {name} to my {database} database', macro: 'add_to_database(database="{database}", name="{name}")' },
      { req: 'set the status of the {topic} page to {value}', macro: 'p = find_page(query="{topic}")\nset_property(page=p, key="status", value="{value}")' },
      { req: 'create a database called {database}', macro: 'create_database(name="{database}")' },
    ],
    vocab: {
      title: ['Trip plan', 'Q3 goals', 'Reading list', 'Meeting notes', 'Project brief', 'Recipes'],
      text: ['remember to confirm the budget', 'add the agenda', 'link the spec', 'note the blockers'],
      topic: ['trip', 'goals', 'project', 'meeting', 'reading'],
      task: ['book flights', 'draft the brief', 'email the vendor', 'review the PR', 'pay the invoice'],
      when: ['today', 'tomorrow', 'Friday', 'next week', 'end of month'],
      name: ['Acme Co', 'Q3 launch', 'Vendor X', 'Idea: dark mode'],
      database: ['Projects', 'CRM', 'Tasks', 'Reading', 'Inventory'],
      value: ['in progress', 'done', 'blocked', 'todo', 'review'],
    },
    oos: ['play music', 'navigate home', 'send a tweet'],
  },
  {
    key: 'x', label: 'X', icon: '𝕏', domain: 'an X (Twitter) operator', scope: 'posting and engagement on X',
    desc: 'Compiles social requests into a macro over posts, replies, reposts, follows, and DMs; bounces anything off-platform.',
    suggest: 'Post "shipping something fun today 🚀" and schedule a follow-up for 5pm.',
    ops: [
      { name: 'find_post', params: ['query'], ret: 'post' },
      { name: 'post', params: ['text'] },
      { name: 'reply', params: ['post', 'text'] },
      { name: 'repost', params: ['post'] },
      { name: 'like', params: ['post'] },
      { name: 'follow', params: ['user'] },
      { name: 'dm', params: ['user', 'text'] },
      { name: 'schedule_post', params: ['text', 'when'] },
      { name: 'bookmark', params: ['post'] },
    ],
    fixed: [
      ['post shipping something fun today and schedule a follow up for 5pm',
        'post(text="shipping something fun today 🚀")\nschedule_post(text="more details soon — stay tuned", when="today 17:00")'],
    ],
    templates: [
      { req: 'post {text}', macro: 'post(text="{text}")' },
      { req: 'reply {text} to the {topic} post', macro: 'p = find_post(query="{topic}")\nreply(post=p, text="{text}")' },
      { req: 'repost the {topic} tweet', macro: 'p = find_post(query="{topic}")\nrepost(post=p)' },
      { req: 'like the {topic} post', macro: 'p = find_post(query="{topic}")\nlike(post=p)' },
      { req: 'follow {user}', macro: 'follow(user="{user}")' },
      { req: 'dm {user} {text}', macro: 'dm(user="{user}", text="{text}")' },
      { req: 'schedule a post {when} saying {text}', macro: 'schedule_post(text="{text}", when="{when}")' },
      { req: 'bookmark the {topic} thread', macro: 'p = find_post(query="{topic}")\nbookmark(post=p)' },
    ],
    vocab: {
      text: ['gm', 'big news coming', 'loved this talk', 'hot take: tabs > spaces', 'thanks for 10k followers'],
      topic: ['the launch', 'the keynote', 'the meme', 'the thread on AI', 'the announcement'],
      user: ['@levelsio', '@naval', '@swyx', '@dhh', '@karpathy'],
      when: WHENS,
    },
    oos: ['archive my inbox', 'play a playlist', 'open a GitHub issue'],
  },
  {
    key: 'instagram', label: 'Instagram', icon: '📷', domain: 'an Instagram operator', scope: 'Instagram posts, stories, and DMs',
    desc: 'Compiles requests into a macro over photo posts, stories, comments, and DMs; bounces anything off-platform.',
    suggest: 'Post a photo with caption "sunset run 🌅" and share it to my story.',
    ops: [
      { name: 'find_post', params: ['query'], ret: 'post' },
      { name: 'post_photo', params: ['caption', 'media'] },
      { name: 'post_story', params: ['media'] },
      { name: 'reply_dm', params: ['user', 'text'] },
      { name: 'like_post', params: ['post'] },
      { name: 'comment', params: ['post', 'text'] },
      { name: 'follow', params: ['user'] },
      { name: 'save_post', params: ['post'] },
    ],
    fixed: [
      ['post a photo with caption sunset run and share it to my story',
        'post_photo(caption="sunset run 🌅", media="latest")\npost_story(media="latest")'],
    ],
    templates: [
      { req: 'post a photo with caption {caption}', macro: 'post_photo(caption="{caption}", media="latest")' },
      { req: 'share {media} to my story', macro: 'post_story(media="{media}")' },
      { req: 'comment {text} on the {topic} post', macro: 'p = find_post(query="{topic}")\ncomment(post=p, text="{text}")' },
      { req: 'like the {topic} post', macro: 'p = find_post(query="{topic}")\nlike_post(post=p)' },
      { req: 'reply {text} to {user} in DMs', macro: 'reply_dm(user="{user}", text="{text}")' },
      { req: 'follow {user}', macro: 'follow(user="{user}")' },
      { req: 'save the {topic} post', macro: 'p = find_post(query="{topic}")\nsave_post(post=p)' },
    ],
    vocab: {
      caption: ['sunset run 🌅', 'weekend vibes', 'new kicks 👟', 'homemade pasta 🍝', 'trail day'],
      media: ['latest', 'the beach photo', 'the reel', 'the carousel'],
      text: ['love this!', 'where is this?', 'so good 🔥', 'congrats!', 'need the recipe'],
      topic: ['the travel', 'the food', 'the fit check', 'the puppy', 'the launch'],
      user: ['@natgeo', '@nike', '@a_friend', '@the_chef'],
    },
    oos: ['merge the pull request', 'set a reminder', 'navigate to work'],
  },
  {
    key: 'youtube', label: 'YouTube', icon: '▶', domain: 'a YouTube operator', scope: 'YouTube playback and library',
    desc: 'Compiles requests into a macro over search, playback, playlists, and subscriptions; bounces anything else.',
    suggest: 'Play a 10-minute beginner yoga video and add it to my Morning playlist.',
    ops: [
      { name: 'find_video', params: ['query'], ret: 'video' },
      { name: 'play_video', params: ['video'] },
      { name: 'queue_video', params: ['video'] },
      { name: 'subscribe', params: ['channel'] },
      { name: 'like_video', params: ['video'] },
      { name: 'add_to_playlist', params: ['playlist', 'video'] },
      { name: 'create_playlist', params: ['name'] },
      { name: 'comment', params: ['video', 'text'] },
    ],
    fixed: [
      ['play a beginner yoga video and add it to my Morning playlist',
        'v = find_video(query="beginner yoga 10 minutes")\nplay_video(video=v)\nadd_to_playlist(playlist="Morning", video=v)'],
    ],
    templates: [
      { req: 'play a video about {query}', macro: 'v = find_video(query="{query}")\nplay_video(video=v)' },
      { req: 'queue a video about {query}', macro: 'v = find_video(query="{query}")\nqueue_video(video=v)' },
      { req: 'subscribe to {channel}', macro: 'subscribe(channel="{channel}")' },
      { req: 'like the {query} video', macro: 'v = find_video(query="{query}")\nlike_video(video=v)' },
      { req: 'add a {query} video to my {name} playlist', macro: 'v = find_video(query="{query}")\nadd_to_playlist(playlist="{name}", video=v)' },
      { req: 'make a playlist called {name}', macro: 'create_playlist(name="{name}")' },
      { req: 'comment {text} on the {query} video', macro: 'v = find_video(query="{query}")\ncomment(video=v, text="{text}")' },
    ],
    vocab: {
      query: ['lo-fi study mix', 'rust tutorial', 'marathon training', 'pasta recipe', 'guitar lesson', 'space documentary'],
      channel: ['Veritasium', 'Fireship', 'MKBHD', 'Kurzgesagt', 'NileRed'],
      name: ['Morning', 'Watch Later', 'Cooking', 'Workouts', 'Learning'],
      text: ['great explanation!', 'first', 'this helped a lot', 'please do a part 2'],
    },
    oos: ['email the team', 'open a PR', 'set my Slack status'],
  },
  {
    key: 'maps', label: 'Maps', icon: '📍', domain: 'a Maps operator', scope: 'navigation and places',
    desc: 'Compiles requests into a macro over places, directions, and navigation; bounces anything off-map.',
    suggest: 'Find the nearest coffee shop and start navigation, then share my ETA with Alex.',
    ops: [
      { name: 'search_place', params: ['query'], ret: 'place' },
      { name: 'find_nearby', params: ['category'], ret: 'place' },
      { name: 'directions', params: ['to', 'mode'] },
      { name: 'start_navigation', params: ['place'] },
      { name: 'save_place', params: ['place', 'list'] },
      { name: 'share_eta', params: ['place', 'contact'] },
    ],
    fixed: [
      ['find the nearest coffee shop and start navigation then share my eta with Alex',
        'p = find_nearby(category="coffee shop")\nstart_navigation(place=p)\nshare_eta(place=p, contact="Alex")'],
    ],
    templates: [
      { req: 'navigate to {place}', macro: 'p = search_place(query="{place}")\nstart_navigation(place=p)' },
      { req: 'directions to {place} by {mode}', macro: 'directions(to="{place}", mode="{mode}")' },
      { req: 'find a {category} near me', macro: 'find_nearby(category="{category}")' },
      { req: 'find the nearest {category} and navigate there', macro: 'p = find_nearby(category="{category}")\nstart_navigation(place=p)' },
      { req: 'save {place} to my {list} list', macro: 'p = search_place(query="{place}")\nsave_place(place=p, list="{list}")' },
      { req: 'share my ETA to {place} with {contact}', macro: 'p = search_place(query="{place}")\nshare_eta(place=p, contact="{contact}")' },
    ],
    vocab: {
      place: ['the airport', 'downtown', 'the office', 'Central Park', 'the train station', 'the stadium'],
      mode: ['driving', 'walking', 'transit', 'cycling'],
      category: ['coffee shop', 'gas station', 'pharmacy', 'grocery store', 'ATM', 'parking'],
      list: ['Favorites', 'Want to go', 'Trip', 'Restaurants'],
      contact: ['Alex', 'mom', 'Dana', 'the group'],
    },
    oos: ['post a tweet', 'play a song', 'create a GitHub repo'],
  },
  {
    key: 'amazon', label: 'Shopping', icon: '🛒', domain: 'a shopping operator', scope: 'shopping cart and orders',
    desc: 'Compiles requests into a macro over product search, cart, orders, and lists; bounces anything that isn’t shopping.',
    suggest: 'Add two packs of AA batteries to my cart and track my last order.',
    ops: [
      { name: 'search_product', params: ['query'], ret: 'product' },
      { name: 'add_to_cart', params: ['product', 'qty'] },
      { name: 'buy_now', params: ['product'] },
      { name: 'find_order', params: ['query'], ret: 'order' },
      { name: 'track_order', params: ['order'], ret: 'status' },
      { name: 'reorder', params: ['query'] },
      { name: 'add_to_list', params: ['product', 'list'] },
    ],
    fixed: [
      ['add two packs of AA batteries to my cart and track my last order',
        'p = search_product(query="AA batteries 2 pack")\nadd_to_cart(product=p, qty=2)\no = find_order(query="last order")\ntrack_order(order=o)'],
    ],
    templates: [
      { req: 'add {qty} {product} to my cart', macro: 'p = search_product(query="{product}")\nadd_to_cart(product=p, qty={qty})' },
      { req: 'buy {product} now', macro: 'p = search_product(query="{product}")\nbuy_now(product=p)' },
      { req: 'reorder {product}', macro: 'reorder(query="{product}")' },
      { req: 'track my {product} order', macro: 'o = find_order(query="{product}")\ntrack_order(order=o)' },
      { req: 'add {product} to my {list} list', macro: 'p = search_product(query="{product}")\nadd_to_list(product=p, list="{list}")' },
      { req: 'search for {product}', macro: 'search_product(query="{product}")' },
    ],
    vocab: {
      product: ['AA batteries', 'USB-C cable', 'olive oil', 'running shoes', 'paper towels', 'a coffee grinder', 'phone case'],
      qty: ['1', '2', '3', '4'],
      list: ['Wishlist', 'Subscribe & Save', 'Home', 'Gifts'],
    },
    oos: ['send an email', 'play a video', 'navigate to the office'],
  },
  {
    key: 'reddit', label: 'Reddit', icon: '👽', domain: 'a Reddit operator', scope: 'Reddit posts and comments',
    desc: 'Compiles requests into a macro over submissions, comments, votes, and subscriptions; bounces anything off-platform.',
    suggest: 'Post "What mechanical keyboard should I buy?" to r/keyboards and subscribe.',
    ops: [
      { name: 'find_post', params: ['query'], ret: 'post' },
      { name: 'submit_post', params: ['subreddit', 'title', 'body'] },
      { name: 'comment', params: ['post', 'text'] },
      { name: 'upvote', params: ['post'] },
      { name: 'reply_comment', params: ['comment', 'text'] },
      { name: 'subscribe', params: ['subreddit'] },
      { name: 'save_post', params: ['post'] },
    ],
    fixed: [
      ['post what mechanical keyboard should I buy to r/keyboards and subscribe',
        'submit_post(subreddit="keyboards", title="What mechanical keyboard should I buy?", body="Budget is flexible — looking for recommendations.")\nsubscribe(subreddit="keyboards")'],
    ],
    templates: [
      { req: 'post {title} to r/{subreddit}', macro: 'submit_post(subreddit="{subreddit}", title="{title}", body="{title}")' },
      { req: 'comment {text} on the {topic} post', macro: 'p = find_post(query="{topic}")\ncomment(post=p, text="{text}")' },
      { req: 'upvote the {topic} post', macro: 'p = find_post(query="{topic}")\nupvote(post=p)' },
      { req: 'subscribe to r/{subreddit}', macro: 'subscribe(subreddit="{subreddit}")' },
      { req: 'save the {topic} post', macro: 'p = find_post(query="{topic}")\nsave_post(post=p)' },
    ],
    vocab: {
      subreddit: ['keyboards', 'programming', 'AskReddit', 'buildapc', 'cooking', 'fitness'],
      title: ['What keyboard should I buy?', 'Best beginner setup?', 'How do I start running?', 'Favorite pasta recipe?'],
      text: ['this is the way', 'underrated take', 'source?', 'thanks for sharing', 'happy cake day'],
      topic: ['the keyboard', 'the build', 'the recipe', 'the AMA', 'the discussion'],
    },
    oos: ['email my mom', 'play a song', 'navigate home'],
  },
  {
    key: 'linkedin', label: 'LinkedIn', icon: '💼', domain: 'a LinkedIn operator', scope: 'LinkedIn networking and posts',
    desc: 'Compiles requests into a macro over posts, connections, messages, and endorsements; bounces anything off-platform.',
    suggest: 'Connect with Priya with a note, then endorse her for product management.',
    ops: [
      { name: 'find_person', params: ['query'], ret: 'person' },
      { name: 'post_update', params: ['text'] },
      { name: 'connect', params: ['user', 'note'] },
      { name: 'message', params: ['user', 'text'] },
      { name: 'endorse', params: ['person', 'skill'] },
      { name: 'find_post', params: ['query'], ret: 'post' },
      { name: 'comment', params: ['post', 'text'] },
    ],
    fixed: [
      ['connect with Priya with a note then endorse her for product management',
        'connect(user="Priya", note="Great working with you — let us stay in touch!")\np = find_person(query="Priya")\nendorse(person=p, skill="product management")'],
    ],
    templates: [
      { req: 'post an update saying {text}', macro: 'post_update(text="{text}")' },
      { req: 'connect with {user} and add a note {note}', macro: 'connect(user="{user}", note="{note}")' },
      { req: 'message {user} {text}', macro: 'message(user="{user}", text="{text}")' },
      { req: 'endorse {user} for {skill}', macro: 'p = find_person(query="{user}")\nendorse(person=p, skill="{skill}")' },
      { req: 'comment {text} on the {topic} post', macro: 'p = find_post(query="{topic}")\ncomment(post=p, text="{text}")' },
    ],
    vocab: {
      text: ['excited to share I started a new role', 'we are hiring engineers', 'grateful for a great quarter', 'thoughts on remote work'],
      user: ['Priya', 'Alex', 'a recruiter', 'Dana', 'my former manager'],
      note: ['Great working with you!', 'Loved your talk', 'Let us connect', 'Fellow alum here'],
      skill: ['product management', 'leadership', 'TypeScript', 'design', 'data science'],
      topic: ['the hiring', 'the milestone', 'the article', 'the announcement'],
    },
    oos: ['play music', 'open a github issue', 'navigate to the airport'],
  },
];

export const SKILLS = DEFS.map((d) => buildSkill(d, 6));

// ── popular apps & websites — end of June 2026 (the dock catalog) ────────────
// Ordered by rough usage/relevance. Entries with `skill` are forgeable today;
// the rest are the vision — "any app you're logged into" — shown as locked tiles.
// Entries keep glyph/color fallback metadata for instant offline rendering. When
// `logo` is present, `icon_pipeline.js` upgrades the tile with the vendored SVG.
// Flagship skills may still carry a hand-authored color `svg` mark.
export const CALENDAR_SVG =
  '<svg viewBox="0 0 24 24" width="100%" height="100%" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">' +
  '<rect x="2.5" y="4" width="19" height="17" rx="3.4" fill="#ffffff"/>' +
  '<path d="M2.5 7.4a3.4 3.4 0 0 1 3.4-3.4h12.2a3.4 3.4 0 0 1 3.4 3.4v2.1H2.5z" fill="#ea4d3d"/>' +
  '<rect x="6" y="2.1" width="2.5" height="4.4" rx="1.25" fill="#b23528"/>' +
  '<rect x="15.5" y="2.1" width="2.5" height="4.4" rx="1.25" fill="#b23528"/>' +
  '<g fill="#cfd4dc">' +
  '<rect x="5" y="11.6" width="3" height="2.7" rx=".7"/><rect x="10.5" y="11.6" width="3" height="2.7" rx=".7"/>' +
  '<rect x="16" y="11.6" width="3" height="2.7" rx=".7"/><rect x="5" y="15.7" width="3" height="2.7" rx=".7"/>' +
  '<rect x="16" y="15.7" width="3" height="2.7" rx=".7"/></g>' +
  '<rect x="10.5" y="15.7" width="3" height="2.7" rx=".7" fill="#2f72c4"/>' +
  '<rect x="2.5" y="4" width="19" height="17" rx="3.4" fill="none" stroke="#0000001f"/></svg>';
export const POPULAR_2026 = [
  { key: 'inbox-calendar', name: 'Inbox & Calendar', skill: 'inbox-calendar', cat: 'productivity', logo: 'google-calendar', bg: 'linear-gradient(#fdfaf2,#efe7d4)', svg: CALENDAR_SVG, glyph: '✉', fs: 22 },
  { key: 'music', name: 'Music', skill: 'music', cat: 'media', logo: 'spotify', bg: '#1db954', glyph: '♪', fs: 24 },
  { key: 'github', name: 'GitHub', skill: 'github', cat: 'developer', logo: 'github', logoBg: '#f7f3e7', bg: '#181717', glyph: 'GH', fs: 15 },
  { key: 'youtube', name: 'YouTube', skill: 'youtube', cat: 'media', logo: 'youtube', bg: '#FF0000', glyph: '▶', fs: 18 },
  { key: 'instagram', name: 'Instagram', skill: 'instagram', cat: 'social', logo: 'instagram', bg: 'linear-gradient(135deg,#feda75,#d62976 48%,#4f5bd5)', glyph: '📷', fs: 20 },
  { key: 'x', name: 'X', skill: 'x', cat: 'social', logo: 'twitter', bg: '#000000', glyph: '𝕏', fs: 23 },
  { key: 'slack', name: 'Slack', skill: 'slack', cat: 'work', logo: 'slack', bg: '#4A154B', glyph: 'S', fs: 24 },
  { key: 'notion', name: 'Notion', skill: 'notion', cat: 'productivity', logo: 'notion', logoBg: '#f7f3e7', bg: '#0f0f0f', glyph: 'N', fs: 24 },
  { key: 'maps', name: 'Maps', skill: 'maps', cat: 'navigation', logo: 'google-maps', bg: '#34A853', glyph: '📍', fs: 20 },
  { key: 'amazon', name: 'Amazon', skill: 'amazon', cat: 'shopping', bg: '#FF9900', fg: '#232F3E', glyph: 'a', fs: 27 },
  { key: 'reddit', name: 'Reddit', skill: 'reddit', cat: 'social', logo: 'reddit', bg: '#FF4500', glyph: '👽', fs: 20 },
  { key: 'linkedin', name: 'LinkedIn', skill: 'linkedin', cat: 'work', logo: 'linkedin', bg: '#0A66C2', glyph: 'in', fs: 17 },
  // ── the broader armory (coming soon) ──
  { key: 'google', name: 'Google', cat: 'productivity', logo: 'google', bg: '#4285F4', glyph: 'G', fs: 25 },
  { key: 'whatsapp', name: 'WhatsApp', cat: 'social', logo: 'whatsapp', bg: '#25D366', glyph: '✆', fs: 22 },
  { key: 'tiktok', name: 'TikTok', cat: 'social', logo: 'tiktok', bg: '#010101', glyph: '♫', fs: 22 },
  { key: 'facebook', name: 'Facebook', cat: 'social', logo: 'facebook', bg: '#1877F2', glyph: 'f', fs: 27 },
  { key: 'snapchat', name: 'Snapchat', cat: 'social', bg: '#FFFC00', fg: '#111', glyph: '👻', fs: 22 },
  { key: 'messenger', name: 'Messenger', cat: 'social', logo: 'messenger', bg: '#0084FF', glyph: '✦', fs: 22 },
  { key: 'discord', name: 'Discord', cat: 'social', logo: 'discord', bg: '#5865F2', glyph: 'D', fs: 24 },
  { key: 'telegram', name: 'Telegram', cat: 'social', logo: 'telegram', bg: '#229ED9', glyph: '✈', fs: 20 },
  { key: 'netflix', name: 'Netflix', cat: 'media', logo: 'netflix', bg: '#E50914', glyph: 'NF', fs: 15 },
  { key: 'twitch', name: 'Twitch', cat: 'media', logo: 'twitch', bg: '#9146FF', glyph: 'tw', fs: 16 },
  { key: 'spotify', name: 'Spotify', cat: 'media', logo: 'spotify', bg: '#1DB954', glyph: '◉', fs: 20 },
  { key: 'pinterest', name: 'Pinterest', cat: 'social', logo: 'pinterest', bg: '#E60023', glyph: 'P', fs: 24 },
  { key: 'threads', name: 'Threads', cat: 'social', logo: 'threads', logoBg: '#f7f3e7', bg: '#000000', glyph: '@', fs: 24 },
  { key: 'uber', name: 'Uber', cat: 'travel', bg: '#000000', glyph: 'U', fs: 24 },
  { key: 'doordash', name: 'DoorDash', cat: 'food', bg: '#FF3008', glyph: 'DD', fs: 14 },
  { key: 'airbnb', name: 'Airbnb', cat: 'travel', logo: 'airbnb', bg: '#FF5A5F', glyph: 'A', fs: 24 },
  { key: 'paypal', name: 'PayPal', cat: 'finance', logo: 'paypal', bg: '#003087', glyph: 'P', fs: 23 },
  { key: 'venmo', name: 'Venmo', cat: 'finance', bg: '#3D95CE', glyph: 'V', fs: 24 },
  { key: 'chatgpt', name: 'ChatGPT', cat: 'ai', logo: 'openai', bg: '#10A37F', glyph: '✸', fs: 20 },
  { key: 'gemini', name: 'Gemini', cat: 'ai', logo: 'google-gemini', bg: '#1C69FF', glyph: '✦', fs: 20 },
  { key: 'perplexity', name: 'Perplexity', cat: 'ai', logo: 'perplexity', bg: '#1FB8CD', glyph: '✺', fs: 20 },
  { key: 'cursor', name: 'Cursor', cat: 'developer', bg: '#0b0b0b', glyph: '▮', fs: 18 },
];
