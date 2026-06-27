// index.ts — assembles the Inbox & Calendar block into the DEF that skills.js#buildSkill
// consumes. The output surface is byte-identical to the old inline definition; only the
// internal shape (port / contract / adapter / manifest) changed.
import type { SkillDef } from '../types.ts';
import { META, DOMAIN, SCOPE, CONTEXT, OPS } from './port.ts';
import { CALENDAR_CONTRACT } from './contract.ts';
import { genCalendar } from './adapters/google.ts';

export { MANIFEST } from './manifest.ts';
export { CALENDAR_LESSONS } from './lessons.ts'; // side effect: registers calendar lessons in the FoT store

export const calendarDef: SkillDef = {
  key: META.key,
  label: META.label,
  icon: META.icon,
  domain: DOMAIN,
  scope: SCOPE,
  desc: META.desc,
  suggest: META.suggest,
  ops: OPS,
  context: CONTEXT,
  examplesFn: genCalendar,
  contract: CALENDAR_CONTRACT,
};
