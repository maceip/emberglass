// contract.ts — the Inbox & Calendar block's declarative invariants (substrate S7).
//
// These are the calendar-family bugs we fixed, now frozen as DATA: predicates checkContract
// evaluates against any emitted macro. Anyone who receives this block can verify a tune with
// nothing but the contract — no app, no GPU, no trust in the trainer.
import type { ContractExtra } from '../types.ts';

const ISO_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/;
const lines = (m: string): string[] => String(m).split('\n');

export const CALENDAR_CONTRACT: ContractExtra = {
  assertions: [
    {
      id: 'iso-times',
      describe: 'start/end/when/after/before literals are ISO 8601 (YYYY-MM-DDTHH:MM)',
      holds: (m) =>
        lines(m).every((ln) =>
          [...ln.matchAll(/(?:start|end|when|after|before)="([^"]+)"/g)].every((x) => ISO_RE.test(x[1])),
        ),
    },
  ],
  forbidden: [
    {
      id: 'zero-duration-event',
      describe: 'create_event must not have start == end',
      violatedBy: (m) =>
        lines(m).some((ln) => {
          const c = ln.match(/create_event\(.*start="([^"]+)".*end="([^"]+)"/);
          return !!c && c[1] === c[2];
        }),
    },
    {
      id: 'unordered-slot-window',
      describe: 'find_slot must have after < before',
      violatedBy: (m) =>
        lines(m).some((ln) => {
          const f = ln.match(/find_slot\(.*after="([^"]+)".*before="([^"]+)"/);
          return !!f && !(f[1] < f[2]);
        }),
    },
  ],
};
