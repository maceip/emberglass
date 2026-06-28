// action/executor.ts — the executor seam. ONE executor today: dry-run.
//
// HARD BOUNDARY: this file performs no I/O. No fetch, no XMLHttpRequest, no DOM, no provider
// SDK, no chrome.* — nothing that leaves the process. Every receipt is 'simulated'. A real
// provider/DOM executor is implementation_required — separate, approval-gated milestone
// (see docs/app-action-layer-design.md); adding one must trip the ratchet invariant
// `executors_are_dry_run` and force that review.
import type { ActionPlan, Executor, Receipt } from '../types.ts';
import { record } from './receipt.ts';

export const DryRunExecutor: Executor = {
  id: 'dry-run',
  // fail closed: never "execute" (even simulated) a plan whose contract did not pass
  canExecute: (plan) => plan.contractOk === true,
  execute: (plan) => {
    if (!DryRunExecutor.canExecute(plan)) {
      throw new Error('dry-run refused: contract not satisfied');
    }
    const at = '1970-01-01T00:00:00Z'; // fixed → deterministic receipts in tests
    const receipts: Receipt[] = plan.steps.map((s) => ({
      step: s.index,
      op: s.op,
      provider: s.provider,
      method: s.providerMethod,
      status: 'simulated',
      idempotencyKey: s.idempotencyKey,
      at,
      detail: `would ${s.effect === 'read' ? 'read via' : 'call'} ${s.providerMethod} [${s.risk}]${s.idempotent ? ' (idempotent)' : ''}`,
    }));
    record(receipts);
    return receipts;
  },
};

// every provider resolves to dry-run today
export const EXECUTORS: Record<string, Executor> = { 'dry-run': DryRunExecutor };

export function executorFor(_provider: string): Executor {
  return DryRunExecutor;
}
