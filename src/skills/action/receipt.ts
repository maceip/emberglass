// action/receipt.ts — the audit log for plan execution.
//
// In dry-run every receipt is 'simulated'. The log is in-memory only; nothing is persisted or
// sent anywhere. It exists so the UI can show "what would have happened" and so a future real
// executor inherits an audit trail from day one.
import type { Receipt } from '../types.ts';

const LOG: Receipt[] = [];

export function record(receipts: Receipt[]): void {
  for (const r of receipts) LOG.push(r);
}
export function auditLog(): Receipt[] {
  return [...LOG];
}
export function clearAudit(): void {
  LOG.length = 0;
}
