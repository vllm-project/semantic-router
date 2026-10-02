import type { RecoveryRequest } from './types'

export interface PendingRecovery {
  version: 1
  actorID: string
  parentID: string
  request: RecoveryRequest
}

export const recoveryKey = (actorID: string, parentID: string) =>
  `sr-bench-recovery:${encodeURIComponent(actorID)}:${encodeURIComponent(parentID)}`
const maximumBytes = 2 * 1024 * 1024

export function readRecovery(
  actorID: string,
  parentID: string,
): {
  saved: PendingRecovery | null
  error: string
} {
  try {
    if (!actorID || !parentID) throw new Error('Missing authenticated scope')
    const raw = sessionStorage.getItem(recoveryKey(actorID, parentID))
    if (!raw) return { saved: null, error: '' }
    if (new Blob([raw]).size > maximumBytes) throw new Error('Oversized saved recovery')
    const saved = JSON.parse(raw) as PendingRecovery
    const body = saved.request
    if (
      saved.version !== 1 ||
      saved.actorID !== actorID ||
      saved.parentID !== parentID ||
      !['undispatched', 'failed'].includes(body.mode) ||
      typeof body.idempotency_key !== 'string' ||
      !body.idempotency_key.trim() ||
      typeof body.plan_sha256 !== 'string' ||
      !/^[a-f0-9]{64}$/.test(body.plan_sha256) ||
      !Array.isArray(body.cells) ||
      !body.cells.length ||
      body.cells.some(
        (cell) =>
          !cell ||
          typeof cell.case_id !== 'string' ||
          !cell.case_id ||
          typeof cell.target_id !== 'string' ||
          !cell.target_id,
      ) ||
      (body.mode === 'failed' && body.acknowledge_new_attempt !== true)
    )
      throw new Error('Invalid saved recovery')
    return { saved, error: '' }
  } catch {
    return {
      saved: null,
      error:
        'The saved recovery cannot be read safely for this account and parent. Reconcile its child attempts before clearing browser data or creating another recovery.',
    }
  }
}

export function saveRecovery(saved: PendingRecovery) {
  const current = readRecovery(saved.actorID, saved.parentID)
  if (current.error || (current.saved && JSON.stringify(current.saved) !== JSON.stringify(saved)))
    throw new Error(
      'Another recovery needs reconciliation. Reload to inspect it. No request was sent.',
    )
  const raw = JSON.stringify(saved)
  if (new Blob([raw]).size > maximumBytes)
    throw new Error(
      'This recovery is too large to preserve in this tab. Use the CLI. No request was sent.',
    )
  try {
    sessionStorage.setItem(recoveryKey(saved.actorID, saved.parentID), raw)
  } catch {
    throw new Error(
      'This browser could not preserve the recovery safely. No request was sent. Enable session storage or use the CLI.',
    )
  }
}

export function clearRecovery(saved: PendingRecovery): boolean {
  const current = readRecovery(saved.actorID, saved.parentID)
  if (current.error || current.saved?.request.idempotency_key !== saved.request.idempotency_key)
    return false
  sessionStorage.removeItem(recoveryKey(saved.actorID, saved.parentID))
  return true
}
