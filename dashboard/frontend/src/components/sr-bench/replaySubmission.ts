import type { ReplayRequest } from './types'

export interface PendingReplay {
  version: 1
  actorID: string
  request: ReplayRequest
}

export const replayKey = (actorID: string) => `sr-bench-replay:${encodeURIComponent(actorID)}`

export function readReplay(actorID: string): { saved: PendingReplay | null; error: string } {
  try {
    if (!actorID) throw new Error('Missing account')
    const raw = sessionStorage.getItem(replayKey(actorID))
    if (!raw) return { saved: null, error: '' }
    if (raw.length > 16384) throw new Error('Oversized submission')
    const saved = JSON.parse(raw) as PendingReplay
    if (
      saved.version !== 1 ||
      saved.actorID !== actorID ||
      !saved.request ||
      ['baseline_run_id', 'preview_run_id', 'idempotency_key'].some((field) => {
        const value = saved.request[field as keyof ReplayRequest]
        return typeof value !== 'string' || !value.trim()
      })
    )
      throw new Error('Invalid submission')
    return { saved, error: '' }
  } catch {
    return {
      saved: null,
      error:
        'The saved estimate submission cannot be read safely. Reconcile the run inventory before clearing this tab’s saved request.',
    }
  }
}

export function saveReplay(saved: PendingReplay) {
  const current = readReplay(saved.actorID)
  if (current.error || (current.saved && JSON.stringify(current.saved) !== JSON.stringify(saved)))
    throw new Error(
      'Another estimate submission needs reconciliation. Reload to inspect it. No request was sent.',
    )
  try {
    sessionStorage.setItem(replayKey(saved.actorID), JSON.stringify(saved))
  } catch {
    throw new Error(
      'This browser could not preserve the estimate submission. No request was sent. Enable session storage or use the CLI.',
    )
  }
}

export function clearReplay(saved: PendingReplay): boolean {
  const current = readReplay(saved.actorID)
  if (current.error || JSON.stringify(current.saved) !== JSON.stringify(saved)) return false
  sessionStorage.removeItem(replayKey(saved.actorID))
  return true
}
