import { validateManifest } from './model'
import type { Manifest } from './types'

export interface PendingSubmission {
  version: 1
  actorID: string
  manifest: Manifest
  idempotencyKey: string
}

const maximumBytes = 2 * 1024 * 1024
export const submissionKey = (actorID: string) => `sr-bench-submission:${actorID}`

export function readSubmission(actorID: string): {
  request: PendingSubmission | null
  error: string
} {
  if (!actorID) return { request: null, error: 'An authenticated account is required.' }
  try {
    const raw = sessionStorage.getItem(submissionKey(actorID))
    if (!raw) return { request: null, error: '' }
    if (new Blob([raw]).size > maximumBytes) throw new Error('Oversized saved submission')
    const request = JSON.parse(raw) as PendingSubmission
    if (
      request.version !== 1 ||
      request.actorID !== actorID ||
      !request.idempotencyKey ||
      validateManifest(request.manifest)
    )
      throw new Error('Invalid saved submission')
    return { request, error: '' }
  } catch {
    return {
      request: null,
      error:
        'The saved submission cannot be read safely in this tab. Reconcile the existing run inventory before clearing browser data or starting another attempt.',
    }
  }
}

export function saveSubmission(request: PendingSubmission) {
  const raw = JSON.stringify(request)
  if (new Blob([raw]).size > maximumBytes)
    throw new Error(
      'This frozen plan is too large to preserve safely in this tab. Start it with the CLI.',
    )
  // Failure must propagate before any paid request is submitted.
  const current = readSubmission(request.actorID)
  if (
    current.error ||
    (current.request &&
      (current.request.idempotencyKey !== request.idempotencyKey ||
        JSON.stringify(current.request.manifest) !== JSON.stringify(request.manifest)))
  )
    throw new Error(
      'Another saved submission needs reconciliation. Reload this page to inspect it. No request was sent.',
    )
  try {
    sessionStorage.setItem(submissionKey(request.actorID), raw)
  } catch {
    throw new Error(
      'This browser could not preserve the submission safely. No request was sent. Enable session storage or start the evaluation with the CLI.',
    )
  }
}

export function clearSubmission(request: PendingSubmission): boolean {
  const current = readSubmission(request.actorID)
  if (
    current.error ||
    current.request?.actorID !== request.actorID ||
    current.request?.idempotencyKey !== request.idempotencyKey
  )
    return false
  sessionStorage.removeItem(submissionKey(request.actorID))
  return true
}
