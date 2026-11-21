import { fetchInsightsJSON } from './insightsPageApi'
import type { RunResponse } from './insightsRunTypes'

/** How long one read of a run serves the views that open it again. */
const FRESH_MS = 30_000
const KEPT = 6

const reads = new Map<string, { at: number; response: Promise<RunResponse> }>()

/**
 * One agent run, read once for the views that open it within a short while:
 * expanding a session in the Sessions view, then opening its run page, reads
 * it once. A failed read is forgotten so the next one retries.
 */
export function loadRun(sessionId: string, { fresh = false } = {}): Promise<RunResponse> {
  const now = Date.now()
  const kept = reads.get(sessionId)
  if (!fresh && kept && now - kept.at < FRESH_MS) return kept.response
  const query = new URLSearchParams({ session_id: sessionId })
  const response = fetchInsightsJSON<RunResponse>(
    `/api/router/api/v1/observability/replays/run?${query.toString()}`,
    'agent run',
  )
  reads.delete(sessionId)
  reads.set(sessionId, { at: now, response })
  response.catch(() => {
    if (reads.get(sessionId)?.response === response) reads.delete(sessionId)
  })
  while (reads.size > KEPT) reads.delete(reads.keys().next().value as string)
  return response
}

/** Starts reading a run ahead of the click that opens it, as when a pointer rests on its row. */
export function prefetchRun(sessionId: string): void {
  loadRun(sessionId).catch(() => undefined)
}

/** Forgets every kept read, for tests. */
export function clearRunReads(): void {
  reads.clear()
}
