import type { InsightsListResponse, InsightsRecord, InsightsTrajectory } from './insightsPageTypes'
import { REVIEW_REQUEST_KIND } from './insightsReviewSupport'

export class InsightsRequestError extends Error {
  readonly status: number

  constructor(label: string, status: number, statusText: string) {
    super(`Failed to fetch ${label}: ${status} ${statusText}`)
    this.name = 'InsightsRequestError'
    this.status = status
  }
}

export async function fetchInsightsJSON<T>(url: string, label: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new InsightsRequestError(label, response.status, response.statusText)
  }
  return (await response.json()) as T
}

export function fetchInsightsRecord(recordId: string) {
  return fetchInsightsJSON<InsightsRecord>(
    `/api/router/api/v1/observability/replays/${recordId}`,
    'insight record',
  )
}

export function fetchInsightsTrajectory(sessionId: string, recipe: string) {
  const query = new URLSearchParams({ session_id: sessionId, recipe })
  return fetchInsightsJSON<InsightsTrajectory>(
    `/api/router/api/v1/observability/replays/trajectory?${query.toString()}`,
    'record trace',
  )
}

// The approval reviews of a session in every recipe: a review entrypoint
// usually routes through a recipe of its own.
export async function fetchInsightsSessionReviews(sessionId: string) {
  const query = new URLSearchParams({
    request_kind: REVIEW_REQUEST_KIND,
    session_id: sessionId,
    limit: '100',
  })
  const list = await fetchInsightsJSON<InsightsListResponse>(
    `/api/router/api/v1/observability/replays?${query.toString()}`,
    'session reviews',
  )
  return list.data || []
}

export function isInsightsReplayUnavailableError(error: unknown) {
  return error instanceof InsightsRequestError && error.status === 404
}
