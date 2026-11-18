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

// The list endpoint returns at most 100 records a page; past this many pages a
// session keeps its newest reviews and is marked truncated.
const SESSION_REVIEW_PAGES = 20

// The approval reviews of a session in every recipe: a review entrypoint
// usually routes through a recipe of its own.
export async function fetchInsightsSessionReviews(sessionId: string) {
  const reviews = new Map<string, InsightsRecord>()
  let offset = 0
  for (let page = 0; page < SESSION_REVIEW_PAGES; page += 1) {
    const query = new URLSearchParams({
      request_kind: REVIEW_REQUEST_KIND,
      session_id: sessionId,
      limit: '100',
      offset: String(offset),
    })
    const list = await fetchInsightsJSON<InsightsListResponse>(
      `/api/router/api/v1/observability/replays?${query.toString()}`,
      'session reviews',
    )
    const data = list.data || []
    // A review recorded while paging shifts the newest-first pages; the id keeps each once.
    for (const review of data) reviews.set(review.id, review)
    if (!list.has_more) return { reviews: [...reviews.values()], truncated: false }
    offset += data.length
  }
  return { reviews: [...reviews.values()], truncated: true }
}

export function isInsightsReplayUnavailableError(error: unknown) {
  return error instanceof InsightsRequestError && error.status === 404
}

/**
 * Why replay data cannot load, when the answer says: Replay is off (404), the
 * viewer may not read it (401, 403), or the dashboard cannot reach the router
 * (502, 503, 504, as when it is down or on setup standby).
 */
export type InsightsUnavailableReason = 'replay_off' | 'forbidden' | 'router_unreachable'

export function insightsUnavailableReason(error: unknown): InsightsUnavailableReason | undefined {
  if (!(error instanceof InsightsRequestError)) return undefined
  if (error.status === 404) return 'replay_off'
  if (error.status === 401 || error.status === 403) return 'forbidden'
  if (error.status === 502 || error.status === 503 || error.status === 504) {
    return 'router_unreachable'
  }
  return undefined
}
