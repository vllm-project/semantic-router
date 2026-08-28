import type { InsightsRecord, InsightsTrajectory } from './insightsPageTypes'

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
    `/api/router/v1/router_replay/${recordId}`,
    'insight record',
  )
}

export function fetchInsightsTrajectory(sessionId: string) {
  const query = new URLSearchParams({ session_id: sessionId })
  return fetchInsightsJSON<InsightsTrajectory>(
    `/api/router/v1/router_replay/trajectory?${query.toString()}`,
    'record trace',
  )
}

export function isInsightsReplayUnavailableError(error: unknown) {
  return error instanceof InsightsRequestError && error.status === 404
}
