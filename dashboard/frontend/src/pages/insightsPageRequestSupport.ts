import { InsightsRequestError } from './insightsPageApi'

export async function fetchAbortableInsightsJSON<T>(
  url: string,
  label: string,
  signal: AbortSignal,
): Promise<T> {
  const response = await fetch(url, { signal })
  if (!response.ok) {
    throw new InsightsRequestError(label, response.status, response.statusText)
  }
  return (await response.json()) as T
}

export function isAbortError(error: unknown): boolean {
  return Boolean(
    error && typeof error === 'object' && 'name' in error && error.name === 'AbortError',
  )
}
