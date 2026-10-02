export async function fetchDashboardJson<T>(
  url: string,
  label: string,
  fetcher: typeof fetch = fetch,
): Promise<T> {
  const response = await fetcher(url)
  if (!response.ok) {
    throw new Error(`${label} request failed (HTTP ${response.status})`)
  }
  return (await response.json()) as T
}

// Waits for every request, so a later success cannot hide an earlier failure.
export async function settleDashboardRequests(
  requests: ReadonlyArray<Promise<unknown>>,
): Promise<string | null> {
  const results = await Promise.allSettled(requests)
  const failures = results.flatMap((result) =>
    result.status === 'rejected'
      ? [result.reason instanceof Error ? result.reason.message : 'Failed to load dashboard data']
      : [],
  )
  return failures.length > 0 ? failures.join('; ') : null
}
