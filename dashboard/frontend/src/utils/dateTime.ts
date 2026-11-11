export function formatDateTime(value?: string | null, fallback = '-'): string {
  if (!value) return fallback

  const timestamp = Date.parse(value)
  if (Number.isNaN(timestamp)) return fallback
  return new Date(timestamp).toLocaleString()
}

// A recorded duration in seconds to the millisecond, such as 1.489s.
export const formatSeconds = (ms: number): string => `${(ms / 1_000).toFixed(3)}s`

export function formatDurationBetween(start?: string | null, end?: string | null): string {
  if (!start) return '-'
  const startTime = Date.parse(start)
  const endTime = end ? Date.parse(end) : Date.now()
  if (Number.isNaN(startTime) || Number.isNaN(endTime)) return '-'

  const durationMs = Math.max(0, endTime - startTime)
  if (durationMs < 1_000) return `${durationMs}ms`
  if (durationMs < 60_000) return `${(durationMs / 1_000).toFixed(1)}s`
  if (durationMs < 3_600_000) {
    return `${Math.floor(durationMs / 60_000)}m ${Math.floor((durationMs % 60_000) / 1_000)}s`
  }
  return `${Math.floor(durationMs / 3_600_000)}h ${Math.floor((durationMs % 3_600_000) / 60_000)}m`
}
