export function elapsedSeconds(start: string | undefined, now: number): number | null {
  const value = start ? Date.parse(start) : NaN
  return Number.isFinite(value) && Number.isFinite(now) ? Math.max(0, (now - value) / 1000) : null
}

export function duration(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return 'Not recorded'
  const total = Math.max(0, Math.floor(value))
  const seconds = total % 60
  const minutes = Math.floor(total / 60) % 60
  const hours = Math.floor(total / 3600)
  return hours
    ? `${hours}h ${minutes}m ${seconds}s`
    : minutes
      ? `${minutes}m ${seconds}s`
      : `${seconds}s`
}

export function responseBytes(value: number): string {
  return Number.isSafeInteger(value) && value >= 0
    ? `${value.toLocaleString()} bytes`
    : 'Not recorded'
}
