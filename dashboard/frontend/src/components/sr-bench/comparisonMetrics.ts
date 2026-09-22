export type ChangeDirection = 'positive' | 'negative' | 'neutral' | 'unknown'

export function changeDirection(value: number | null | undefined): ChangeDirection {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'unknown'
  return value > 0 ? 'positive' : value < 0 ? 'negative' : 'neutral'
}

/** Format a measured difference without hiding a small change as zero. */
export function formatSignedChange(value: number | null | undefined, digits = 2): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Unknown'
  if (value === 0) return '0'
  const magnitude = Math.abs(value)
  const sign = value > 0 ? '+' : '−'
  const formatted = magnitude.toLocaleString(
    undefined,
    magnitude < 10 ** -digits
      ? { maximumSignificantDigits: 2, notation: magnitude < 0.000001 ? 'scientific' : 'standard' }
      : { maximumFractionDigits: digits },
  )
  return sign + formatted
}

export function changeSummary(
  value: number | null | undefined,
  metric: 'quality' | 'cost',
): string {
  const direction = changeDirection(value)
  if (direction === 'unknown') return 'Not available'
  if (metric === 'quality') {
    return direction === 'positive'
      ? 'Higher score'
      : direction === 'negative'
        ? 'Lower score'
        : 'Same score'
  }
  return direction === 'positive'
    ? 'Lower cost'
    : direction === 'negative'
      ? 'Higher cost'
      : 'Same cost'
}
