const MINIMUM_VISIBLE_COST = 0.0001

/** Keep nonzero estimates distinguishable from free usage at display precision. */
export function formatInsightsCost(value?: number, currency?: string): string {
  const currencyCode = currency?.trim()
  if (typeof value !== 'number' || !Number.isFinite(value) || !currencyCode) return 'N/A'

  const tiny = value !== 0 && Math.abs(value) < MINIMUM_VISIBLE_COST
  const displayed = tiny ? Math.sign(value) * MINIMUM_VISIBLE_COST : value === 0 ? 0 : value
  // A negative amount between -0.0001 and zero is greater than the negative bound.
  const prefix = tiny ? (value < 0 ? '>' : '<') : ''
  try {
    return (
      prefix +
      new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currencyCode,
        minimumFractionDigits: Math.abs(displayed) >= 0.01 ? 2 : 4,
        maximumFractionDigits: 4,
      }).format(displayed)
    )
  } catch {
    return `${prefix}${displayed.toFixed(4)} ${currencyCode}`
  }
}
