const MINIMUM_VISIBLE_COST = 0.0001

// ISO 4217 XXX names no currency: prices configured in plain units.
const NO_CURRENCY = 'XXX'

export interface InsightsCostFormat {
  /** A total rounds to cents from one cent up; a request's cost keeps four decimals. */
  total?: boolean
  /** Leave the currency out where a heading or column already names it. */
  bare?: boolean
}

export function isInsightsNoCurrency(currency?: string): boolean {
  return currency?.trim().toUpperCase() === NO_CURRENCY
}

/** The currency a heading names: empty for prices configured without one. */
export function insightsCurrencyLabel(currency?: string): string {
  return isInsightsNoCurrency(currency) ? '' : (currency?.trim() ?? '')
}

/** Keep nonzero estimates distinguishable from free usage at display precision. */
export function formatInsightsCost(
  value?: number,
  currency?: string,
  format: InsightsCostFormat = {},
): string {
  const currencyCode = currency?.trim()
  if (typeof value !== 'number' || !Number.isFinite(value) || !currencyCode) return 'N/A'

  const tiny = value !== 0 && Math.abs(value) < MINIMUM_VISIBLE_COST
  const displayed = tiny ? Math.sign(value) * MINIMUM_VISIBLE_COST : value === 0 ? 0 : value
  // A negative amount between -0.0001 and zero is greater than the negative bound.
  const prefix = tiny ? (value < 0 ? '>' : '<') : ''
  const magnitude = Math.abs(displayed)
  const digits = {
    minimumFractionDigits: magnitude >= 0.01 || (format.total && displayed === 0) ? 2 : 4,
    maximumFractionDigits: format.total && magnitude >= 0.01 ? 2 : 4,
  }
  const plain = format.bare || isInsightsNoCurrency(currencyCode)
  try {
    return (
      prefix +
      new Intl.NumberFormat(
        'en-US',
        plain ? digits : { style: 'currency', currency: currencyCode, ...digits },
      ).format(displayed)
    )
  } catch {
    return `${prefix}${new Intl.NumberFormat('en-US', digits).format(displayed)} ${currencyCode}`
  }
}

export function formatInsightsCount(value: number): string {
  return value.toLocaleString('en-US')
}

/** A large count shortened to one decimal (7.0B, 412.3M); counts below a thousand stay whole. */
export function formatInsightsCompactCount(value: number): string {
  if (Math.abs(value) < 1000) return formatInsightsCount(value)
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value)
}

export function formatInsightsShare(part: number, whole: number): string {
  return whole > 0 ? `${((part / whole) * 100).toFixed(1)}%` : 'N/A'
}
