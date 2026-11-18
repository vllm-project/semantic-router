import { isInsightsNoCurrency } from '../utils/insightsCost'

const NONE = '—'

const compactCount = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
})
const significant = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 3 })
const percent = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 })

const missing = (value: number | null | undefined): value is null | undefined =>
  value === null || value === undefined || !Number.isFinite(value)

/** A count from Prometheus, whose increases are extrapolated and need not be whole. */
export function formatCount(value: number | null | undefined): string {
  if (missing(value)) return NONE
  return Math.abs(value) < 1_000 ? String(Math.round(value)) : compactCount.format(value)
}

/** A per-second rate, per minute or per hour when that reads better than a small fraction. */
export function formatRate(perSecond: number | null | undefined): string {
  if (missing(perSecond)) return NONE
  if (perSecond === 0) return '0/s'
  if (perSecond >= 1_000) return `${compactCount.format(perSecond)}/s`
  if (perSecond >= 1) return `${significant.format(perSecond)}/s`
  if (perSecond * 60 >= 1) return `${significant.format(perSecond * 60)}/min`
  return `${significant.format(perSecond * 3_600)}/h`
}

/** Axis ticks of a rate or count: distinct at small magnitudes, short at large ones. */
export function formatAxisNumber(value: number): string {
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumSignificantDigits: 3,
  }).format(value)
}

export function formatSeconds(seconds: number | null | undefined): string {
  if (missing(seconds)) return NONE
  if (seconds > 0 && seconds < 0.001) return '<1 ms'
  if (seconds < 1) return `${Math.round(seconds * 1_000)} ms`
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${Math.round(seconds % 60)}s`
}

export function formatSecondsAxis(seconds: number): string {
  if (seconds === 0) return '0'
  return seconds < 1
    ? `${significant.format(seconds * 1_000)} ms`
    : `${significant.format(seconds)} s`
}

/** A ratio between 0 and 1 as a percentage. */
export function formatPercent(ratio: number | null | undefined): string {
  if (missing(ratio)) return NONE
  if (ratio > 0 && ratio < 0.001) return '<0.1%'
  return `${percent.format(ratio * 100)}%`
}

export function formatMoney(amount: number | null | undefined, currency: string): string {
  if (missing(amount)) return NONE
  const digits =
    amount === 0 || Math.abs(amount) >= 1
      ? { minimumFractionDigits: 2, maximumFractionDigits: 2 }
      : { maximumSignificantDigits: 3 }
  if (isInsightsNoCurrency(currency)) return new Intl.NumberFormat('en-US', digits).format(amount)
  try {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency, ...digits }).format(amount)
  } catch {
    // Not an ISO 4217 code: keep the label the Router exported.
    return `${new Intl.NumberFormat('en-US', digits).format(amount)} ${currency}`
  }
}

const clock = new Intl.DateTimeFormat(undefined, {
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
})
const day = new Intl.DateTimeFormat(undefined, { weekday: 'short', day: 'numeric' })
const moment = new Intl.DateTimeFormat(undefined, {
  weekday: 'short',
  day: 'numeric',
  month: 'short',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
})

/** A time-axis tick: the clock within a day, the day across days. */
export function formatTimeTick(time: number, spanMs: number): string {
  return spanMs > 2 * 86_400_000 ? day.format(time) : clock.format(time)
}

export function formatMoment(time: number): string {
  return moment.format(time)
}

export function formatClock(time: number): string {
  return clock.format(time)
}
