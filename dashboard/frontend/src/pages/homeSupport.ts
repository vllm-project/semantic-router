import type {
  InsightsAggregateCurrencySummary,
  InsightsAggregateResponse,
  InsightsRecord,
} from './insightsPageTypes'

export type HomeRange = '1h' | '24h' | '7d'

export const HOME_RANGES: { value: HomeRange; label: string }[] = [
  { value: '1h', label: '1h' },
  { value: '24h', label: '24h' },
  { value: '7d', label: '7d' },
]

export const HOME_RANGE_LABEL: Record<HomeRange, string> = {
  '1h': 'last hour',
  '24h': 'last 24 hours',
  '7d': 'last 7 days',
}

const RANGE_MS: Record<HomeRange, number> = {
  '1h': 3_600_000,
  '24h': 86_400_000,
  '7d': 604_800_000,
}

// 5-minute, 1-hour and 6-hour buckets.
const BUCKETS: Record<HomeRange, number> = { '1h': 12, '24h': 24, '7d': 28 }

export function readHomeRange(value: string | null): HomeRange {
  return value === '1h' || value === '7d' ? value : '24h'
}

export function percentile(values: readonly number[], p: number): number | undefined {
  if (values.length === 0) return undefined
  const sorted = [...values].sort((left, right) => left - right)
  const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))
  return sorted[rank]
}

const durationOf = (record: InsightsRecord) =>
  record.lifecycle_state !== 'in_progress' &&
  typeof record.duration_ms === 'number' &&
  record.duration_ms >= 0
    ? record.duration_ms
    : undefined

export interface LatencyStats {
  p50?: number
  p95?: number
  /** How many records the percentiles read. */
  sampled: number
}

export function latencyStats(records: readonly InsightsRecord[]): LatencyStats {
  const durations = records.map(durationOf).filter((value): value is number => value !== undefined)
  return {
    p50: percentile(durations, 50),
    p95: percentile(durations, 95),
    sampled: durations.length,
  }
}

export interface TrafficSeries {
  requests: number[]
  p95: number[]
  cost: number[]
}

/**
 * Requests, p95 latency and cost per bucket of the range, from the window of
 * newest records. Null unless the window holds every record of the range,
 * since a cut-off window would draw a false drop at its start.
 */
export function bucketTraffic(
  records: readonly InsightsRecord[],
  range: HomeRange,
  windowTotal: number,
  now = Date.now(),
): TrafficSeries | null {
  if (records.length === 0 || windowTotal > records.length) return null
  const span = RANGE_MS[range]
  const count = BUCKETS[range]
  const start = now - span
  const width = span / count
  const requests = Array<number>(count).fill(0)
  const cost = Array<number>(count).fill(0)
  const durations: number[][] = Array.from({ length: count }, () => [])
  for (const record of records) {
    const time = Date.parse(record.timestamp)
    if (!Number.isFinite(time) || time < start || time > now) continue
    const index = Math.min(count - 1, Math.floor((time - start) / width))
    requests[index] += 1
    if (typeof record.actual_cost === 'number') cost[index] += record.actual_cost
    const duration = durationOf(record)
    if (duration !== undefined) durations[index].push(duration)
  }
  let previous = 0
  const p95 = durations.map((values) => {
    previous = percentile(values, 95) ?? previous
    return previous
  })
  return { requests, p95, cost }
}

export interface SpendSummary {
  currency?: string
  spend: number
  /** Net savings against the baseline; undefined when no request was compared. */
  saved?: number
  baseline?: number
  /** Further currencies the totals leave out of the lead one. */
  otherCurrencies: number
}

/** The lead currency's spend and savings, as the Requests summary reads them. */
export function spendSummary(aggregate: InsightsAggregateResponse): SpendSummary {
  const summary = aggregate.summary
  const groups: Array<InsightsAggregateCurrencySummary | typeof summary> = summary.by_currency
    ?.length
    ? summary.by_currency
    : [summary]
  const lead = groups[0]
  const compared = lead.cost_record_count > 0
  return {
    currency: lead.currency,
    spend: lead.total_spend ?? lead.actual_spend,
    saved: compared ? lead.total_saved : undefined,
    baseline: compared ? lead.baseline_spend : undefined,
    otherCurrencies: groups.length - 1,
  }
}

export function greeting(hour: number): string {
  if (hour < 5) return 'Good evening'
  if (hour < 12) return 'Good morning'
  if (hour < 18) return 'Good afternoon'
  return 'Good evening'
}

/** The largest values of a distribution, with their share of the whole. */
export function topShares(
  values: readonly { name: string; value: number }[] | undefined,
  limit = 6,
): { name: string; value: number; share: number }[] {
  const entries = (values ?? []).filter((entry) => entry.name && entry.value > 0)
  const total = entries.reduce((sum, entry) => sum + entry.value, 0)
  return [...entries]
    .sort((left, right) => right.value - left.value)
    .slice(0, limit)
    .map((entry) => ({ ...entry, share: total > 0 ? (entry.value / total) * 100 : 0 }))
}
