import { ListTree, Lock, ServerOff, type LucideIcon } from 'lucide-react'

import type { Tone } from '../ui'
import type { InsightsUnavailableReason } from './insightsPageApi'
import type {
  InsightsAggregateCurrencySummary,
  InsightsAggregateResponse,
  InsightsRecord,
  InsightsTrafficBucket,
  InsightsTrafficSeries,
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

/** The interval the replay aggregate splits each range into, as its `interval` takes it. */
export const TRAFFIC_INTERVAL: Record<HomeRange, string> = { '1h': '5m', '24h': '1h', '7d': '6h' }

/** Why Home has no traffic to show, by the reason the replay API gave. */
export const TRAFFIC_UNAVAILABLE: Record<
  InsightsUnavailableReason,
  { title: string; description: string; tone: Tone; icon: LucideIcon }
> = {
  replay_off: {
    title: 'Traffic metrics need Router Replay',
    description:
      'Turn on the router_replay plugin for a recipe, then send a request through the gateway.',
    tone: 'blue',
    icon: ListTree,
  },
  router_unreachable: {
    title: 'Traffic is unavailable while the router is unreachable',
    description: 'Requests, latency and spend appear here once the router API answers again.',
    tone: 'slate',
    icon: ServerOff,
  },
  forbidden: {
    title: 'Your role cannot read Router Replay',
    description: 'Traffic metrics need the replay.read permission. Ask an admin for access.',
    tone: 'slate',
    icon: Lock,
  },
}

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

export interface TrafficPoint {
  /** The start of the interval, in ms since the epoch. */
  start: number
  requests: number
  cached: number
  failed: number
  p50?: number
  p95?: number
  /** Spend in the lead currency. */
  cost: number
  /** What the interval would have cost without routing, when any request was compared. */
  baseline?: number
}

export interface TrafficView {
  points: TrafficPoint[]
  intervalMs: number
  /** The points cover the whole range; otherwise only the newest records of it. */
  coversRange: boolean
  /** The records the points read, when they come from the newest records rather than the aggregate. */
  sampled?: number
  currency?: string
  /** Latency percentiles over everything the points read. */
  latency: { p50?: number; p95?: number }
  hasBaseline: boolean
}

const isFailed = (state: InsightsRecord['lifecycle_state']) =>
  state === 'failed' || state === 'aborted'

const leadCost = (bucket: InsightsTrafficBucket, currency?: string) =>
  currency ? bucket.cost.find((group) => group.currency === currency) : bucket.cost[0]

// The aggregate's since is to the minute, so its series can end in a sliver of
// the current minute; it folds into the full interval before it, which keeps
// the latency of the busier one.
function foldTail(buckets: InsightsTrafficBucket[], count: number): InsightsTrafficBucket[] {
  if (buckets.length <= count) return buckets
  const tail = buckets.slice(count - 1)
  const busiest = tail.reduce((best, bucket) => (bucket.requests > best.requests ? bucket : best))
  const costs = new Map<string, InsightsAggregateCurrencySummary>()
  for (const group of tail.flatMap((bucket) => bucket.cost)) {
    const total = costs.get(group.currency)
    costs.set(
      group.currency,
      total
        ? {
            ...total,
            total_saved: total.total_saved + group.total_saved,
            baseline_spend: total.baseline_spend + group.baseline_spend,
            actual_spend: total.actual_spend + group.actual_spend,
            total_spend: (total.total_spend ?? 0) + (group.total_spend ?? 0),
            cost_record_count: total.cost_record_count + group.cost_record_count,
          }
        : { ...group },
    )
  }
  return [
    ...buckets.slice(0, count - 1),
    {
      start: tail[0].start,
      requests: tail.reduce((sum, bucket) => sum + bucket.requests, 0),
      cached: tail.reduce((sum, bucket) => sum + bucket.cached, 0),
      failed: tail.reduce((sum, bucket) => sum + bucket.failed, 0),
      latency_p50_ms: busiest.latency_p50_ms,
      latency_p95_ms: busiest.latency_p95_ms,
      cost: [...costs.values()],
    },
  ]
}

/** The range's traffic from the aggregate's series, in the lead currency of its spend. */
export function trafficFromSeries(
  series: InsightsTrafficSeries,
  range: HomeRange,
  currency?: string,
): TrafficView {
  const buckets = foldTail(series.buckets, BUCKETS[range])
  const lead = currency ?? buckets.flatMap((bucket) => bucket.cost)[0]?.currency
  let hasBaseline = false
  const points = buckets.map((bucket): TrafficPoint => {
    const cost = leadCost(bucket, lead)
    const spend = cost?.total_spend ?? cost?.actual_spend ?? 0
    const compared = Boolean(cost && cost.cost_record_count > 0)
    if (compared) hasBaseline = true
    return {
      start: Date.parse(bucket.start),
      requests: bucket.requests,
      cached: bucket.cached,
      failed: bucket.failed,
      p50: bucket.latency_p50_ms,
      p95: bucket.latency_p95_ms,
      cost: spend,
      // Without routing, each compared request costs its baseline instead,
      // which is the spend plus the net saving.
      baseline: spend + (compared && cost ? cost.total_saved : 0),
    }
  })
  return {
    points,
    intervalMs: series.interval_seconds * 1000,
    coversRange: true,
    currency: lead,
    latency: { p50: series.latency_p50_ms, p95: series.latency_p95_ms },
    hasBaseline,
  }
}

/**
 * The traffic of the newest records, for a router whose aggregate has no
 * series: the whole range when they hold every record of it, else the span
 * from the oldest of them to now.
 */
export function trafficFromWindow(
  records: readonly InsightsRecord[],
  range: HomeRange,
  windowTotal: number,
  now = Date.now(),
): TrafficView | null {
  const times = records.map((record) => Date.parse(record.timestamp)).filter(Number.isFinite)
  if (times.length === 0) return null
  const coversRange = windowTotal <= records.length
  const count = BUCKETS[range]
  const start = coversRange ? now - RANGE_MS[range] : Math.min(...times)
  const width = Math.max(1, (now - start) / count)
  const currency = records.find((record) => record.currency)?.currency
  const points: TrafficPoint[] = Array.from({ length: count }, (_, index) => ({
    start: start + index * width,
    requests: 0,
    cached: 0,
    failed: 0,
    cost: 0,
    baseline: 0,
  }))
  const durations: number[][] = Array.from({ length: count }, () => [])
  let hasBaseline = false
  for (const record of records) {
    const time = Date.parse(record.timestamp)
    if (!Number.isFinite(time) || time < start || time > now) continue
    const index = Math.min(count - 1, Math.floor((time - start) / width))
    const point = points[index]
    point.requests += 1
    if (record.from_cache) point.cached += 1
    if (isFailed(record.lifecycle_state)) point.failed += 1
    const sameCurrency = !record.currency || record.currency === currency
    const spend = sameCurrency && typeof record.actual_cost === 'number' ? record.actual_cost : 0
    const compared = sameCurrency && typeof record.baseline_cost === 'number'
    if (compared) hasBaseline = true
    point.cost += spend
    point.baseline = (point.baseline ?? 0) + (compared ? (record.baseline_cost ?? 0) : spend)
    const duration = durationOf(record)
    if (duration !== undefined) durations[index].push(duration)
  }
  durations.forEach((values, index) => {
    points[index].p50 = percentile(values, 50)
    points[index].p95 = percentile(values, 95)
  })
  const stats = latencyStats(records)
  return {
    points,
    intervalMs: width,
    coversRange,
    sampled: records.length,
    currency,
    latency: { p50: stats.p50, p95: stats.p95 },
    hasBaseline,
  }
}

/** The range's traffic: the aggregate's series when the router returns one, else the newest records'. */
export function readTraffic(
  aggregate: InsightsAggregateResponse | null,
  records: readonly InsightsRecord[],
  windowTotal: number,
  range: HomeRange,
  now = Date.now(),
): TrafficView | null {
  if (aggregate?.traffic?.buckets?.length) {
    return trafficFromSeries(aggregate.traffic, range, spendSummary(aggregate).currency)
  }
  return trafficFromWindow(records, range, windowTotal, now)
}

/** The p95 of each interval for a sparkline, carrying the last one through intervals without any. */
export function carriedP95(points: readonly TrafficPoint[]): number[] {
  let previous = 0
  return points.map((point) => {
    previous = point.p95 ?? previous
    return previous
  })
}

const MINUTE_MS = 60_000
const HOUR_MS = 60 * MINUTE_MS
const DAY_MS = 24 * HOUR_MS
const TICK_STEPS = [5, 10, 15, 30, 60, 120, 180, 240, 360, 720, 1440].map(
  (minutes) => minutes * MINUTE_MS,
)

/**
 * Round local times that label a time axis from start to end with at most
 * about `target` ticks, such as every 4 hours on the hour for a day, and the
 * step between them.
 */
export function timeTicks(
  start: number,
  end: number,
  target = 6,
): { ticks: number[]; stepMs: number } {
  const span = Math.max(0, end - start)
  const stepMs = TICK_STEPS.find((step) => span / step <= target) ?? DAY_MS
  const unit = stepMs >= DAY_MS ? 'day' : stepMs >= HOUR_MS ? 'hour' : 'minute'
  const cursor = new Date(start)
  if (unit === 'day') cursor.setHours(0, 0, 0, 0)
  else if (unit === 'hour') cursor.setMinutes(0, 0, 0)
  else cursor.setSeconds(0, 0)
  const ticks: number[] = []
  while (cursor.getTime() <= end) {
    const round =
      unit === 'day' ||
      (unit === 'hour'
        ? cursor.getHours() % (stepMs / HOUR_MS) === 0
        : cursor.getMinutes() % (stepMs / MINUTE_MS) === 0)
    if (round && cursor.getTime() >= start) ticks.push(cursor.getTime())
    if (unit === 'day') cursor.setDate(cursor.getDate() + 1)
    else if (unit === 'hour') cursor.setHours(cursor.getHours() + 1)
    else cursor.setMinutes(cursor.getMinutes() + 1)
  }
  return { ticks, stepMs }
}

/** How long ago a refresh was, for the range control's tooltip. */
export function formatUpdatedAgo(at: number, now: number): string {
  const seconds = Math.max(0, Math.round((now - at) / 1000))
  if (seconds < 5) return 'Updated just now'
  if (seconds < 60) return `Updated ${seconds} s ago`
  const minutes = Math.floor(seconds / 60)
  return minutes < 60
    ? `Updated ${minutes} min ago`
    : `Updated at ${new Date(at).toLocaleTimeString()}`
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
