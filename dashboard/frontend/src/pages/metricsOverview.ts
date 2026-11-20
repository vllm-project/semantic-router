import { LATENCY_QUANTILES, type ModelMetric } from './metricsQueries'
import {
  ratioValues,
  sumValues,
  totalOf,
  valuesWhere,
  type MetricSample,
  type MetricSeries,
} from './metricsSeries'

const failedSeries = (outcomes: readonly MetricSeries[]) =>
  outcomes.filter((series) => series.labels.outcome !== 'success').map((series) => series.values)

export function requestSeries(outcomes: readonly MetricSeries[], count: number) {
  const succeeded = valuesWhere(outcomes, 'outcome', 'success', count)
  const failed = sumValues(failedSeries(outcomes), count)
  const total = sumValues([succeeded, failed], count)
  return { succeeded, failed, total, errorRate: ratioValues(failed, total) }
}

export interface RequestSummary {
  total: number | null
  failed: number | null
  /** Share of requests that did not succeed; null without requests. */
  errorRate: number | null
}

export function requestSummary(outcomes: readonly MetricSample[]): RequestSummary {
  const total = totalOf(outcomes)
  if (total === null) return { total: null, failed: null, errorRate: null }
  const failed = totalOf(outcomes.filter((sample) => sample.labels.outcome !== 'success')) ?? 0
  return { total, failed, errorRate: total > 0 ? failed / total : null }
}

export type QuantileKey = (typeof LATENCY_QUANTILES)[number]['key']

export function quantileSummary(
  samples: readonly MetricSample[],
): Record<QuantileKey, number | null> {
  const value = (key: QuantileKey) =>
    samples.find((sample) => sample.labels.quantile === key)?.value ?? null
  return { p50: value('p50'), p95: value('p95'), p99: value('p99') }
}

export function quantileSeries(
  series: readonly MetricSeries[],
  count: number,
): Record<QuantileKey, (number | null)[]> {
  const values = (key: QuantileKey) => valuesWhere(series, 'quantile', key, count)
  return { p50: values('p50'), p95: values('p95'), p99: values('p99') }
}

export function tokenSummary(samples: readonly MetricSample[]) {
  const prompt = totalOf(samples.filter((sample) => sample.labels.type === 'prompt'))
  const completion = totalOf(samples.filter((sample) => sample.labels.type === 'completion'))
  return { prompt, completion, total: totalOf(samples) }
}

export function tokenSeries(series: readonly MetricSeries[], count: number) {
  const prompt = valuesWhere(series, 'type', 'prompt', count)
  const completion = valuesWhere(series, 'type', 'completion', count)
  return { prompt, completion, total: sumValues([prompt, completion], count) }
}

export interface CurrencyAmount {
  currency: string
  amount: number
}

/** Amounts per currency, largest first. Empty when the Router exports no cost. */
export function amountsByCurrency(samples: readonly MetricSample[]): CurrencyAmount[] {
  const amounts = new Map<string, number>()
  for (const sample of samples) {
    if (sample.value === null) continue
    const currency = sample.labels.currency || 'USD'
    amounts.set(currency, (amounts.get(currency) ?? 0) + sample.value)
  }
  return [...amounts]
    .map(([currency, amount]) => ({ currency, amount }))
    .sort(
      (left, right) => right.amount - left.amount || left.currency.localeCompare(right.currency),
    )
}

export interface CacheSummary {
  hits: number
  misses: number
  lookups: number
  hitRate: number | null
}

/** Null when the Router reported no response-cache lookup: no decision uses the cache. */
export function cacheSummary(samples: readonly MetricSample[]): CacheSummary | null {
  if (!samples.length) return null
  const hits = totalOf(samples.filter((sample) => sample.labels.result === 'hit')) ?? 0
  const misses = totalOf(samples.filter((sample) => sample.labels.result === 'miss')) ?? 0
  const lookups = hits + misses
  return { hits, misses, lookups, hitRate: lookups > 0 ? hits / lookups : null }
}

export function cacheRateSeries(series: readonly MetricSeries[], count: number) {
  const hits = valuesWhere(series, 'result', 'hit', count)
  const misses = valuesWhere(series, 'result', 'miss', count)
  return ratioValues(hits, sumValues([hits, misses], count))
}

export interface ModelRow {
  model: string
  requests: number | null
  errors: number | null
  /** Errors over requests, capped at all of them; null without requests. */
  errorRate: number | null
  /** P95 completion latency in seconds. */
  latency: number | null
  tokens: number | null
  cost: CurrencyAmount[]
}

/** Rows of the per-model table from the `stat`-tagged summary query, busiest first. */
export function modelRows(samples: readonly MetricSample[]): ModelRow[] {
  const rows = new Map<string, ModelRow>()
  const costs = new Map<string, MetricSample[]>()
  for (const sample of samples) {
    const model = sample.labels.model
    if (!model) continue
    const row = rows.get(model) ?? {
      model,
      requests: null,
      errors: null,
      errorRate: null,
      latency: null,
      tokens: null,
      cost: [],
    }
    rows.set(model, row)
    const stat = sample.labels.stat as ModelMetric | undefined
    if (stat === 'cost') costs.set(model, [...(costs.get(model) ?? []), sample])
    else if (stat === 'requests' || stat === 'errors' || stat === 'latency' || stat === 'tokens') {
      row[stat] = sample.value
    }
  }
  for (const row of rows.values()) {
    row.cost = amountsByCurrency(costs.get(row.model) ?? [])
    row.errorRate =
      row.requests !== null && row.requests > 0
        ? Math.min(1, (row.errors ?? 0) / row.requests)
        : null
  }
  return [...rows.values()].sort(
    (left, right) =>
      (right.requests ?? -1) - (left.requests ?? -1) || left.model.localeCompare(right.model),
  )
}

/** The recipe whose decisions keep unscoped keys in Router metrics. */
const DEFAULT_RECIPE = 'default'

function decodeQueryComponent(value: string): string {
  try {
    return decodeURIComponent(value.replace(/\+/g, ' '))
  } catch {
    return value
  }
}

/**
 * Splits a decision key from the Router's cache metrics: a default-recipe
 * decision's own name, or `recipe::decision` with both parts query-escaped.
 */
export function parseDecisionKey(key: string): { recipe: string; decision: string } {
  const separator = key.indexOf('::')
  if (separator < 0) return { recipe: DEFAULT_RECIPE, decision: key }
  return {
    recipe: decodeQueryComponent(key.slice(0, separator)),
    decision: decodeQueryComponent(key.slice(separator + 2)),
  }
}

export const decisionId = (recipe: string, decision: string) => `${recipe}\u0000${decision}`

export interface DecisionRow {
  id: string
  recipe: string
  decision: string
  /** Model selections the decision made. */
  requests: number
  models: { model: string; requests: number }[]
  cache: CacheSummary | null
}

export function decisionRows(
  selections: readonly MetricSample[],
  cache: readonly MetricSample[],
): DecisionRow[] {
  const rows = new Map<string, DecisionRow>()
  const row = (recipe: string, decision: string) => {
    const id = decisionId(recipe, decision)
    const existing = rows.get(id)
    if (existing) return existing
    const created: DecisionRow = { id, recipe, decision, requests: 0, models: [], cache: null }
    rows.set(id, created)
    return created
  }
  for (const sample of selections) {
    if (sample.value === null) continue
    const entry = row(sample.labels.recipe || DEFAULT_RECIPE, sample.labels.decision || 'unknown')
    entry.requests += sample.value
    if (sample.labels.model)
      entry.models.push({ model: sample.labels.model, requests: sample.value })
  }
  const lookups = new Map<string, MetricSample[]>()
  for (const sample of cache) {
    const { recipe, decision } = parseDecisionKey(sample.labels.decision_name ?? '')
    const id = row(recipe, decision || 'unknown').id
    lookups.set(id, [...(lookups.get(id) ?? []), sample])
  }
  for (const [id, samples] of lookups) {
    const entry = rows.get(id)
    if (entry) entry.cache = cacheSummary(samples)
  }
  for (const entry of rows.values()) {
    entry.models.sort((left, right) => right.requests - left.requests)
  }
  return [...rows.values()].sort(
    (left, right) =>
      right.requests - left.requests ||
      left.recipe.localeCompare(right.recipe) ||
      left.decision.localeCompare(right.decision),
  )
}

export interface ScrapeTarget {
  instance: string
  values: (number | null)[]
  /** Whether the last scrape in the window succeeded; null without one. */
  up: boolean | null
  /** Share of scrapes in the window that succeeded. */
  availability: number | null
}

export function scrapeTargets(up: readonly MetricSeries[]): ScrapeTarget[] {
  return up
    .map((series) => {
      const observed = series.values.filter((value) => value !== null)
      const last = series.values[series.values.length - 1] ?? null
      return {
        instance: series.labels.instance || 'unknown',
        values: series.values,
        up: last === null ? null : last >= 1,
        availability: observed.length
          ? observed.filter((value) => value >= 1).length / observed.length
          : null,
      }
    })
    .sort((left, right) => left.instance.localeCompare(right.instance))
}

export type AvailabilityState = 'up' | 'partial' | 'down' | 'none'

export interface AvailabilityBucket {
  start: number
  end: number
  state: AvailabilityState
}

/** Groups scrape results into at most `buckets` spans of the window for a status strip. */
export function availabilityBuckets(
  values: ReadonlyArray<number | null>,
  times: readonly number[],
  buckets = 60,
): AvailabilityBucket[] {
  const count = Math.min(buckets, values.length)
  const size = values.length / count
  return Array.from({ length: count }, (_, index) => {
    const from = Math.floor(index * size)
    const to = Math.max(from + 1, Math.floor((index + 1) * size))
    const observed = values.slice(from, to).filter((value) => value !== null)
    const up = observed.filter((value) => value >= 1).length
    const state: AvailabilityState = !observed.length
      ? 'none'
      : up === observed.length
        ? 'up'
        : up === 0
          ? 'down'
          : 'partial'
    return { start: times[from], end: times[to - 1], state }
  })
}
