import type { MetricsWindow } from './metricsQueries'

export type Labels = Record<string, string>
/** One series of a range query, a value or null (no sample) at each step of the window. */
export interface MetricSeries {
  labels: Labels
  values: (number | null)[]
}
/** One series of an instant query. */
export interface MetricSample {
  labels: Labels
  value: number | null
}

type Grid = Pick<MetricsWindow, 'start' | 'end' | 'step'>

export function sampleCount(grid: Grid): number {
  return Math.floor((grid.end - grid.start) / grid.step) + 1
}

/** Milliseconds of each step of the window. */
export function sampleTimes(grid: Grid): number[] {
  return Array.from(
    { length: sampleCount(grid) },
    (_, index) => (grid.start + index * grid.step) * 1_000,
  )
}

function readValue(raw: unknown): number | null {
  if (typeof raw !== 'string' || !raw.trim()) return null
  const value = Number(raw)
  return Number.isFinite(value) ? value : null
}

function readLabels(metric: unknown): Labels {
  const labels: Labels = {}
  if (!metric || typeof metric !== 'object') return labels
  for (const [key, value] of Object.entries(metric)) {
    if (typeof value === 'string') labels[key] = value
  }
  return labels
}

function readResult(data: unknown, resultType: 'matrix' | 'vector'): unknown[] {
  const response = data as {
    status?: unknown
    data?: { resultType?: unknown; result?: unknown }
  } | null
  if (
    response?.status !== 'success' ||
    response.data?.resultType !== resultType ||
    !Array.isArray(response.data.result)
  ) {
    throw new Error(
      resultType === 'matrix'
        ? 'Prometheus did not return a time series.'
        : 'Prometheus did not return a result.',
    )
  }
  return response.data.result
}

/**
 * Places a range query's samples on the window's grid. A step without a sample,
 * and a NaN or infinite one (a ratio with nothing to divide), stays null: a gap.
 */
export function readMatrix(data: unknown, grid: Grid): MetricSeries[] {
  const count = sampleCount(grid)
  return readResult(data, 'matrix').flatMap((entry) => {
    const { metric, values } = (entry ?? {}) as { metric?: unknown; values?: unknown }
    if (!Array.isArray(values)) return []
    const points: (number | null)[] = Array.from({ length: count }, () => null)
    for (const sample of values) {
      if (!Array.isArray(sample) || typeof sample[0] !== 'number') continue
      const index = Math.round((sample[0] - grid.start) / grid.step)
      if (index < 0 || index >= count) continue
      if (Math.abs(grid.start + index * grid.step - sample[0]) > 0.001) continue
      points[index] = readValue(sample[1])
    }
    return [{ labels: readLabels(metric), values: points }]
  })
}

export function readVector(data: unknown): MetricSample[] {
  return readResult(data, 'vector').flatMap((entry) => {
    const { metric, value } = (entry ?? {}) as { metric?: unknown; value?: unknown }
    if (!Array.isArray(value)) return []
    return [{ labels: readLabels(metric), value: readValue(value[1]) }]
  })
}

/** Adds series point by point. A point is null only where every series lacks it. */
export function sumValues(
  series: ReadonlyArray<ReadonlyArray<number | null>>,
  count: number,
): (number | null)[] {
  return Array.from({ length: count }, (_, index) => {
    let total: number | null = null
    for (const values of series) {
      const value = values[index]
      if (value !== null && value !== undefined) total = (total ?? 0) + value
    }
    return total
  })
}

/** `part / whole` per point, null where the whole is missing or zero. A missing part is zero. */
export function ratioValues(
  part: ReadonlyArray<number | null>,
  whole: ReadonlyArray<number | null>,
): (number | null)[] {
  return whole.map((total, index) =>
    total === null || total <= 0 ? null : (part[index] ?? 0) / total,
  )
}

export function valuesWhere(
  series: readonly MetricSeries[],
  label: string,
  value: string,
  count: number,
): (number | null)[] {
  return sumValues(
    series.filter((entry) => entry.labels[label] === value).map((entry) => entry.values),
    count,
  )
}

/** The sum of the samples, or null when the query returned none. */
export function totalOf(samples: readonly MetricSample[]): number | null {
  const values = samples.map((sample) => sample.value).filter((value) => value !== null)
  return values.length ? values.reduce((total, value) => total + value, 0) : null
}

export function hasSamples(values: ReadonlyArray<number | null>): boolean {
  return values.some((value) => value !== null)
}

export interface ChartSeriesValues {
  key: string
  values: ReadonlyArray<number | null>
}

/**
 * Rows for a recharts chart. Keys are positional, never label text: recharts
 * reads a data key with a dot (a model such as `qwen2.5`) as a nested path.
 */
export function chartRows(
  times: readonly number[],
  series: readonly ChartSeriesValues[],
): Array<Record<string, number | null>> {
  return times.map((time, index) => {
    const row: Record<string, number | null> = { time }
    for (const entry of series) row[entry.key] = entry.values[index] ?? null
    return row
  })
}

/** Series colors in rank order. Rose is left out: it marks failures. */
export const RANKED_COLORS = [
  'var(--chart-1)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-2)',
  'var(--chart-6)',
  'var(--chart-7)',
]
export const OTHER_COLOR = 'var(--chart-8)'

interface SeriesEntry {
  name: string
  label?: string
  values: (number | null)[]
}

export interface RankedSeries<T extends SeriesEntry> {
  label: string
  color: string
  values: (number | null)[]
  /** The charted entry; absent for the sum of the others. */
  source?: T
}

const seriesTotal = (values: ReadonlyArray<number | null>) =>
  values.reduce<number>((total, value) => total + (value ?? 0), 0)

/**
 * Colors the first names of `order` (or, without an order, the largest series)
 * by rank, so a name keeps its color across metrics and in a table that uses
 * the same order. Every other series is added into one "others" series when
 * the metric is additive, and left out when it is not.
 */
export function rankedSeries<T extends SeriesEntry>(
  series: readonly T[],
  order: readonly string[],
  count: number,
  additive: boolean,
): RankedSeries<T>[] {
  const ranking = order.length
    ? order
    : [...series]
        .sort((left, right) => seriesTotal(right.values) - seriesTotal(left.values))
        .map((entry) => entry.name)
  const top = ranking.slice(0, RANKED_COLORS.length)
  const shown = series
    .filter((entry) => top.includes(entry.name))
    .sort((left, right) => top.indexOf(left.name) - top.indexOf(right.name))
    .map((entry) => ({
      label: entry.label ?? entry.name,
      color: RANKED_COLORS[top.indexOf(entry.name)],
      values: entry.values,
      source: entry,
    }))
  const rest = series.filter((entry) => !top.includes(entry.name))
  if (!additive || !rest.length) return shown
  const names = new Set(rest.map((entry) => entry.name)).size
  return [
    ...shown,
    {
      label: `${names} other${names === 1 ? '' : 's'}`,
      color: OTHER_COLOR,
      values: sumValues(
        rest.map((entry) => entry.values),
        count,
      ),
    },
  ]
}

/** The color a name gets in {@link rankedSeries} for the same order. */
export function rankedColor(order: readonly string[], name: string): string {
  const index = order.indexOf(name)
  return index >= 0 && index < RANKED_COLORS.length ? RANKED_COLORS[index] : OTHER_COLOR
}

const MINUTE = 60_000
const HOUR = 60 * MINUTE
const DAY = 24 * HOUR
const TICK_STEPS = [
  5 * MINUTE,
  10 * MINUTE,
  15 * MINUTE,
  30 * MINUTE,
  HOUR,
  2 * HOUR,
  3 * HOUR,
  6 * HOUR,
  12 * HOUR,
  DAY,
]

/** Round local times for a time axis: at most about `target` ticks between start and end. */
export function timeTicks(start: number, end: number, target = 6): number[] {
  const span = Math.max(0, end - start)
  const step = TICK_STEPS.find((candidate) => span / candidate <= target) ?? DAY
  const cursor = new Date(start)
  if (step >= DAY) cursor.setHours(0, 0, 0, 0)
  else if (step >= HOUR) cursor.setMinutes(0, 0, 0)
  else cursor.setSeconds(0, 0)
  const ticks: number[] = []
  while (cursor.getTime() <= end) {
    const aligned =
      step >= DAY ||
      (step >= HOUR
        ? cursor.getHours() % (step / HOUR) === 0
        : cursor.getMinutes() % (step / MINUTE) === 0)
    if (aligned && cursor.getTime() >= start) ticks.push(cursor.getTime())
    if (step >= DAY) cursor.setDate(cursor.getDate() + 1)
    else if (step >= HOUR) cursor.setHours(cursor.getHours() + 1)
    else cursor.setMinutes(cursor.getMinutes() + 1)
  }
  return ticks
}
