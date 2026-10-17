export const DECISION_MODEL_METRICS = [
  {
    key: 'calls',
    label: 'Runtime calls / sec',
    unit: 'rate',
    color: '#7297f5',
    help: 'Calls observed by the router, averaged over 5 minutes.',
  },
  {
    key: 'errors',
    label: 'Unsuccessful calls',
    unit: 'percent',
    color: '#ee8b9b',
    help: 'Timeouts, unavailable, overloaded, rejected, and failed calls.',
  },
  {
    key: 'latency',
    label: 'Mean call latency',
    unit: 'seconds',
    color: '#7297f5',
    help: 'Router-observed latency for calls that reached the runtime.',
  },
  {
    key: 'p95',
    label: 'P95 call latency',
    unit: 'seconds',
    color: '#b19ae8',
    help: 'Estimated from the runtime call latency histogram.',
  },
  {
    key: 'cache',
    label: 'Result cache hit rate',
    unit: 'percent',
    color: '#49bba5',
    help: 'Hits among recorded result-cache lookups; unavailable when caching is not observed.',
  },
  {
    key: 'forward',
    label: 'Mean model forward',
    unit: 'seconds',
    color: '#dca965',
    help: 'Runtime-reported forward time per call. Calls in one batch share its timing.',
  },
] as const

export const DECISION_MODEL_TIME_WINDOWS = [
  { label: '15m', seconds: 900 },
  { label: '1h', seconds: 3_600 },
  { label: '6h', seconds: 21_600 },
] as const

export type DecisionModelMetricKey = (typeof DECISION_MODEL_METRICS)[number]['key']
export type DecisionModelTimeWindow = (typeof DECISION_MODEL_TIME_WINDOWS)[number]['seconds']
export type DecisionModelMetricSample = { time: number; value: number | null }
export type DecisionModelMetricSeries = Partial<
  Record<DecisionModelMetricKey, Record<string, DecisionModelMetricSample[]>>
>
export type DecisionModelChartPoint = { time: number } & Partial<
  Record<DecisionModelMetricKey, number | null>
>
export type DecisionModelMetricRange = { start: number; end: number; step: number }

export function formatDecisionModelRateAxis(value: number): string {
  // A quiet runtime can average far below one call per second. Fixed decimal
  // precision would turn every tick in that range into a misleading zero.
  return new Intl.NumberFormat(undefined, {
    notation: 'compact',
    maximumSignificantDigits: 3,
  }).format(value)
}

export function decisionModelMetricRange(
  window: DecisionModelTimeWindow,
  now = Date.now(),
): DecisionModelMetricRange {
  const end = Math.floor(now / 1_000)
  // At most 181 samples per series, including both ends of the window.
  return { start: end - window, end, step: Math.max(15, Math.ceil(window / 180)) }
}

function deploymentSelector(deployments: string[]): string {
  // Escape the RE2 pattern first, then its PromQL string literal. Deployment
  // names may contain regex metacharacters, quotes, backslashes or newlines.
  const pattern = deployments.map((name) => name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')
  return `deployment=~${JSON.stringify(pattern)}`
}

export function decisionModelMetricQueries(
  deployments: string[],
): Record<DecisionModelMetricKey, string> {
  const selector = deploymentSelector(deployments)
  const rate = (metric: string, extra = '') =>
    `sum by (deployment) (rate(${metric}{${selector}${extra}}[5m]))`
  const calls = rate('vsr_model_runtime_requests_total')
  const cache = rate('vsr_model_runtime_result_cache_total')
  return {
    calls,
    // A missing outcome is zero only when the deployment has observed calls.
    // With no traffic the denominator is zero, yielding NaN (unavailable).
    errors: `((${rate('vsr_model_runtime_requests_total', ',outcome!="ok"')} or (${calls} * 0)) / ${calls}) * 100`,
    latency: `${rate('vsr_model_runtime_request_duration_seconds_sum')} / ${rate('vsr_model_runtime_request_duration_seconds_count')}`,
    p95: `histogram_quantile(0.95, sum by (deployment, le) (rate(vsr_model_runtime_request_duration_seconds_bucket{${selector}}[5m])))`,
    cache: `((${rate('vsr_model_runtime_result_cache_total', ',result="hit"')} or (${cache} * 0)) / ${cache}) * 100`,
    forward: `${rate('vsr_model_runtime_server_seconds_sum', ',phase="forward"')} / ${rate('vsr_model_runtime_server_seconds_count', ',phase="forward"')}`,
  }
}

export function readDecisionModelMetricMatrix(
  data: unknown,
  deployments: string[],
  range: DecisionModelMetricRange,
): Record<string, DecisionModelMetricSample[]> {
  const response = data as {
    status?: string
    data?: {
      resultType?: string
      result?: Array<{ metric?: { deployment?: string }; values?: unknown[] }>
    }
  } | null
  if (
    response?.status !== 'success' ||
    response.data?.resultType !== 'matrix' ||
    !Array.isArray(response.data.result)
  ) {
    throw new Error('Prometheus did not return model metric history.')
  }
  const allowed = new Set(deployments)
  const series: Record<string, DecisionModelMetricSample[]> = {}
  for (const result of response.data.result) {
    const deployment = result?.metric?.deployment
    if (!deployment || !allowed.has(deployment) || !Array.isArray(result.values)) continue
    const samples = new Map<number, number | null>()
    for (const sample of result.values) {
      if (!Array.isArray(sample)) continue
      const [timestamp, raw] = sample
      if (
        typeof timestamp !== 'number' ||
        !Number.isFinite(timestamp) ||
        timestamp < range.start ||
        timestamp > range.end ||
        (timestamp - range.start) % range.step !== 0
      )
        continue
      const value = typeof raw === 'string' && raw.trim() ? Number(raw) : NaN
      samples.set(timestamp, Number.isFinite(value) && value >= 0 ? value : null)
    }
    // Retain gaps explicitly. A missing or NaN last observation must not reuse
    // an earlier healthy value or draw a line through an outage.
    const points: DecisionModelMetricSample[] = []
    for (let time = range.start; time <= range.end; time += range.step) {
      points.push({ time: time * 1_000, value: samples.get(time) ?? null })
    }
    series[deployment] = points
  }
  return series
}

export function decisionModelChartPoints(
  series: DecisionModelMetricSeries,
  deployment: string,
): DecisionModelChartPoint[] {
  const points = new Map<number, DecisionModelChartPoint>()
  for (const metric of DECISION_MODEL_METRICS) {
    for (const sample of series[metric.key]?.[deployment] ?? []) {
      const point = points.get(sample.time) ?? { time: sample.time }
      point[metric.key] = sample.value
      points.set(sample.time, point)
    }
  }
  return [...points.values()].sort((a, b) => a.time - b.time)
}

export function formatDecisionModelMetric(
  value: number | null | undefined,
  unit: 'rate' | 'percent' | 'seconds',
): string {
  if (value == null || !Number.isFinite(value)) return 'Not reported'
  if (unit === 'percent') return `${value.toFixed(1)}%`
  if (unit === 'seconds')
    return value < 1 ? `${(value * 1000).toFixed(1)} ms` : `${value.toFixed(2)} s`
  return value.toFixed(2)
}
