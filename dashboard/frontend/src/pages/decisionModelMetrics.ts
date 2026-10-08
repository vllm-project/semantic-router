export const DECISION_MODEL_METRICS = [
  {
    key: 'calls',
    label: 'Runtime calls / sec',
    unit: 'rate',
    help: 'Calls observed by the router, averaged over 5 minutes.',
  },
  {
    key: 'errors',
    label: 'Unsuccessful calls',
    unit: 'percent',
    help: 'Timeouts, unavailable, overloaded, rejected, and failed calls.',
  },
  {
    key: 'latency',
    label: 'Mean call latency',
    unit: 'seconds',
    help: 'Router-observed latency for calls that reached the runtime.',
  },
  {
    key: 'p95',
    label: 'P95 call latency',
    unit: 'seconds',
    help: 'Estimated from the runtime call latency histogram.',
  },
  {
    key: 'cache',
    label: 'Result cache hit rate',
    unit: 'percent',
    help: 'Hits among recorded result-cache lookups; unavailable when caching is not observed.',
  },
  {
    key: 'forward',
    label: 'Mean model forward',
    unit: 'seconds',
    help: 'Runtime-reported forward time per call. Calls in one batch share its timing.',
  },
] as const

export type DecisionModelMetricKey = (typeof DECISION_MODEL_METRICS)[number]['key']
export type DecisionModelMetricValues = Partial<
  Record<DecisionModelMetricKey, Record<string, number>>
>

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

export function readDecisionModelMetricVector(
  data: unknown,
  deployments: string[],
): Record<string, number> {
  const response = data as {
    status?: string
    data?: {
      resultType?: string
      result?: Array<{ metric?: { deployment?: string }; value?: unknown[] }>
    }
  } | null
  if (
    response?.status !== 'success' ||
    response.data?.resultType !== 'vector' ||
    !Array.isArray(response.data.result)
  ) {
    throw new Error('Prometheus did not return model metrics.')
  }
  const allowed = new Set(deployments)
  const values: Record<string, number> = {}
  for (const sample of response.data.result) {
    const deployment = sample?.metric?.deployment
    const raw = sample?.value?.[1]
    if (!deployment || !allowed.has(deployment) || typeof raw !== 'string' || !raw.trim()) continue
    const value = Number(raw)
    if (Number.isFinite(value) && value >= 0) values[deployment] = value
  }
  return values
}

export function formatDecisionModelMetric(
  value: number | undefined,
  unit: 'rate' | 'percent' | 'seconds',
): string {
  if (value === undefined || !Number.isFinite(value)) return 'Not reported'
  if (unit === 'percent') return `${value.toFixed(1)}%`
  if (unit === 'seconds')
    return value < 1 ? `${(value * 1000).toFixed(1)} ms` : `${value.toFixed(2)} s`
  return value.toFixed(2)
}
