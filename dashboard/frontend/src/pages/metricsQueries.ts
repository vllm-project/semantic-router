export const METRICS_RANGES = ['1h', '6h', '24h', '7d'] as const
export type MetricsRange = (typeof METRICS_RANGES)[number]
export const DEFAULT_METRICS_RANGE: MetricsRange = '6h'

export const METRICS_RANGE_LABEL: Record<MetricsRange, string> = {
  '1h': 'last hour',
  '6h': 'last 6 hours',
  '24h': 'last 24 hours',
  '7d': 'last 7 days',
}

const RANGE_SECONDS: Record<MetricsRange, number> = {
  '1h': 3_600,
  '6h': 21_600,
  '24h': 86_400,
  '7d': 604_800,
}

export function readMetricsRange(value: string | null): MetricsRange {
  return METRICS_RANGES.find((range) => range === value) ?? DEFAULT_METRICS_RANGE
}

/** The Router's scrape job in every maintained Prometheus configuration and Grafana dashboard. */
export const ROUTER_JOB = 'semantic-router'

/** Scrape interval of the maintained Prometheus configurations. */
const SCRAPE_INTERVAL_SECONDS = 15
const STEPS_SECONDS = [15, 30, 60, 120, 300, 600, 900, 1_800, 3_600, 7_200]
const MAX_POINTS = 240

export interface MetricsWindow {
  /** Unix seconds of the first and last sample. */
  start: number
  end: number
  /** Seconds between samples. */
  step: number
  /** Range selector of the per-sample rates, e.g. `135s`. */
  rate: string
  /** Range selector covering the whole window, e.g. `21600s`. */
  range: string
  seconds: number
}

export function metricsWindow(range: MetricsRange, now = Date.now()): MetricsWindow {
  const seconds = RANGE_SECONDS[range]
  const step =
    STEPS_SECONDS.find((candidate) => seconds / candidate <= MAX_POINTS) ??
    STEPS_SECONDS[STEPS_SECONDS.length - 1]
  const end = Math.floor(now / 1_000)
  // Grafana's $__rate_interval: a step plus one scrape, and at least four scrapes,
  // so every rate sees two samples and adjacent windows overlap.
  const rate = Math.max(step + SCRAPE_INTERVAL_SECONDS, 4 * SCRAPE_INTERVAL_SECONDS)
  return { start: end - seconds, end, step, rate: `${rate}s`, range: `${seconds}s`, seconds }
}

function selector(labels?: string): string {
  return `{job="${ROUTER_JOB}"${labels ? `,${labels}` : ''}}`
}

const INFERENCE = 'traffic_kind="inference"'

/** Adds a constant label so several expressions can share one query with `or`. */
function tagged(expression: string, label: string, value: string): string {
  return `label_replace(${expression}, "${label}", "${value}", "", "")`
}

export const LATENCY_QUANTILES = [
  { key: 'p50', quantile: 0.5 },
  { key: 'p95', quantile: 0.95 },
  { key: 'p99', quantile: 0.99 },
] as const

function quantiles(buckets: string): string {
  return LATENCY_QUANTILES.map(({ key, quantile }) =>
    tagged(`histogram_quantile(${quantile}, sum by (le) (${buckets}))`, 'quantile', key),
  ).join(' or ')
}

function rates(window: string) {
  return (metric: string, labels?: string) => `rate(${metric}${selector(labels)}[${window}])`
}

function increases(range: string) {
  return (metric: string, labels?: string) => `increase(${metric}${selector(labels)}[${range}])`
}

function usage(over: (metric: string, labels?: string) => string) {
  return {
    // Counted once per client request, with a zero baseline from Router start.
    outcomes: `sum by (outcome) (${over('llm_request_outcomes_total', INFERENCE)})`,
    latency: quantiles(over('llm_request_duration_seconds_bucket', INFERENCE)),
    tokens: [
      tagged(`sum(${over('llm_model_prompt_tokens_total')})`, 'type', 'prompt'),
      tagged(`sum(${over('llm_model_completion_tokens_total')})`, 'type', 'completion'),
    ].join(' or '),
    cost: `sum by (currency) (${over('llm_model_cost_total')})`,
    cache: [
      tagged(`sum(${over('llm_cache_plugin_hits_total')})`, 'result', 'hit'),
      tagged(`sum(${over('llm_cache_plugin_misses_total')})`, 'result', 'miss'),
    ].join(' or '),
  }
}

export type UsageQueryKey = keyof ReturnType<typeof usage>

/** Range queries for the overview charts and router health, one sample per step. */
export function overviewSeriesQueries(window: MetricsWindow): Record<UsageQueryKey | 'up', string> {
  const series = usage(rates(window.rate))
  // Spend is easier to read per hour than per second.
  return { ...series, cost: `${series.cost} * 3600`, up: `up${selector()}` }
}

/** Instant queries totalling the whole window, for the summary figures. */
export function overviewSummaryQueries(window: MetricsWindow): Record<UsageQueryKey, string> {
  return usage(increases(window.range))
}

export const MODEL_METRICS = ['requests', 'errors', 'latency', 'tokens', 'cost'] as const
export type ModelMetric = (typeof MODEL_METRICS)[number]

function modelExpressions(over: (metric: string, labels?: string) => string) {
  return {
    requests: `sum by (model) (${over('llm_model_requests_total')})`,
    errors: `sum by (model) (${over('llm_request_errors_total')})`,
    latency: `histogram_quantile(0.95, sum by (model, le) (${over('llm_model_completion_latency_seconds_bucket')}))`,
    tokens: `sum by (model) (${over('llm_model_tokens_total')})`,
    cost: `sum by (model, currency) (${over('llm_model_cost_total')})`,
  } satisfies Record<ModelMetric, string>
}

/** One instant query for the per-model table: each series carries its `stat`. */
export function modelSummaryQuery(window: MetricsWindow): string {
  const expressions = modelExpressions(increases(window.range))
  return MODEL_METRICS.map((metric) => tagged(expressions[metric], 'stat', metric)).join(' or ')
}

export function modelSeriesQuery(metric: ModelMetric, window: MetricsWindow): string {
  const expression = modelExpressions(rates(window.rate))[metric]
  return metric === 'cost' ? `${expression} * 3600` : expression
}

export function decisionSummaryQueries(window: MetricsWindow) {
  const over = increases(window.range)
  return {
    // Successful model selections, by the decision that chose the model.
    selections: `sum by (recipe, decision, model) (${over('llm_recipe_selections_total')})`,
    cache: [
      tagged(`sum by (decision_name) (${over('llm_cache_plugin_hits_total')})`, 'result', 'hit'),
      tagged(`sum by (decision_name) (${over('llm_cache_plugin_misses_total')})`, 'result', 'miss'),
    ].join(' or '),
  }
}

export function decisionSeriesQuery(window: MetricsWindow): string {
  return `sum by (recipe, decision) (${rates(window.rate)('llm_recipe_selections_total')})`
}

const PROMETHEUS_API = '/embedded/prometheus/api/v1'

export function prometheusRangeUrl(query: string, window: MetricsWindow): string {
  const params = new URLSearchParams({
    query,
    start: String(window.start),
    end: String(window.end),
    step: String(window.step),
  })
  return `${PROMETHEUS_API}/query_range?${params}`
}

export function prometheusInstantUrl(query: string, time: number): string {
  return `${PROMETHEUS_API}/query?${new URLSearchParams({ query, time: String(time) })}`
}

const GRAFANA_DASHBOARD_UID = 'vllm-semantic-router'

/** The provisioned Router dashboard in Grafana, over the same range. */
export function grafanaDashboardUrl(range: MetricsRange, theme: string): string {
  const query = new URLSearchParams({ orgId: '1', theme, from: `now-${range}`, to: 'now' })
  return `/embedded/grafana/d/${encodeURIComponent(GRAFANA_DASHBOARD_UID)}?${query}`
}

export const GRAFANA_HEALTH_URL = '/embedded/grafana/api/health'
