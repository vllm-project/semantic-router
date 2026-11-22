import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import {
  DEFAULT_METRICS_RANGE,
  METRICS_RANGES,
  MODEL_METRICS,
  decisionSeriesQuery,
  decisionSummaryQueries,
  grafanaDashboardUrl,
  metricsWindow,
  modelSeriesQuery,
  modelSummaryQuery,
  overviewSeriesQueries,
  overviewSummaryQueries,
  prometheusInstantUrl,
  prometheusRangeUrl,
  readMetricsRange,
} from './metricsQueries'

const NOW = Date.UTC(2026, 10, 18, 2, 42, 7, 512)

function balanced(query: string): boolean {
  let depth = 0
  for (const character of query.replace(/"(?:\\.|[^"\\])*"/g, '""')) {
    if ('([{'.includes(character)) depth += 1
    if (')]}'.includes(character)) depth -= 1
    if (depth < 0) return false
  }
  return depth === 0
}

function everyQuery() {
  return METRICS_RANGES.flatMap((range) => {
    const window = metricsWindow(range, NOW)
    return [
      ...Object.values(overviewSeriesQueries(window)),
      ...Object.values(overviewSummaryQueries(window)),
      ...Object.values(decisionSummaryQueries(window)),
      decisionSeriesQuery(window),
      modelSummaryQuery(window),
      ...MODEL_METRICS.map((metric) => modelSeriesQuery(metric, window)),
    ]
  })
}

describe('metrics window', () => {
  it('reads the range from the URL and falls back to six hours', () => {
    expect(readMetricsRange('24h')).toBe('24h')
    expect(readMetricsRange('bogus')).toBe(DEFAULT_METRICS_RANGE)
    expect(readMetricsRange(null)).toBe('6h')
  })

  it.each([
    ['1h', 15, '60s'],
    ['6h', 120, '135s'],
    ['24h', 600, '615s'],
    ['7d', 3_600, '3615s'],
  ] as const)('derives the %s step and rate window from the range', (range, step, rate) => {
    const window = metricsWindow(range, NOW)
    expect(window.step).toBe(step)
    expect(window.rate).toBe(rate)
    expect(window.end).toBe(Math.floor(NOW / 1_000))
    expect(window.end - window.start).toBe(window.seconds)
    expect(window.range).toBe(`${window.seconds}s`)
    expect(window.seconds / window.step).toBeLessThanOrEqual(240)
  })
})

describe('Router PromQL', () => {
  it('reads only the Router scrape job and keeps every query well formed', () => {
    const queries = everyQuery()
    for (const query of queries) {
      expect(balanced(query), query).toBe(true)
      const selectors = query.match(/[a-z_]+\{[^}]*\}/g) ?? []
      expect(selectors.length, query).toBeGreaterThan(0)
      for (const selector of selectors) expect(selector, query).toContain('job="semantic-router"')
    }
  })

  it('charts inference outcomes, latency quantiles, tokens, spend and cache lookups per step', () => {
    const window = metricsWindow('6h', NOW)
    const queries = overviewSeriesQueries(window)
    expect(queries.outcomes).toBe(
      'sum by (outcome) (rate(llm_request_outcomes_total{job="semantic-router",traffic_kind="inference"}[135s]))',
    )
    for (const [key, quantile] of [
      ['p50', '0.5'],
      ['p95', '0.95'],
      ['p99', '0.99'],
    ]) {
      expect(queries.latency).toContain(
        `label_replace(histogram_quantile(${quantile}, sum by (le) (rate(llm_request_duration_seconds_bucket{job="semantic-router",traffic_kind="inference"}[135s]))), "quantile", "${key}", "", "")`,
      )
    }
    expect(queries.tokens).toContain('"type", "prompt"')
    expect(queries.tokens).toContain('llm_model_completion_tokens_total')
    expect(queries.cost).toBe(
      'sum by (currency) (rate(llm_model_cost_total{job="semantic-router"}[135s])) * 3600',
    )
    expect(queries.cache).toContain('llm_cache_plugin_hits_total')
    expect(queries.cache).toContain('"result", "miss"')
    expect(queries.up).toBe('up{job="semantic-router"}')
  })

  it('totals the whole range for the summary figures', () => {
    const window = metricsWindow('24h', NOW)
    const queries = overviewSummaryQueries(window)
    expect(queries.outcomes).toContain('increase(llm_request_outcomes_total')
    expect(queries.outcomes).toContain('[86400s]')
    expect(queries.latency).toContain('increase(llm_request_duration_seconds_bucket')
    expect(queries.cost).toBe(
      'sum by (currency) (increase(llm_model_cost_total{job="semantic-router"}[86400s]))',
    )
    expect(Object.values(queries).every((query) => !query.includes('rate('))).toBe(true)
  })

  it('tags each per-model statistic and breaks spend down by currency', () => {
    const window = metricsWindow('1h', NOW)
    const summary = modelSummaryQuery(window)
    for (const metric of MODEL_METRICS) expect(summary).toContain(`"stat", "${metric}"`)
    expect(summary).toContain(
      'histogram_quantile(0.95, sum by (model, le) (increase(llm_model_completion_latency_seconds_bucket{job="semantic-router"}[3600s])))',
    )
    expect(modelSeriesQuery('requests', window)).toBe(
      'sum by (model) (rate(llm_model_requests_total{job="semantic-router"}[60s]))',
    )
    expect(modelSeriesQuery('errors', window)).toContain('llm_request_errors_total')
    expect(modelSeriesQuery('cost', window)).toBe(
      'sum by (model, currency) (rate(llm_model_cost_total{job="semantic-router"}[60s])) * 3600',
    )
  })

  it('reads decisions from model selections and the per-decision cache counters', () => {
    const window = metricsWindow('6h', NOW)
    const summary = decisionSummaryQueries(window)
    expect(summary.selections).toBe(
      'sum by (recipe, decision, model) (increase(llm_recipe_selections_total{job="semantic-router"}[21600s]))',
    )
    expect(summary.cache).toContain('sum by (decision_name) (increase(llm_cache_plugin_hits_total')
    expect(decisionSeriesQuery(window)).toBe(
      'sum by (recipe, decision) (rate(llm_recipe_selections_total{job="semantic-router"}[135s]))',
    )
  })

  it('queries Prometheus through the Dashboard proxy', () => {
    const window = metricsWindow('1h', NOW)
    const range = new URL(prometheusRangeUrl('up{job="semantic-router"}', window), 'http://x')
    expect(range.pathname).toBe('/embedded/prometheus/api/v1/query_range')
    expect(range.searchParams.get('query')).toBe('up{job="semantic-router"}')
    expect(Number(range.searchParams.get('start'))).toBe(window.start)
    expect(Number(range.searchParams.get('end'))).toBe(window.end)
    expect(range.searchParams.get('step')).toBe('15')
    const instant = new URL(prometheusInstantUrl('up', window.end), 'http://x')
    expect(instant.pathname).toBe('/embedded/prometheus/api/v1/query')
    expect(Number(instant.searchParams.get('time'))).toBe(window.end)
  })
})

describe('Grafana link', () => {
  const provisioned = JSON.parse(
    readFileSync(
      new URL(
        '../../../../src/vllm-sr/cli/templates/llm-router-dashboard.serve.json',
        import.meta.url,
      ),
      'utf8',
    ),
  ) as { uid: string }

  it.each(['dark', 'light'])('opens the provisioned Router dashboard in %s mode', (theme) => {
    const target = new URL(grafanaDashboardUrl('24h', theme), 'https://dashboard.example.test')
    expect(provisioned.uid).toBeTruthy()
    expect(target.pathname).toBe(`/embedded/grafana/d/${encodeURIComponent(provisioned.uid)}`)
    expect(target.searchParams.get('theme')).toBe(theme)
    expect(target.searchParams.get('orgId')).toBe('1')
    expect(target.searchParams.get('from')).toBe('now-24h')
    expect(target.searchParams.get('to')).toBe('now')
  })
})
