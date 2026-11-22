import { readFileSync } from 'node:fs'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '../ui'
import { DOCS_LINKS } from '../utils/docsLinks'
import MonitoringPage from './MonitoringPage'
import { metricsWindow } from './metricsQueries'
import type { MetricSample, MetricSeries } from './metricsSeries'
import { ObservabilityError } from './observabilityProxy'
import type { MetricsDashboardData, QueryResult } from './useMetricsDashboard'

const state = vi.hoisted(() => ({
  range: '',
  grafana: false,
  dashboard: {
    data: null as unknown,
    loading: false,
    updatedAt: null as number | null,
    refreshError: null as unknown,
    refresh: () => undefined,
  },
}))

vi.mock('../shell/theme', () => ({ useTheme: () => ({ resolved: 'dark' }) }))
vi.mock('./useMetricsDashboard', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./useMetricsDashboard')>()),
  useMetricsDashboard: (range: string) => {
    state.range = range
    return state.dashboard
  },
  useGrafanaAvailable: () => state.grafana,
  usePrometheusRange: () => ({ result: null, loading: true }),
}))

const NOW = Date.UTC(2026, 10, 18, 2, 0, 0)
const ok = <T,>(data: T): QueryResult<T> => ({ data })
const flat = (value: number) => Array.from({ length: 181 }, () => value)
const series = (labels: Record<string, string>, value: number): MetricSeries => ({
  labels,
  values: flat(value),
})
const sample = (labels: Record<string, string>, value: number): MetricSample => ({ labels, value })

function dashboardData(
  change: (data: MetricsDashboardData) => void = () => undefined,
): MetricsDashboardData {
  const data: MetricsDashboardData = {
    window: metricsWindow('6h', NOW),
    series: {
      outcomes: ok([series({ outcome: 'success' }, 2), series({ outcome: 'server_error' }, 0.1)]),
      latency: ok([series({ quantile: 'p95' }, 1.8)]),
      tokens: ok([]),
      cost: ok([]),
      cache: ok([]),
      up: ok([series({ instance: 'router:9190' }, 1)]),
      decisions: ok([]),
    },
    summary: {
      outcomes: ok([sample({ outcome: 'success' }, 950), sample({ outcome: 'server_error' }, 50)]),
      latency: ok([
        sample({ quantile: 'p50' }, 0.42),
        sample({ quantile: 'p95' }, 1.8),
        sample({ quantile: 'p99' }, 4.2),
      ]),
      tokens: ok([]),
      cost: ok([]),
      cache: ok([]),
      models: ok([
        sample({ model: 'qwen3-8b', stat: 'requests' }, 800),
        sample({ model: 'qwen3-8b', stat: 'errors' }, 8),
      ]),
      decisions: ok([sample({ recipe: 'default', decision: 'coding', model: 'qwen3-8b' }, 600)]),
      decisionCache: ok([]),
    },
  }
  change(data)
  return data
}

function render(path = '/observe/metrics') {
  return renderToStaticMarkup(
    <TooltipProvider>
      <MemoryRouter initialEntries={[path]}>
        <MonitoringPage />
      </MemoryRouter>
    </TooltipProvider>,
  )
}

beforeEach(() => {
  state.grafana = false
  state.dashboard = {
    data: null,
    loading: false,
    updatedAt: null,
    refreshError: null,
    refresh: () => undefined,
  }
})

describe('Metrics page', () => {
  it('loads the range from the URL and falls back to six hours', () => {
    state.dashboard.loading = true
    const markup = render('/observe/metrics?range=24h')
    expect(state.range).toBe('24h')
    expect(markup).toContain('Loading metrics')
    expect(markup).toContain('aria-label="Time range"')
    render('/observe/metrics?range=bogus')
    expect(state.range).toBe('6h')
  })

  it('asks for a Prometheus connection when the Dashboard has none', () => {
    state.dashboard.data = dashboardData((data) => {
      data.series.up = {
        error: new ObservabilityError('unconfigured', 'Prometheus is not connected.', 503),
      }
    })
    const markup = render()
    expect(markup).toContain('Connect Prometheus')
    expect(markup).toContain('TARGET_PROMETHEUS_URL')
    expect(markup).toContain(`href="${DOCS_LINKS.observability.replace(/&/g, '&amp;')}"`)
    expect(markup).not.toContain('aria-label="Time range"')
    expect(markup).not.toContain('Refresh')
  })

  it('explains a denied read and offers a retry instead of metrics', () => {
    state.dashboard.data = dashboardData((data) => {
      data.series.up = {
        error: new ObservabilityError(
          'denied',
          'Access to Prometheus was denied. Check your session and permissions, then try again.',
          403,
        ),
      }
    })
    const markup = render()
    expect(markup).toContain('role="alert"')
    expect(markup).toContain('Access to Prometheus was denied')
    expect(markup).toContain('Try again')
    expect(markup).not.toContain('By model')
  })

  it('summarizes traffic and breaks it down by model, decision and scrape target', () => {
    state.dashboard.data = dashboardData()
    state.dashboard.updatedAt = NOW
    const markup = render()
    expect(markup).toContain('Updated')
    expect(markup).toContain('2.78/min')
    expect(markup).toContain('1K in the last 6 hours')
    expect(markup).toContain('>5%<')
    expect(markup).toContain('50 of 1K failed')
    expect(markup).toContain('1.80 s')
    expect(markup).toContain('p50 420 ms · p99 4.20 s')
    expect(markup).toContain('No token usage reported')
    expect(markup).not.toContain('>Spend<')
    expect(markup).toContain('By model')
    expect(markup).toContain('qwen3-8b')
    expect(markup).toContain('By decision')
    expect(markup).toContain('coding')
    expect(markup).toContain('router:9190')
    expect(markup).toContain('100% of scrapes succeeded')
    expect(markup).not.toContain('role="alert"')
    expect(markup).not.toContain('<iframe')
  })

  it('says why the charts are empty when Prometheus does not scrape the Router', () => {
    state.dashboard.data = dashboardData((data) => {
      data.series.up = ok([])
    })
    const markup = render()
    expect(markup).toContain('Prometheus does not scrape the Router')
    expect(markup).toContain('semantic-router')
  })

  it('warns when every Router target failed its last scrape', () => {
    state.dashboard.data = dashboardData((data) => {
      data.series.up = ok([series({ instance: 'router:9190' }, 0)])
    })
    expect(render()).toContain('Prometheus cannot read the Router&#x27;s metrics')
  })

  it('links to the provisioned Grafana dashboard only when Grafana answers', () => {
    state.dashboard.data = dashboardData()
    expect(render()).not.toContain('Open Grafana')
    state.grafana = true
    const markup = render('/observe/metrics?range=7d')
    const href = markup.match(/href="(\/embedded\/grafana\/[^"]+)"/)?.[1].replace(/&amp;/g, '&')
    const target = new URL(href ?? '', 'https://dashboard.example.test')
    expect(target.pathname).toBe('/embedded/grafana/d/vllm-semantic-router')
    expect(target.searchParams.get('from')).toBe('now-7d')
    expect(target.searchParams.get('theme')).toBe('dark')
    expect(markup).toContain('Open Grafana')
  })
})

describe('observability pages', () => {
  const sources = ['./MonitoringPage.tsx', './TracingPage.tsx'].map((path) =>
    readFileSync(new URL(path, import.meta.url), 'utf8'),
  )

  it('draw their own views instead of embedding Grafana or Jaeger', () => {
    for (const source of sources) {
      expect(source).not.toContain('<iframe')
      expect(source).not.toContain('contentWindow')
    }
  })

  it('point their setup docs at a live docs page', () => {
    // Regression guard for #3112: the Metrics and Traces setup states once
    // linked to an archived docs page that now returns 404.
    const deletedDocsPath = '/tutorials/observability/dashboard'
    expect(DOCS_LINKS.observability).toMatch(/^https:\/\/vllm-sr\.ai\/docs\//)
    expect(DOCS_LINKS.observability).not.toContain(deletedDocsPath)
    expect(DOCS_LINKS.observability).toContain('/tutorials/global/api-and-observability')
    for (const source of sources) {
      expect(source).toContain('DOCS_LINKS.observability')
      expect(source).not.toContain(deletedDocsPath)
    }
  })
})
