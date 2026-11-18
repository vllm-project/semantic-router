import { useCallback, useEffect, useRef, useState } from 'react'

import {
  GRAFANA_HEALTH_URL,
  decisionSeriesQuery,
  decisionSummaryQueries,
  metricsWindow,
  modelSummaryQuery,
  overviewSeriesQueries,
  overviewSummaryQueries,
  prometheusInstantUrl,
  prometheusRangeUrl,
  type MetricsRange,
  type MetricsWindow,
} from './metricsQueries'
import { readMatrix, readVector, type MetricSeries } from './metricsSeries'
import { ObservabilityError, fetchObservabilityJson, isAbortError } from './observabilityProxy'

export type QueryResult<T> =
  | { data: T; error?: undefined }
  | { data?: undefined; error: ObservabilityError }

interface LoadOptions {
  signal?: AbortSignal
  fetcher?: typeof fetch
}

async function settle<T>(load: () => Promise<T>): Promise<QueryResult<T>> {
  try {
    return { data: await load() }
  } catch (error) {
    if (isAbortError(error)) throw error
    return {
      error:
        error instanceof ObservabilityError
          ? error
          : new ObservabilityError(
              'query',
              error instanceof Error ? error.message : 'The query failed.',
            ),
    }
  }
}

export function rangeQuery(query: string, window: MetricsWindow, options: LoadOptions = {}) {
  return settle(async () =>
    readMatrix(
      await fetchObservabilityJson('Prometheus', prometheusRangeUrl(query, window), options),
      window,
    ),
  )
}

function instantQuery(query: string, window: MetricsWindow, options: LoadOptions) {
  return settle(async () =>
    readVector(
      await fetchObservabilityJson('Prometheus', prometheusInstantUrl(query, window.end), options),
    ),
  )
}

async function settleAll<K extends string, T>(
  queries: Record<K, string>,
  run: (query: string) => Promise<QueryResult<T>>,
): Promise<Record<K, QueryResult<T>>> {
  const keys = Object.keys(queries) as K[]
  const results = await Promise.all(keys.map((key) => run(queries[key])))
  return Object.fromEntries(keys.map((key, index) => [key, results[index]])) as Record<
    K,
    QueryResult<T>
  >
}

export async function loadMetricsDashboard(window: MetricsWindow, options: LoadOptions = {}) {
  const decisions = decisionSummaryQueries(window)
  const [series, summary] = await Promise.all([
    settleAll(
      { ...overviewSeriesQueries(window), decisions: decisionSeriesQuery(window) },
      (query) => rangeQuery(query, window, options),
    ),
    settleAll(
      {
        ...overviewSummaryQueries(window),
        models: modelSummaryQuery(window),
        decisions: decisions.selections,
        decisionCache: decisions.cache,
      },
      (query) => instantQuery(query, window, options),
    ),
  ])
  return { window, series, summary }
}

export type MetricsDashboardData = Awaited<ReturnType<typeof loadMetricsDashboard>>

const PAGE_FAILURES = new Set(['unconfigured', 'denied', 'unreachable', 'timeout'])

/** What keeps the whole page from showing metrics, judged by the scrape-health query. */
export function dashboardFailure(data: MetricsDashboardData): ObservabilityError | null {
  const error = data.series.up.error
  return error && PAGE_FAILURES.has(error.failure) ? error : null
}

const REFRESH_MS = 60_000

interface DashboardState {
  range: MetricsRange
  data: MetricsDashboardData | null
  loading: boolean
  updatedAt: number | null
  /** A refresh that failed while the last metrics stay on screen. */
  refreshError: ObservabilityError | null
}

export function useMetricsDashboard(range: MetricsRange) {
  const [state, setState] = useState<DashboardState>(() => ({
    range,
    data: null,
    loading: true,
    updatedAt: null,
    refreshError: null,
  }))
  const [refreshKey, setRefreshKey] = useState(0)
  const loadedAt = useRef(0)

  useEffect(() => {
    const controller = new AbortController()
    setState((current) =>
      current.range === range
        ? { ...current, loading: true }
        : { range, data: null, loading: true, updatedAt: null, refreshError: null },
    )
    loadMetricsDashboard(metricsWindow(range), { signal: controller.signal }).then(
      (data) => {
        loadedAt.current = Date.now()
        setState((current) => {
          const failure = dashboardFailure(data)
          const keepLast =
            (failure?.failure === 'unreachable' || failure?.failure === 'timeout') &&
            current.range === range &&
            current.data !== null &&
            !dashboardFailure(current.data)
          return keepLast
            ? { ...current, loading: false, refreshError: failure }
            : { range, data, loading: false, updatedAt: Date.now(), refreshError: null }
        })
      },
      (error: unknown) => {
        if (!isAbortError(error)) setState((current) => ({ ...current, loading: false }))
      },
    )
    return () => controller.abort()
  }, [range, refreshKey])

  useEffect(() => {
    const refresh = () => {
      if (!document.hidden) setRefreshKey((key) => key + 1)
    }
    const refreshIfStale = () => {
      if (Date.now() - loadedAt.current >= REFRESH_MS) refresh()
    }
    const timer = window.setInterval(refresh, REFRESH_MS)
    document.addEventListener('visibilitychange', refreshIfStale)
    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', refreshIfStale)
    }
  }, [])

  const refresh = useCallback(() => setRefreshKey((key) => key + 1), [])
  return { ...state, refresh }
}

/** Whether Grafana answers through the Dashboard proxy, so a link to its dashboard can work. */
export function useGrafanaAvailable() {
  const [available, setAvailable] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    fetchObservabilityJson('Grafana', GRAFANA_HEALTH_URL, {
      signal: controller.signal,
      timeoutMs: 5_000,
    }).then(
      () => setAvailable(true),
      () => {
        if (!controller.signal.aborted) setAvailable(false)
      },
    )
    return () => controller.abort()
  }, [])
  return available
}

/** One range query that follows the dashboard's window, keeping its last series while it reloads. */
export function usePrometheusRange(query: string, window: MetricsWindow | null) {
  const [state, setState] = useState<{
    query: string
    window: MetricsWindow | null
    result: QueryResult<MetricSeries[]> | null
  }>({ query: '', window: null, result: null })

  useEffect(() => {
    if (!window) return
    const controller = new AbortController()
    rangeQuery(query, window, { signal: controller.signal }).then(
      (result) => setState({ query, window, result }),
      () => undefined,
    )
    return () => controller.abort()
  }, [query, window])

  return {
    result: state.query === query ? state.result : null,
    loading: Boolean(window) && (state.query !== query || state.window !== window),
  }
}
