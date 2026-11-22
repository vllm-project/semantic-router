import { useEffect, useState } from 'react'

import { fetchDashboardJson } from './dashboardPageRequests'
import { ObservabilityError, fetchObservabilityJson, isAbortError } from './observabilityProxy'

export interface JaegerRead<T> {
  data: T | null
  error: ObservabilityError | null
  loading: boolean
}

/**
 * Reads a Jaeger answer, through the Dashboard's Jaeger proxy or its trace
 * search, and parses it with `read`, again whenever the URL or `refreshKey`
 * changes. A refresh of the same URL keeps its data on screen.
 */
export function useJaegerRead<T>(
  url: string | null,
  read: (body: unknown) => T,
  refreshKey = 0,
  timeoutMs?: number,
): JaegerRead<T> {
  const [state, setState] = useState<JaegerRead<T> & { url: string | null }>({
    url: null,
    data: null,
    error: null,
    loading: false,
  })

  useEffect(() => {
    if (!url) {
      setState({ url, data: null, error: null, loading: false })
      return
    }
    const controller = new AbortController()
    setState((current) => ({
      url,
      data: current.url === url ? current.data : null,
      error: null,
      loading: true,
    }))
    fetchObservabilityJson('Jaeger', url, { signal: controller.signal, timeoutMs })
      .then((body) => setState({ url, data: read(body), error: null, loading: false }))
      .catch((error: unknown) => {
        if (isAbortError(error)) return
        setState({
          url,
          data: null,
          error:
            error instanceof ObservabilityError
              ? error
              : new ObservabilityError(
                  'query',
                  error instanceof Error ? error.message : 'Jaeger answered with unreadable data.',
                ),
          loading: false,
        })
      })
    return () => controller.abort()
  }, [url, read, refreshKey, timeoutMs])

  return state.url === url ? state : { data: null, error: null, loading: Boolean(url) }
}

export interface RouterTracingConfig {
  enabled?: boolean
  resource?: { service_name?: string }
}

/**
 * The Router's trace switch and service name, from `global.services.observability.tracing`:
 * null while loading, empty when the config cannot be read.
 */
export function useRouterTracingConfig() {
  const [tracing, setTracing] = useState<RouterTracingConfig | null>(null)
  useEffect(() => {
    let active = true
    fetchDashboardJson<{
      global?: { services?: { observability?: { tracing?: RouterTracingConfig } } }
    }>('/api/router/config/all', 'Router config')
      .then((config) => config.global?.services?.observability?.tracing ?? {})
      .catch(() => ({}))
      .then((settled) => {
        if (active) setTracing(settled)
      })
    return () => {
      active = false
    }
  }, [])
  return tracing
}
