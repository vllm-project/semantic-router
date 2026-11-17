import { useCallback, useEffect, useRef, useState } from 'react'

import {
  buildPlaygroundRequestHeaders,
  collectResponseHeaders,
} from '../components/chatRequestSupport'
import { playgroundHTTPFailure } from '../components/playgroundErrorPresentation'
import { listRouterModels, type RouterModelOption } from '../utils/routerModelSelection'
import type { SystemStatus } from '../utils/routerRuntime'
import { withRoutingView } from './configPageCanonicalLayout'
import { fetchDashboardJson } from './dashboardPageRequests'
import type { RouterConfig } from './dashboardPageTypes'
import { fetchInsightsJSON, insightsUnavailableReason } from './insightsPageApi'
import type { InsightsListResponse, InsightsRecord } from './insightsPageTypes'
import type { ReplayFacts } from './startSupport'

const REPLAY_POLL_MS = 5_000
const STATUS_POLL_MS = 15_000

export interface GetStartedData {
  config: RouterConfig | null
  configError: string | null
  status: SystemStatus | null
  statusFailed: boolean
  /** The listener port setup reports, for a viewer who cannot read the configuration. */
  setupPort?: number
  /** The chat models the router advertises now; null until it answers or when it cannot. */
  liveModels: RouterModelOption[] | null
  replay: ReplayFacts
  latest: InsightsRecord | null
  loading: boolean
  refreshReplay: () => void
}

/**
 * What the Get started page reads: the configuration (listeners and public
 * models), the router status, the advertised models, and the newest recorded
 * request, which it polls so a first request shows up as it lands.
 */
export function useGetStartedData(): GetStartedData {
  const [config, setConfig] = useState<RouterConfig | null>(null)
  const [configError, setConfigError] = useState<string | null>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [statusFailed, setStatusFailed] = useState(false)
  const [setupPort, setSetupPort] = useState<number>()
  const [liveModels, setLiveModels] = useState<RouterModelOption[] | null>(null)
  const [replay, setReplay] = useState<ReplayFacts>({ state: 'loading', count: 0 })
  const [latest, setLatest] = useState<InsightsRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const replaySequence = useRef(0)

  const loadReplay = useCallback(async () => {
    const current = ++replaySequence.current
    try {
      const list = await fetchInsightsJSON<InsightsListResponse>(
        '/api/router/api/v1/observability/replays?limit=1&offset=0&showDetails=false',
        'recent requests',
      )
      if (current !== replaySequence.current) return
      const record = list.data?.[0] ?? null
      setLatest(record)
      setReplay({
        state: 'ready',
        count: typeof list.total === 'number' ? list.total : (list.data?.length ?? 0),
        latestAt: record?.timestamp,
      })
    } catch (error) {
      if (current !== replaySequence.current) return
      setReplay({ state: 'unavailable', count: 0, reason: insightsUnavailableReason(error) })
    }
  }, [])

  const loadStatus = useCallback(async () => {
    try {
      setStatus(await fetchDashboardJson<SystemStatus>('/api/status', 'System status'))
      setStatusFailed(false)
    } catch {
      setStatusFailed(true)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    const loadConfig = fetchDashboardJson<RouterConfig>('/api/router/config/all', 'Router config')
      .then((value) => {
        if (cancelled) return
        setConfig(withRoutingView(value))
        setConfigError(null)
      })
      .catch((error: unknown) => {
        if (!cancelled) setConfigError(error instanceof Error ? error.message : String(error))
      })
    const loadSetup = fetchDashboardJson<{ listenerPort?: number }>(
      '/api/setup/state',
      'Setup state',
    )
      .then((value) => {
        if (!cancelled && typeof value.listenerPort === 'number') setSetupPort(value.listenerPort)
      })
      .catch(() => undefined)
    const loadModels = fetchDashboardJson<unknown>('/api/router/v1/models', 'Router models')
      .then((payload) => {
        if (!cancelled) setLiveModels(listRouterModels(payload))
      })
      .catch(() => undefined)
    void Promise.allSettled([loadConfig, loadStatus(), loadSetup, loadModels, loadReplay()]).then(
      () => {
        if (!cancelled) setLoading(false)
      },
    )
    return () => {
      cancelled = true
    }
  }, [loadReplay, loadStatus])

  useEffect(() => {
    const tick = (load: () => Promise<void>) => () => {
      if (!document.hidden) void load()
    }
    const replayTimer = window.setInterval(tick(loadReplay), REPLAY_POLL_MS)
    const statusTimer = window.setInterval(tick(loadStatus), STATUS_POLL_MS)
    const onVisible = () => {
      if (!document.hidden) {
        void loadReplay()
        void loadStatus()
      }
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      window.clearInterval(replayTimer)
      window.clearInterval(statusTimer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [loadReplay, loadStatus])

  const refreshReplay = useCallback(() => void loadReplay(), [loadReplay])

  return {
    config,
    configError,
    status,
    statusFailed,
    setupPort,
    liveModels,
    replay,
    latest,
    loading,
    refreshReplay,
  }
}

export interface TestRequestResult {
  /** The routing headers of the answer, as the playground reads them. */
  headers: Record<string, string>
  latencyMs: number
  answer: string
}

/**
 * One non-streamed request through the dashboard's gateway proxy, the path the
 * playground sends on, so the answer carries the same routing headers.
 */
export async function sendGatewayTestRequest(model: string): Promise<TestRequestResult> {
  const startedAt = performance.now()
  const response = await fetch('/api/router/v1/chat/completions', {
    method: 'POST',
    headers: buildPlaygroundRequestHeaders(`get-started-${Date.now()}`),
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: 'Hello!' }],
      stream: false,
    }),
  })
  if (!response.ok) {
    throw playgroundHTTPFailure(response.status, response.statusText, await response.text())
  }
  const headers = collectResponseHeaders(response)
  const body = (await response.json().catch(() => null)) as {
    choices?: Array<{ message?: { content?: unknown } }>
  } | null
  const content = body?.choices?.[0]?.message?.content
  return {
    headers,
    latencyMs: Math.round(performance.now() - startedAt),
    answer: typeof content === 'string' ? content : '',
  }
}
