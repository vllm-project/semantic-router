import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { MessagesSquare, Rocket } from 'lucide-react'

import { Alert, Badge, Button, PageContainer, PageHeader, Segmented, Tooltip } from '../ui'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import type { SystemStatus } from '../utils/routerRuntime'
import { useDashboardObservation } from '../utils/useDashboardObservation'
import { withRoutingView } from './configPageCanonicalLayout'
import DashboardRouterIntelligence from './DashboardRouterIntelligence'
import { decisionModelHealth } from './dashboardRouterIntelligenceSupport'
import { fetchDashboardJson, settleDashboardRequests } from './dashboardPageRequests'
import type { RouterConfig } from './dashboardPageTypes'
import { describeServingMode } from './dashboardServingMode'
import type { ModelRuntimeInventory } from './decisionModelManagement'
import {
  HomeAgentsInUse,
  HomeConfigTiles,
  HomeModelsInUse,
  HomeRoutingMix,
  HomeServices,
} from './HomeCards'
import { HomeKpis, HomeRecentRequests } from './HomeTraffic'
import { HomeTrafficChart } from './HomeTrafficChart'
import {
  formatUpdatedAgo,
  greeting,
  HOME_RANGES,
  homeTrafficView,
  readHomeRange,
  type HomeRange,
} from './homeSupport'
import { useHomeTraffic } from './homeTraffic'
import {
  buildAgentRuleIndex,
  buildModelProviders,
  EMPTY_CONFIG_FACTS,
  InsightsConfigContext,
  type ConfigDocument,
} from './insightsConfigFacts'
import { useNow } from './insightsDisplayFormat'
import { ModelLabelsProvider } from './insightsModelLabelSupport'
import { fetchInsightsJSON, insightsUnavailableReason } from './insightsPageApi'
import type { InsightsListResponse } from './insightsPageTypes'
import { listPublicModels, buildStartChecklist, type ReplayFacts } from './startSupport'
import { StartChecklistCard } from './StartChecklist'
import { useInstanceDeployment } from './useInstanceDeployment'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'

const STATUS_POLL_MS = 10_000
const REFRESH_MS = 30_000

/**
 * Whether any request was ever recorded, for the checklist. The range's newest
 * records answer it when they hold any; only an empty range asks Router Replay.
 */
function useLatestReplay(refreshKey: number, enabled: boolean): ReplayFacts {
  const [facts, setFacts] = useState<ReplayFacts>({ state: 'loading', count: 0 })
  useEffect(() => {
    if (!enabled) return undefined
    let cancelled = false
    fetchInsightsJSON<InsightsListResponse>(
      '/api/router/api/v1/observability/replays?limit=1&offset=0&showDetails=false',
      'recent requests',
    )
      .then((list) => {
        if (cancelled) return
        setFacts({
          state: 'ready',
          count: typeof list.total === 'number' ? list.total : (list.data?.length ?? 0),
          latestAt: list.data?.[0]?.timestamp,
        })
      })
      .catch((error: unknown) => {
        if (!cancelled)
          setFacts({ state: 'unavailable', count: 0, reason: insightsUnavailableReason(error) })
      })
    return () => {
      cancelled = true
    }
  }, [refreshKey, enabled])
  return facts
}

/** The range control's tooltip, ticking only while it is open. */
function RefreshNote({ updatedAt }: { updatedAt: number | null }) {
  const now = useNow(1_000)
  return (
    <span className="flex flex-col">
      {updatedAt ? <span>{formatUpdatedAgo(updatedAt, now)}</span> : null}
      <span className="font-normal opacity-70">Refreshes every 30 seconds</span>
    </span>
  )
}

const DashboardPage = () => {
  const navigate = useNavigate()
  const { user } = useAuth()
  const { isReadonly } = useReadonly()
  const instance = useInstanceDeployment()
  const [params, setParams] = useSearchParams()
  const range = readHomeRange(params.get('range'))

  const [rawConfig, setRawConfig] = useState<ConfigDocument | null>(null)
  const [config, setConfig] = useState<RouterConfig | null>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [statusFailed, setStatusFailed] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [updatedAt, setUpdatedAt] = useState<number | null>(null)
  const [replayKey, setReplayKey] = useState(0)

  const fetchStatus = useCallback(async () => {
    try {
      setStatus(await fetchDashboardJson<SystemStatus>('/api/status', 'System status'))
      setStatusFailed(false)
    } catch (failure) {
      setStatusFailed(true)
      throw failure
    }
  }, [])

  const fetchConfig = useCallback(async () => {
    const document = await fetchDashboardJson<RouterConfig>(
      '/api/router/config/all',
      'Router config',
    )
    setRawConfig(document as ConfigDocument)
    setConfig(withRoutingView(document))
  }, [])

  const statusRequest = useMemo(() => createVisibilityAwareRequest(fetchStatus), [fetchStatus])
  const configRequest = useMemo(() => createVisibilityAwareRequest(fetchConfig), [fetchConfig])

  const fetchAll = useCallback(async () => {
    const failure = await settleDashboardRequests([
      configRequest.run({ allowHidden: true }),
      statusRequest.run({ allowHidden: true }),
    ])
    if (failure) {
      setError(failure)
    } else {
      setUpdatedAt(Date.now())
      setError(null)
    }
    setLoading(false)
  }, [configRequest, statusRequest])

  const facts = useMemo(
    () =>
      rawConfig
        ? {
            agentRules: buildAgentRuleIndex(rawConfig),
            modelProviders: buildModelProviders(rawConfig),
          }
        : EMPTY_CONFIG_FACTS,
    [rawConfig],
  )
  const traffic = useHomeTraffic(range, user?.id ?? 'anonymous')
  const view = useMemo(
    () => homeTrafficView(traffic.aggregate, traffic.window, range),
    [range, traffic.aggregate, traffic.window],
  )
  const inventory = useDashboardObservation<ModelRuntimeInventory>(
    '/api/router/api/v1/inventory/model-runtime',
  )
  const rangeReplay: ReplayFacts | null = traffic.unavailable
    ? { state: 'unavailable', count: 0, reason: traffic.unavailable }
    : traffic.rows.length
      ? {
          state: 'ready',
          count: traffic.aggregate?.record_count || traffic.rows.length,
          latestAt: traffic.rows[0].timestamp,
        }
      : null
  const latestReplay = useLatestReplay(replayKey, !rangeReplay && !traffic.rowsLoading)
  const now = useNow(REFRESH_MS)

  const refreshTraffic = useRef(traffic.refresh)
  refreshTraffic.current = traffic.refresh

  useEffect(() => {
    const pollStatus = () => {
      void statusRequest.run().catch(() => {
        // Ignore transient status polling errors.
      })
    }
    // Full refreshes settle config and status together before touching the update time.
    const refreshWhenVisible = () => {
      if (document.hidden) return
      void fetchAll()
      refreshTraffic.current()
      setReplayKey((key) => key + 1)
    }
    const onConfigDeployed = () => {
      void fetchAll()
    }

    void fetchAll()
    const statusInterval = window.setInterval(pollStatus, STATUS_POLL_MS)
    const refreshInterval = window.setInterval(refreshWhenVisible, REFRESH_MS)
    window.addEventListener('config-deployed', onConfigDeployed)
    document.addEventListener('visibilitychange', refreshWhenVisible)
    return () => {
      window.clearInterval(statusInterval)
      window.clearInterval(refreshInterval)
      window.removeEventListener('config-deployed', onConfigDeployed)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [fetchAll, statusRequest])

  const setRange = (next: HomeRange) =>
    setParams(
      (current) => {
        const nextParams = new URLSearchParams(current)
        if (next === '24h') nextParams.delete('range')
        else nextParams.set('range', next)
        return nextParams
      },
      { replace: true },
    )

  const servingMode = describeServingMode(
    instance.status?.observed_mode === 'unknown'
      ? status?.serving_mode
      : (instance.status?.observed_mode ?? status?.serving_mode),
  )
  const modeKnown = Boolean(status || statusFailed || instance.status)
  const publicModels = useMemo(
    () => listPublicModels(config).filter((model) => model.api === 'chat'),
    [config],
  )
  const backendCount = Array.isArray(config?.providers?.models) ? config.providers.models.length : 0
  const steps = buildStartChecklist({
    status,
    statusFailed,
    config,
    replay: rangeReplay ?? latestReplay,
    decisionModel: config ? decisionModelHealth(config, inventory.data, status?.models) : null,
    now,
  })
  const unhealthy = status?.services.filter((service) => !service.healthy) ?? []
  const overall = status?.overall?.toLowerCase()
  const firstName = user?.name?.trim().split(/\s+/)[0]
  const title = `${greeting(new Date(now).getHours())}${firstName ? `, ${firstName}` : ''}`
  const instanceName =
    status?.deployment_type && status.deployment_type !== 'none'
      ? `vllm-sr · ${status.deployment_type}`
      : 'vLLM Semantic Router'
  // The mode and access qualify the instance, so they sit beside its name.
  const chips = (
    <>
      {modeKnown ? (
        <Tooltip content={servingMode.description}>
          <span className="mx-1.5 inline-flex align-[1px]" data-testid="dashboard-serving-mode">
            <Badge tone="outline" size="sm">
              {servingMode.label}
            </Badge>
            <span className="sr-only">: {servingMode.description}</span>
          </span>
        </Tooltip>
      ) : null}
      {isReadonly ? (
        <Badge tone="neutral" size="sm" className="mr-1.5 align-[1px]">
          Read-only
        </Badge>
      ) : null}
    </>
  )
  const summary = config ? (
    <>
      <span className="font-medium text-foreground">{instanceName}</span>
      {modeKnown || isReadonly ? chips : ' '}
      is routing {publicModels.length} public model{publicModels.length === 1 ? '' : 's'} across{' '}
      {backendCount} backend model{backendCount === 1 ? '' : 's'}.
    </>
  ) : (
    <>
      Your router’s traffic, routing and health at a glance.
      {chips}
    </>
  )

  return (
    <PageContainer>
      <PageHeader
        title={title}
        description={summary}
        actions={
          <>
            <Tooltip content={<RefreshNote updatedAt={updatedAt} />} side="bottom">
              <span className="inline-flex" data-testid="home-range">
                <Segmented
                  aria-label="Time range"
                  value={range}
                  onValueChange={setRange}
                  options={HOME_RANGES}
                />
              </span>
            </Tooltip>
            <Button variant="outline" onClick={() => navigate('/playground')}>
              <MessagesSquare />
              Open playground
            </Button>
            <Button onClick={() => navigate('/get-started')}>
              <Rocket />
              Connect an app
            </Button>
          </>
        }
      />

      <div className="mb-5 flex flex-col gap-3 empty:hidden">
        {error ? (
          <Alert
            tone="danger"
            title="Some dashboard data could not load"
            actions={
              <Button size="sm" variant="outline" onClick={() => void fetchAll()}>
                Retry
              </Button>
            }
          >
            Failed to load data: {error}
          </Alert>
        ) : null}
        {status && overall !== 'healthy' ? (
          <Alert
            tone={overall === 'degraded' ? 'warning' : overall === 'starting' ? 'info' : 'danger'}
            title={
              overall === 'degraded'
                ? `${unhealthy.length} of ${status.services.length} services need attention`
                : overall === 'starting'
                  ? 'The router is starting'
                  : overall === 'standby'
                    ? 'The router is on standby'
                    : 'The router is not running'
            }
            actions={
              <Button size="sm" variant="outline" onClick={() => navigate('/observe/status')}>
                View status
              </Button>
            }
          >
            {overall === 'degraded'
              ? unhealthy.map((service) => service.name).join(', ')
              : overall === 'starting'
                ? 'Requests route as soon as the models it needs are ready.'
                : 'Requests, live metrics and the playground need it. Start it with vllm-sr serve, then check its status.'}
          </Alert>
        ) : null}
      </div>

      <InsightsConfigContext.Provider value={facts}>
        <ModelLabelsProvider>
          <div className="flex flex-col gap-4">
            <HomeKpis traffic={traffic} view={view} range={range} />

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
              <HomeTrafficChart traffic={traffic} view={view} range={range} />
              <StartChecklistCard steps={steps} />
            </div>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
              <HomeRoutingMix aggregate={traffic.aggregate} config={config} range={range} />
              <HomeModelsInUse
                aggregate={traffic.aggregate}
                config={config}
                range={range}
                modelProviders={facts.modelProviders}
              />
              <HomeAgentsInUse traffic={traffic} range={range} />
            </div>

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
              <HomeRecentRequests
                traffic={traffic}
                rules={facts.agentRules}
                range={range}
                now={now}
              />
              <HomeServices
                status={status}
                failed={statusFailed}
                onRetry={() => void statusRequest.run({ allowHidden: true }).catch(() => undefined)}
              />
            </div>

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
              <DashboardRouterIntelligence
                config={config}
                loading={loading}
                inventory={inventory.data}
                routerModels={status?.models}
                onConfigure={() => navigate('/models/decision')}
                className="xl:col-span-2"
              />
              <HomeConfigTiles config={config} loading={loading} />
            </div>
          </div>
        </ModelLabelsProvider>
      </InsightsConfigContext.Provider>
    </PageContainer>
  )
}

export default DashboardPage
