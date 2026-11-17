import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { MessagesSquare, RefreshCw, Rocket } from 'lucide-react'

import {
  Alert,
  Badge,
  Button,
  MetaItem,
  PageContainer,
  PageHeader,
  Segmented,
  StatusLabel,
  Tooltip,
  cn,
  type StatusTone,
} from '../ui'
import RouterModelInventory from '../components/RouterModelInventory'
import ViewModal from '../components/ViewModal'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canAccessDashboardPath } from '../utils/accessControl'
import {
  getRouterModelConsumers,
  type RouterModelInfo,
  type SystemStatus,
} from '../utils/routerRuntime'
import { withRoutingView } from './configPageCanonicalLayout'
import DashboardRouterIntelligence from './DashboardRouterIntelligence'
import { fetchDashboardJson, settleDashboardRequests } from './dashboardPageRequests'
import type { RouterConfig } from './dashboardPageTypes'
import { describeServingMode } from './dashboardServingMode'
import {
  HomeConfigTiles,
  HomeModelsInUse,
  HomePublicModels,
  HomeRoutingMix,
  HomeServices,
  HomeShortcuts,
} from './HomeCards'
import { HomeKpis, HomeRecentRequests } from './HomeTraffic'
import { greeting, HOME_RANGES, readHomeRange, type HomeRange } from './homeSupport'
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
import { readRequestFilters, useRecentRuns, useRequestsData } from './insightsRequestsData'
import { listPublicModels, buildStartChecklist, type ReplayFacts } from './startSupport'
import { StartChecklistCard } from './StartChecklist'
import { useInstanceDeployment } from './useInstanceDeployment'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'

const STATUS_POLL_MS = 10_000
const REFRESH_MS = 30_000

function routerState(
  status: SystemStatus | null,
  failed: boolean,
): { tone: StatusTone; label: string } {
  if (!status)
    return failed
      ? { tone: 'danger', label: 'Status unavailable' }
      : { tone: 'neutral', label: 'Checking…' }
  switch (status.overall?.toLowerCase()) {
    case 'healthy':
      return { tone: 'success', label: 'Router healthy' }
    case 'degraded':
      return { tone: 'warning', label: 'Degraded' }
    case 'starting':
      return { tone: 'info', label: 'Starting' }
    case 'standby':
      return { tone: 'neutral', label: 'Standby' }
    default:
      return { tone: 'danger', label: 'Router not running' }
  }
}

/** Whether any request was ever recorded, for the checklist; independent of the range. */
function useLatestReplay(refreshKey: number): ReplayFacts {
  const [facts, setFacts] = useState<ReplayFacts>({ state: 'loading', count: 0 })
  useEffect(() => {
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
  }, [refreshKey])
  return facts
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
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [selectedRuntimeModel, setSelectedRuntimeModel] = useState<RouterModelInfo | null>(null)
  const runtimeReturnFocus = useRef<HTMLElement | null>(null)
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

  const fetchAll = useCallback(
    async (manual = false) => {
      if (manual) setRefreshing(true)
      const failure = await settleDashboardRequests([
        configRequest.run({ allowHidden: true }),
        statusRequest.run({ allowHidden: true }),
      ])
      if (failure) {
        setError(failure)
      } else {
        setLastUpdated(new Date())
        setError(null)
      }
      setLoading(false)
      setRefreshing(false)
    },
    [configRequest, statusRequest],
  )

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
  const filters = useMemo(() => readRequestFilters(new URLSearchParams({ range })), [range])
  const requests = useRequestsData({
    filters,
    page: 1,
    live: false,
    held: false,
    rules: facts.agentRules,
  })
  const runs = useRecentRuns(
    requests.rows.some((record) => record.session_id),
    requests.rows[0]?.id ?? '',
  )
  const latestReplay = useLatestReplay(replayKey)
  const now = useNow(REFRESH_MS)

  const refreshTraffic = useRef(requests.refresh)
  refreshTraffic.current = requests.refresh

  useEffect(() => {
    const pollStatus = () => {
      void statusRequest.run().catch(() => {
        // Ignore transient status polling errors.
      })
    }
    // Full refreshes settle config and status together before touching "Updated".
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

  const openRuntimeModel = (model: RouterModelInfo) => {
    runtimeReturnFocus.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    setSelectedRuntimeModel(model)
  }

  // The sheet opens without a dialog trigger, so closing it returns focus to nothing.
  const closeRuntimeModel = () => {
    setSelectedRuntimeModel(null)
    const target = runtimeReturnFocus.current
    window.requestAnimationFrame(() => target?.focus())
  }

  const refreshAll = () => {
    void fetchAll(true)
    requests.refresh()
    setReplayKey((key) => key + 1)
  }

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
  const router = routerState(status, statusFailed)
  const publicModels = useMemo(
    () => listPublicModels(config).filter((model) => model.api === 'chat'),
    [config],
  )
  const backendCount = Array.isArray(config?.providers?.models) ? config.providers.models.length : 0
  const steps = buildStartChecklist({ status, statusFailed, config, replay: latestReplay, now })
  const unhealthy = status?.services.filter((service) => !service.healthy) ?? []
  const overall = status?.overall?.toLowerCase()
  const firstName = user?.name?.trim().split(/\s+/)[0]
  const title = `${greeting(new Date(now).getHours())}${firstName ? `, ${firstName}` : ''}`
  const description = config
    ? `vLLM Semantic Router routes ${publicModels.length} public model${publicModels.length === 1 ? '' : 's'} across ${backendCount} backend model${backendCount === 1 ? '' : 's'}.`
    : 'Your router’s traffic, routing and health at a glance.'

  return (
    <PageContainer>
      <PageHeader
        title={title}
        description={description}
        meta={
          <>
            <StatusLabel tone={router.tone} pulse={router.tone === 'success'}>
              {router.label}
            </StatusLabel>
            {status?.version ? <MetaItem label="Version">{status.version}</MetaItem> : null}
            <span
              className="inline-flex items-center gap-1.5"
              data-testid="dashboard-serving-mode"
              title={servingMode.description}
            >
              <Badge tone="outline" size="sm">
                {servingMode.label}
              </Badge>
              <span className="max-xl:hidden">{servingMode.description}</span>
            </span>
            {isReadonly ? (
              <Badge tone="neutral" size="sm">
                Read-only
              </Badge>
            ) : null}
            {lastUpdated ? (
              <span className="tabular text-subtle-foreground">
                Updated {lastUpdated.toLocaleTimeString()}
              </span>
            ) : null}
          </>
        }
        actions={
          <>
            <Segmented
              aria-label="Time range"
              value={range}
              onValueChange={setRange}
              options={HOME_RANGES}
            />
            <Tooltip content="Refresh">
              <Button
                variant="outline"
                size="icon"
                aria-label="Refresh"
                onClick={refreshAll}
                disabled={refreshing}
              >
                <RefreshCw className={cn(refreshing && 'animate-spin')} />
              </Button>
            </Tooltip>
            <Button
              variant="outline"
              className="max-xl:hidden"
              onClick={() => navigate('/playground')}
            >
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
              <Button size="sm" variant="outline" onClick={() => void fetchAll(true)}>
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
            <HomeKpis data={requests} range={range} now={now} />
            <HomeConfigTiles config={config} loading={loading} />

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
              <HomeRecentRequests
                data={requests}
                runs={runs}
                rules={facts.agentRules}
                range={range}
                now={now}
              />
              <StartChecklistCard steps={steps} />
            </div>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
              <HomeRoutingMix aggregate={requests.aggregate} config={config} range={range} />
              <HomeModelsInUse
                aggregate={requests.aggregate}
                config={config}
                range={range}
                modelProviders={facts.modelProviders}
              />
              <HomePublicModels config={config} />
            </div>

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
              <DashboardRouterIntelligence
                config={config}
                modelsInfo={status?.models}
                onConfigure={() => navigate('/models/decision')}
                onOpenStatus={() => navigate('/observe/status')}
                onSelectModel={openRuntimeModel}
              />
              <div className="flex flex-col gap-4">
                <HomeServices
                  status={status}
                  failed={statusFailed}
                  onRetry={() =>
                    void statusRequest.run({ allowHidden: true }).catch(() => undefined)
                  }
                />
                <HomeShortcuts canOpen={(path) => canAccessDashboardPath(user, path)} />
              </div>
            </div>
          </div>
        </ModelLabelsProvider>
      </InsightsConfigContext.Provider>

      {selectedRuntimeModel ? (
        <ViewModal
          isOpen
          title="Runtime model details"
          onClose={closeRuntimeModel}
          sections={[
            {
              fields: [
                {
                  label: 'Runtime',
                  fullWidth: true,
                  value: (
                    <RouterModelInventory
                      mode="detail"
                      modelsInfo={{
                        models: getRouterModelConsumers(
                          status?.models?.models ?? [],
                          selectedRuntimeModel,
                        ),
                      }}
                    />
                  ),
                },
              ],
            },
          ]}
        />
      ) : null}
    </PageContainer>
  )
}

export default DashboardPage
