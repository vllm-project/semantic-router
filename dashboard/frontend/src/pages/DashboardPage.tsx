import React, { useEffect, useState, useCallback, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import { useAuth } from '../contexts/AuthContext'
import { useInferenceRoutingAccess } from '../contexts/InferenceRoutingAccessContext'
import { useSystemStatus } from '../contexts/SystemStatusContext'
import { canAccessDashboardPath } from '../utils/accessControl'
import { inferenceAccessApi, type AccessOverview } from '../utils/inferenceAccessApi'
import {
  fetchManagedRoutingOverviewSnapshot,
  type ManagedRoutingSnapshot,
} from '../utils/managedRoutingSnapshot'
import { DashboardMiniFlowDiagram } from './DashboardMiniFlowDiagram'
import DashboardRoutingHero from './DashboardRoutingHero'
import DashboardRoutingProfiles from './DashboardRoutingProfiles'
import StatusAvailabilityPanel from './StatusAvailabilityPanel'
import {
  categorizeDecisions,
  countDecisions,
  countModels,
  countPlugins,
  countSignals,
  getAllDecisions,
} from './dashboardPageStats'
import { buildDecisionPreviewRows, buildSignalBreakdownRows } from './dashboardPageOverview'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'
import styles from './DashboardPage.module.css'

const formatWholeCount = (value: string) => new Intl.NumberFormat('en-US').format(BigInt(value))

const DashboardPage: React.FC = () => {
  const navigate = useNavigate()
  const { user } = useAuth()
  const {
    status,
    isLoading: statusLoading,
    lastUpdated: statusLastUpdated,
    routingAccess,
    refresh: refreshSystemStatus,
  } = useSystemStatus()
  const { catalogSnapshot, catalogStatus, usesKeyScopedCatalog } = useInferenceRoutingAccess()

  const [config, setConfig] = useState<ManagedRoutingSnapshot | null>(null)
  const [accessOverview, setAccessOverview] = useState<AccessOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchConfig = useCallback(async () => {
    const snapshot = await fetchManagedRoutingOverviewSnapshot()
    setConfig(snapshot)
    setError(null)
  }, [])

  const routingIdentityUnavailable = Boolean(
    user?.managementIdentityStatus && user.managementIdentityStatus !== 'ready',
  )
  const routingAccessUnavailable = routingIdentityUnavailable || routingAccess !== 'operational'
  const canReadConfig = !routingAccessUnavailable && canAccessDashboardPath(user, '/config/models')
  const canReadIntelligence =
    canReadConfig ||
    (!routingAccessUnavailable &&
      usesKeyScopedCatalog &&
      canAccessDashboardPath(user, '/topology'))
  const canReadAccess = !routingAccessUnavailable && canAccessDashboardPath(user, '/access/usage')
  const canReadStatus = canAccessDashboardPath(user, '/status')
  const showSystemHealth = !routingAccessUnavailable
  const canUsePlayground = !routingAccessUnavailable && canAccessDashboardPath(user, '/playground')
  const overviewConfig = canReadConfig ? config : usesKeyScopedCatalog ? catalogSnapshot : null
  const fetchAccess = useCallback(async () => {
    if (!canReadAccess) return
    setAccessOverview(await inferenceAccessApi.overview())
  }, [canReadAccess])

  const configRequest = useMemo(() => createVisibilityAwareRequest(fetchConfig), [fetchConfig])

  const fetchAll = useCallback(
    async (manual = false) => {
      if (manual) setRefreshing(true)
      try {
        await Promise.all([
          canReadConfig ? configRequest.run({ allowHidden: true }) : Promise.resolve(),
          fetchAccess(),
          manual ? refreshSystemStatus() : Promise.resolve(),
        ])
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard data')
      } finally {
        setLoading(false)
        setRefreshing(false)
      }
    },
    [canReadConfig, configRequest, fetchAccess, refreshSystemStatus],
  )

  useEffect(() => {
    const pollConfig = () => {
      if (!canReadConfig) return
      void configRequest.run().catch((pollError) => {
        setError(
          pollError instanceof Error ? pollError.message : 'Failed to refresh dashboard config',
        )
      })
    }
    const onVisibilityChange = () => {
      if (!document.hidden) pollConfig()
    }
    void fetchAll()
    const configInterval = window.setInterval(pollConfig, 30000)
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => {
      window.clearInterval(configInterval)
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
  }, [canReadConfig, configRequest, fetchAll])

  const signalStats = useMemo(
    () => (overviewConfig ? countSignals(overviewConfig) : { total: 0, byType: {} }),
    [overviewConfig],
  )
  const decisionCount = useMemo(
    () => (overviewConfig ? countDecisions(overviewConfig) : 0),
    [overviewConfig],
  )
  const modelCount = useMemo(
    () => (overviewConfig ? countModels(overviewConfig) : 0),
    [overviewConfig],
  )
  const pluginCount = useMemo(
    () => (overviewConfig ? countPlugins(overviewConfig) : 0),
    [overviewConfig],
  )
  const currentDecisions = useMemo(() => (config ? getAllDecisions(config) : []), [config])
  const categorizedDecisions = useMemo(
    () => (config ? categorizeDecisions(config) : { guardrails: [], routing: [], fallbacks: [] }),
    [config],
  )
  const signalBreakdownRows = useMemo(
    () => buildSignalBreakdownRows(signalStats.byType),
    [signalStats.byType],
  )
  const decisionPreviewRows = useMemo(
    () =>
      buildDecisionPreviewRows([
        ...categorizedDecisions.guardrails,
        ...categorizedDecisions.routing,
        ...categorizedDecisions.fallbacks,
      ]),
    [categorizedDecisions],
  )
  if ((loading || statusLoading) && !config && !status) {
    return <ProductLoadingState label="Loading dashboard" />
  }

  if (routingAccessUnavailable) {
    return (
      <div
        className={`${styles.page} ${styles.statusOnlyPage}`}
        data-testid="routing-access-status-only"
      >
        <StatusAvailabilityPanel status={status} lastUpdated={statusLastUpdated} />
      </div>
    )
  }

  return (
    <div className={styles.page}>
      <DashboardRoutingHero
        modelCount={modelCount}
        signalCount={signalStats.total}
        decisionCount={decisionCount}
        apiKeyCount={accessOverview?.activeKeys ?? null}
        showRoutingMetrics={canReadConfig}
        showAPIKeyMetric={canReadAccess}
        showPlaygroundAction={canUsePlayground}
        showStatus={canReadStatus}
        overallStatus={canReadStatus ? status?.overall : undefined}
        refreshing={refreshing}
        lastUpdated={statusLastUpdated}
        onRefresh={() => void fetchAll(true)}
        onNavigate={navigate}
      />

      {error ? (
        <div className={styles.errorBanner} role="alert">
          <span>{`Failed to load data: ${error}`}</span>
          <button onClick={() => void fetchAll(true)}>Retry</button>
        </div>
      ) : null}

      <div className={`${styles.mainGrid} ${!canReadIntelligence ? styles.mainGridCompact : ''}`}>
        {canReadIntelligence ? (
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2 className={styles.cardTitle}>Intelligence Layers</h2>
              <button
                type="button"
                className={styles.cardAction}
                onClick={() => navigate('/topology')}
              >
                View topology
                <ProductIcon name="chevron-right" aria-hidden="true" />
              </button>
            </div>
            <div className={styles.flowContainer}>
              {overviewConfig ? (
                <DashboardMiniFlowDiagram
                  signals={signalStats}
                  decisions={decisionCount}
                  models={modelCount}
                  plugins={pluginCount}
                />
              ) : (canReadConfig && loading) ||
                (usesKeyScopedCatalog && catalogStatus === 'loading') ? (
                <ProductLoadingState compact label="Loading routing overview" />
              ) : (
                <div className={styles.emptyState}>
                  {usesKeyScopedCatalog && catalogStatus === 'error'
                    ? 'Routing overview unavailable'
                    : 'No accessible routing paths'}
                </div>
              )}
            </div>
          </div>
        ) : null}

        {showSystemHealth || canReadAccess ? (
          <div className={styles.rightCol}>
            {showSystemHealth ? (
              <div className={styles.card}>
                <div className={styles.cardHeader}>
                  <h2 className={styles.cardTitle}>System Health</h2>
                  {canReadStatus ? (
                    <button
                      type="button"
                      className={styles.cardAction}
                      onClick={() => navigate('/status')}
                    >
                      Details
                      <ProductIcon name="chevron-right" aria-hidden="true" />
                    </button>
                  ) : null}
                </div>
                <div className={styles.healthContent}>
                  {status ? (
                    <>
                      <div className={styles.healthOverall}>
                        <span
                          className={`${styles.healthDot} ${
                            status.overall === 'healthy'
                              ? styles.healthDotGreen
                              : status.overall === 'degraded'
                                ? styles.healthDotYellow
                                : styles.healthDotRed
                          }`}
                        />
                        <span className={styles.healthLabel}>
                          {status.overall === 'not_running'
                            ? 'Not Running'
                            : status.overall.charAt(0).toUpperCase() + status.overall.slice(1)}
                        </span>
                      </div>
                      <div className={styles.servicesList}>
                        {status.services.slice(0, 6).map((svc, i) => (
                          <div key={i} className={styles.serviceRow}>
                            <span
                              className={`${styles.svcDot} ${svc.healthy ? styles.svcDotOk : styles.svcDotFail}`}
                            />
                            <span className={styles.svcName}>{svc.name}</span>
                            <span
                              className={`${styles.svcStatus} ${svc.healthy ? styles.svcStatusOk : styles.svcStatusFail}`}
                            >
                              {svc.status}
                            </span>
                          </div>
                        ))}
                        {status.services.length > 6 && (
                          <div className={styles.moreServices}>
                            +{status.services.length - 6} more
                          </div>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className={styles.emptyState}>Unable to fetch status</div>
                  )}
                </div>
              </div>
            ) : null}

            {canReadAccess ? (
              <div className={styles.card}>
                <div className={styles.cardHeader}>
                  <h2 className={styles.cardTitle}>Access</h2>
                  <button
                    type="button"
                    className={styles.cardAction}
                    onClick={() => navigate('/access/usage')}
                  >
                    Details
                    <ProductIcon name="chevron-right" aria-hidden="true" />
                  </button>
                </div>
                {accessOverview ? (
                  <div className={styles.accessSnapshot}>
                    {accessOverview.activeKeys !== null && (
                      <div>
                        <strong>{formatWholeCount(accessOverview.activeKeys)}</strong>
                        <span>active keys</span>
                      </div>
                    )}
                    <div>
                      <strong>{accessOverview.requestsToday.toLocaleString('en-US')}</strong>
                      <span>requests today</span>
                    </div>
                    <div>
                      <strong>{accessOverview.tokensToday.toLocaleString('en-US')}</strong>
                      <span>tokens today</span>
                    </div>
                    <div>
                      <strong>
                        {accessOverview.requestsToday
                          ? `${Math.round((accessOverview.successfulToday / accessOverview.requestsToday) * 1000) / 10}%`
                          : '—'}
                      </strong>
                      <span>success rate</span>
                    </div>
                  </div>
                ) : (
                  <div className={styles.emptyState}>Access data is unavailable</div>
                )}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      {canReadConfig && config ? (
        <DashboardRoutingProfiles
          config={config}
          onOpenTopology={(scopeId) => navigate(`/topology?scope=${encodeURIComponent(scopeId)}`)}
        />
      ) : null}

      {canReadConfig ? (
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <div>
              <h2 className={styles.cardTitle}>Models</h2>
              <span className={styles.cardSubtitle}>{modelCount} connected</span>
            </div>
            <button
              type="button"
              className={styles.cardAction}
              onClick={() => navigate('/config/models')}
            >
              View all
              <ProductIcon name="chevron-right" aria-hidden="true" />
            </button>
          </div>
          {config?.models.length ? (
            <div className={`${styles.servicesList} ${styles.modelList}`}>
              {config.models.slice(0, 6).map((model) => (
                <article key={model.id} className={styles.modelRow}>
                  <span className={styles.modelIdentity}>
                    <strong>{model.name}</strong>
                    <small>
                      {[model.card.paramSize, model.card.modality].filter(Boolean).join(' · ') ||
                        'Connected model'}
                    </small>
                  </span>
                  <span className={styles.modelCapability}>
                    {model.card.capabilities[0] || 'Ready'}
                  </span>
                </article>
              ))}
              {config.models.length > 6 ? (
                <div className={styles.moreServices}>+{config.models.length - 6} more</div>
              ) : null}
            </div>
          ) : (
            <div className={styles.emptyState}>No models connected</div>
          )}
        </div>
      ) : null}

      {canReadConfig ? (
        <div className={styles.bottomGrid}>
          {signalStats.total > 0 && (
            <div className={styles.card}>
              <div className={styles.cardHeader}>
                <h2 className={styles.cardTitle}>Signal Breakdown</h2>
                <span className={styles.cardSubtitle}>{signalStats.total} total</span>
              </div>
              <div className={styles.signalBreakdown}>
                {signalBreakdownRows.map((row) => (
                  <div
                    key={row.type}
                    className={styles.breakdownRow}
                    title={`${row.type}: ${row.count} signal(s)`}
                  >
                    <span className={styles.breakdownLabel}>
                      <span className={styles.breakdownDot} style={{ background: row.color }} />
                      {row.type}
                    </span>
                    <div className={styles.breakdownBar}>
                      <div
                        className={styles.breakdownFill}
                        style={{ width: `${row.percent}%`, background: row.color }}
                      />
                    </div>
                    <span className={styles.breakdownCount}>{row.count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {currentDecisions.length > 0 && (
            <div className={styles.card}>
              <div className={styles.cardHeader}>
                <h2 className={styles.cardTitle}>Decisions Overview</h2>
                <button
                  type="button"
                  className={styles.cardAction}
                  onClick={() => navigate('/config/decisions')}
                >
                  View all
                  <ProductIcon name="chevron-right" aria-hidden="true" />
                </button>
              </div>
              <div className={styles.decisionTable}>
                <div className={styles.decisionTableHead}>
                  <span>Name</span>
                  <span>Priority</span>
                  <span>Type</span>
                  <span>Models</span>
                </div>
                {decisionPreviewRows.map((row) => (
                  <div key={row.key} className={styles.decisionTableRow}>
                    <span className={styles.decisionIdentity} title={row.title}>
                      <span className={styles.decisionName}>{row.name}</span>
                      {row.scopeLabel ? (
                        <span className={styles.decisionScope}>{row.scopeLabel}</span>
                      ) : null}
                    </span>
                    <span className={styles.decisionPriority}>{row.priorityLabel}</span>
                    <span
                      className={`${styles.decisionBadge} ${
                        row.category === 'guardrail'
                          ? styles.badgeGuardrail
                          : row.category === 'fallback'
                            ? styles.badgeFallback
                            : styles.badgeRouting
                      }`}
                    >
                      {row.typeLabel}
                    </span>
                    <span className={styles.decisionModels} title={row.modelNames}>
                      {row.modelNames}
                    </span>
                  </div>
                ))}
                {currentDecisions.length > 10 && (
                  <button
                    type="button"
                    className={styles.decisionTableMore}
                    onClick={() => navigate('/config/decisions')}
                  >
                    +{currentDecisions.length - 10} more decisions
                    <ProductIcon name="chevron-right" aria-hidden="true" />
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}

export default DashboardPage
