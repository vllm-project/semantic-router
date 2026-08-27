import React, { useEffect, useState, useCallback, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import RouterModelInventory from '../components/RouterModelInventory'
import ProductLoadingState from '../components/ProductLoadingState'
import ProductIcon from '../components/ProductIcon'
import {
  getLoadedModelCount,
  getModelStatusSummary,
  getRouterModelAnchor,
  getTotalKnownModelCount,
  type SystemStatus,
} from '../utils/routerRuntime'
import { DashboardMiniFlowDiagram } from './DashboardMiniFlowDiagram'
import DashboardRoutingProfiles from './DashboardRoutingProfiles'
import type { RouterConfig } from './dashboardPageTypes'
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

const DashboardPage: React.FC = () => {
  const navigate = useNavigate()

  const [config, setConfig] = useState<RouterConfig | null>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const fetchStatus = useCallback(async () => {
    const statusRes = await fetch('/api/status')
    if (statusRes.ok) {
      setStatus(await statusRes.json())
    }
  }, [])

  const fetchConfig = useCallback(async () => {
    const configResult = await fetch('/api/router/config/all')
    if (configResult.ok) {
      setConfig(await configResult.json())
      setLastUpdated(new Date())
      setError(null)
    }
  }, [])

  const statusRequest = useMemo(() => createVisibilityAwareRequest(fetchStatus), [fetchStatus])
  const configRequest = useMemo(() => createVisibilityAwareRequest(fetchConfig), [fetchConfig])

  const fetchAll = useCallback(
    async (manual = false) => {
      if (manual) setRefreshing(true)
      try {
        await Promise.all([
          configRequest.run({ allowHidden: true }),
          statusRequest.run({ allowHidden: true }),
        ])
        setLastUpdated(new Date())
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard data')
      } finally {
        setLoading(false)
        setRefreshing(false)
      }
    },
    [configRequest, statusRequest],
  )

  useEffect(() => {
    const pollStatus = () => {
      void statusRequest.run().catch(() => {
        // Ignore transient status polling errors.
      })
    }
    const pollConfig = () => {
      void configRequest.run().catch((pollError) => {
        setError(
          pollError instanceof Error ? pollError.message : 'Failed to refresh dashboard config',
        )
      })
    }
    const onVisibilityChange = () => {
      if (!document.hidden) {
        pollStatus()
        pollConfig()
      }
    }
    const onConfigDeployed = () => {
      void fetchAll()
    }

    void fetchAll()
    const statusInterval = window.setInterval(pollStatus, 10000)
    const configInterval = window.setInterval(pollConfig, 30000)
    window.addEventListener('config-deployed', onConfigDeployed)
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => {
      window.clearInterval(statusInterval)
      window.clearInterval(configInterval)
      window.removeEventListener('config-deployed', onConfigDeployed)
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
  }, [configRequest, fetchAll, statusRequest])

  const signalStats = useMemo(
    () => (config ? countSignals(config) : { total: 0, byType: {} }),
    [config],
  )
  const decisionCount = useMemo(() => (config ? countDecisions(config) : 0), [config])
  const modelCount = useMemo(() => (config ? countModels(config) : 0), [config])
  const pluginCount = useMemo(() => (config ? countPlugins(config) : 0), [config])
  const currentDecisions = useMemo(() => (config ? getAllDecisions(config) : []), [config])
  const healthyServices = useMemo(
    () => status?.services.filter((s) => s.healthy).length ?? 0,
    [status],
  )
  const totalServices = useMemo(() => status?.services.length ?? 0, [status])
  const modelStatus = useMemo(() => getModelStatusSummary(status), [status])
  const loadedModels = useMemo(() => getLoadedModelCount(status?.models), [status])
  const knownModels = useMemo(() => getTotalKnownModelCount(status?.models), [status])
  const previewModelLimit = 6

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

  if (loading && !config && !status) {
    return (
      <div className={styles.page}>
        <ProductLoadingState label="Preparing your workspace" />
      </div>
    )
  }

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.heroIdentity}>
          <img src="/vllm.png" alt="" aria-hidden="true" />
          <div>
            <span className={styles.eyebrow}>vLLM Semantic Router</span>
            <h1 className={styles.title}>Compose the right model path.</h1>
            <p className={styles.subtitle}>
              Design, connect, and understand every Mixture-of-Models.
            </p>
          </div>
        </div>
        <div className={styles.headerActions}>
          {lastUpdated && (
            <span className={styles.lastUpdated}>Updated {lastUpdated.toLocaleTimeString()}</span>
          )}
          <button
            className={`${styles.refreshBtn} ${refreshing ? styles.refreshBtnSpin : ''}`}
            onClick={() => fetchAll(true)}
            disabled={refreshing}
          >
            <ProductIcon name="refresh" />
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && (
        <div className={styles.errorBanner}>
          <span>Failed to load data: {error}</span>
          <button onClick={() => fetchAll(true)}>Retry</button>
        </div>
      )}

      <div className={styles.statsGrid}>
        <button
          type="button"
          className={styles.statCard}
          onClick={() => navigate('/config/models')}
        >
          <div className={styles.statIcon}>
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <rect x="2" y="3" width="20" height="18" rx="3" />
              <path d="M8 7v10M12 7v10M16 7v10" />
            </svg>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{modelCount}</span>
            <span className={styles.statLabel}>Models</span>
          </div>
          <span className={styles.statArrow}>&rsaquo;</span>
        </button>

        <button
          type="button"
          className={styles.statCard}
          onClick={() => navigate('/config/decisions')}
        >
          <div className={styles.statIcon}>
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <path d="M4 6h16M4 12h8M4 18h12" />
            </svg>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{decisionCount}</span>
            <span className={styles.statLabel}>Decisions</span>
          </div>
          <span className={styles.statArrow}>&rsaquo;</span>
        </button>

        <button
          type="button"
          className={styles.statCard}
          onClick={() => navigate('/config/signals')}
        >
          <div className={styles.statIcon}>
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{signalStats.total}</span>
            <span className={styles.statLabel}>Signals</span>
          </div>
          <span className={styles.statArrow}>&rsaquo;</span>
        </button>

        <button type="button" className={styles.statCard} onClick={() => navigate('/status')}>
          <div
            className={`${styles.statIcon} ${
              status?.overall === 'healthy'
                ? styles.statIconHealthy
                : status?.overall === 'degraded'
                  ? styles.statIconDegraded
                  : styles.statIconDown
            }`}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#fff"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>
              {healthyServices}/{totalServices}
            </span>
            <span className={styles.statLabel}>Services Healthy</span>
          </div>
          <span className={styles.statArrow}>&rsaquo;</span>
        </button>

        <button type="button" className={styles.statCard} onClick={() => navigate('/status')}>
          <div
            className={`${styles.statIcon} ${
              modelStatus.tone === 'ok'
                ? styles.statIconHealthy
                : modelStatus.tone === 'warn'
                  ? styles.statIconStarting
                  : styles.statIconDown
            }`}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#fff"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 3v10" />
              <path d="M8.5 9.5 12 13l3.5-3.5" />
              <path d="M4 19h16" />
            </svg>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{modelStatus.value}</span>
            <span className={styles.statLabel}>Model Status</span>
            <span className={styles.statDetail}>{modelStatus.detail}</span>
          </div>
          <span className={styles.statArrow}>&rsaquo;</span>
        </button>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Intelligence Layers</h2>
            <button
              type="button"
              className={styles.cardAction}
              onClick={() => navigate('/topology')}
            >
              View Full Layers &rsaquo;
            </button>
          </div>
          <div className={styles.flowContainer}>
            {config ? (
              <DashboardMiniFlowDiagram
                signals={signalStats}
                decisions={decisionCount}
                models={modelCount}
                plugins={pluginCount}
              />
            ) : (
              <div className={styles.emptyState}>No configuration loaded</div>
            )}
          </div>
        </div>

        <div className={styles.rightCol}>
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2 className={styles.cardTitle}>System Health</h2>
              <button
                type="button"
                className={styles.cardAction}
                onClick={() => navigate('/status')}
              >
                Details &rsaquo;
              </button>
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
                    {status.version && (
                      <span className={styles.versionBadge}>{status.version}</span>
                    )}
                    {status.deployment_type && status.deployment_type !== 'none' && (
                      <span className={styles.deployBadge}>{status.deployment_type}</span>
                    )}
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
                      <div className={styles.moreServices}>+{status.services.length - 6} more</div>
                    )}
                  </div>
                </>
              ) : (
                <div className={styles.emptyState}>Unable to fetch status</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {config ? (
        <DashboardRoutingProfiles
          config={config}
          onOpenTopology={(scopeId) => navigate(`/topology?scope=${encodeURIComponent(scopeId)}`)}
        />
      ) : null}

      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <div>
            <h2 className={styles.cardTitle}>Loaded Models</h2>
            {knownModels > 0 && (
              <span className={styles.cardSubtitle}>
                {loadedModels}/{knownModels} ready
              </span>
            )}
          </div>
          <button type="button" className={styles.cardAction} onClick={() => navigate('/status')}>
            Status &rsaquo;
          </button>
        </div>
        <RouterModelInventory
          mode="preview"
          previewLimit={previewModelLimit > 0 ? previewModelLimit : undefined}
          modelsInfo={status?.models}
          emptyMessage="Router model inventory will appear here after the router reports its active models."
          onSelectModel={(model) =>
            navigate(`/status#${encodeURIComponent(getRouterModelAnchor(model))}`)
          }
        />
      </div>

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
                Manage &rsaquo;
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
                  +{currentDecisions.length - 10} more decisions &rsaquo;
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default DashboardPage
