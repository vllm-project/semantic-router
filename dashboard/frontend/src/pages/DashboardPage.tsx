import React, { useEffect, useState, useCallback, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import RouterModelInventory from '../components/RouterModelInventory'
import { useAuth } from '../contexts/AuthContext'
import { canAccessMLSetup } from '../utils/accessControl'
import {
  getLoadedModelCount,
  getModelStatusSummary,
  getRouterModelAnchor,
  getTotalKnownModelCount,
  type SystemStatus,
} from '../utils/routerRuntime'
import { DashboardMiniFlowDiagram } from './DashboardMiniFlowDiagram'
import type { RouterConfig } from './dashboardPageTypes'
import {
  categorizeDecisions,
  countDecisions,
  countModels,
  countPlugins,
  countSignals,
} from './dashboardPageStats'
import { buildDecisionPreviewRows, buildSignalBreakdownRows } from './dashboardPageOverview'
import styles from './DashboardPage.module.css'

const DashboardPage: React.FC = () => {
  const navigate = useNavigate()
  const { user } = useAuth()

  const [config, setConfig] = useState<RouterConfig | null>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const configTickRef = useRef(0)

  const fetchStatus = useCallback(async () => {
    try {
      const statusRes = await fetch('/api/status')
      if (statusRes.ok) {
        setStatus(await statusRes.json())
      }
    } catch {
      // Ignore transient polling errors.
    }
  }, [])

  const fetchAll = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true)
    try {
      const [cfgRes, statusRes] = await Promise.all([
        fetch('/api/router/config/all'),
        fetch('/api/status'),
      ])
      if (cfgRes.ok) {
        setConfig(await cfgRes.json())
      }
      if (statusRes.ok) {
        setStatus(await statusRes.json())
      }
      setLastUpdated(new Date())
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    fetchAll()
    const statusInterval = setInterval(fetchStatus, 10000)
    const configInterval = setInterval(() => {
      configTickRef.current += 1
      if (configTickRef.current % 3 === 0) {
        fetchAll()
      } else {
        fetchStatus()
      }
    }, 10000)
    const onConfigDeployed = () => fetchAll()
    window.addEventListener('config-deployed', onConfigDeployed)
    return () => {
      clearInterval(statusInterval)
      clearInterval(configInterval)
      window.removeEventListener('config-deployed', onConfigDeployed)
    }
  }, [fetchAll, fetchStatus])

  const signalStats = useMemo(() => (config ? countSignals(config) : { total: 0, byType: {} }), [config])
  const decisionCount = useMemo(() => (config ? countDecisions(config) : 0), [config])
  const modelCount = useMemo(() => (config ? countModels(config) : 0), [config])
  const pluginCount = useMemo(() => (config ? countPlugins(config) : 0), [config])
  const currentDecisions = useMemo(() => config?.routing?.decisions ?? config?.decisions ?? [], [config])
  const healthyServices = useMemo(() => status?.services.filter(s => s.healthy).length ?? 0, [status])
  const showMLSetupQuickLink = canAccessMLSetup(user)
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
    () => buildDecisionPreviewRows([
      ...categorizedDecisions.guardrails,
      ...categorizedDecisions.routing,
      ...categorizedDecisions.fallbacks,
    ]),
    [categorizedDecisions],
  )

  if (loading && !config && !status) {
    return (
      <div className={styles.page}>
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Dashboard</h1>
          <p className={styles.subtitle}>Routing health, policy coverage, and model readiness</p>
        </div>
        <div className={styles.headerActions}>
          {lastUpdated && (
            <span className={styles.lastUpdated}>
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            className={`${styles.refreshBtn} ${refreshing ? styles.refreshBtnSpin : ''}`}
            onClick={() => fetchAll(true)}
            disabled={refreshing}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M14 8A6 6 0 1 1 8 2" strokeLinecap="round" />
              <path d="M14 2v6h-6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
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
        <button type="button" className={styles.statCard} onClick={() => navigate('/config/models')}>
          <div className={styles.statIcon}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
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

        <button type="button" className={styles.statCard} onClick={() => navigate('/config/decisions')}>
          <div className={styles.statIcon}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M4 6h16M4 12h8M4 18h12" />
            </svg>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{decisionCount}</span>
            <span className={styles.statLabel}>Decisions</span>
          </div>
          <span className={styles.statArrow}>&rsaquo;</span>
        </button>

        <button type="button" className={styles.statCard} onClick={() => navigate('/config/signals')}>
          <div className={styles.statIcon}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
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
          <div className={`${styles.statIcon} ${
            status?.overall === 'healthy' ? styles.statIconHealthy :
            status?.overall === 'degraded' ? styles.statIconDegraded :
            styles.statIconDown
          }`}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{healthyServices}/{totalServices}</span>
            <span className={styles.statLabel}>Services Healthy</span>
          </div>
          <span className={styles.statArrow}>&rsaquo;</span>
        </button>

        <button type="button" className={styles.statCard} onClick={() => navigate('/status')}>
          <div className={`${styles.statIcon} ${
            modelStatus.tone === 'ok' ? styles.statIconHealthy :
            modelStatus.tone === 'warn' ? styles.statIconStarting :
            styles.statIconDown
          }`}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
            <button type="button" className={styles.cardAction} onClick={() => navigate('/topology')}>
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
              <button type="button" className={styles.cardAction} onClick={() => navigate('/status')}>
                Details &rsaquo;
              </button>
            </div>
            <div className={styles.healthContent}>
              {status ? (
                <>
                  <div className={styles.healthOverall}>
                    <span className={`${styles.healthDot} ${
                      status.overall === 'healthy' ? styles.healthDotGreen :
                      status.overall === 'degraded' ? styles.healthDotYellow :
                      styles.healthDotRed
                    }`}
                    />
                    <span className={styles.healthLabel}>
                      {status.overall === 'not_running' ? 'Not Running' :
                       status.overall.charAt(0).toUpperCase() + status.overall.slice(1)}
                    </span>
                    {status.version && <span className={styles.versionBadge}>{status.version}</span>}
                    {status.deployment_type && status.deployment_type !== 'none' && (
                      <span className={styles.deployBadge}>{status.deployment_type}</span>
                    )}
                  </div>
                  <div className={styles.servicesList}>
                    {status.services.slice(0, 6).map((svc, i) => (
                      <div key={i} className={styles.serviceRow}>
                        <span className={`${styles.svcDot} ${svc.healthy ? styles.svcDotOk : styles.svcDotFail}`} />
                        <span className={styles.svcName}>{svc.name}</span>
                        <span className={`${styles.svcStatus} ${svc.healthy ? styles.svcStatusOk : styles.svcStatusFail}`}>
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

          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2 className={styles.cardTitle}>Quick Actions</h2>
            </div>
            <div className={styles.quickLinks}>
              <button type="button" className={styles.quickLink} onClick={() => navigate('/playground')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                </svg>
                Test in Playground
              </button>
              <button type="button" className={styles.quickLink} onClick={() => navigate('/builder')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
                </svg>
                Open Builder
              </button>
              <button type="button" className={styles.quickLink} onClick={() => navigate('/topology')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" />
                </svg>
                View Topology
              </button>
              <button type="button" className={styles.quickLink} onClick={() => navigate('/evaluation')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M9 11l3 3L22 4" /><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" />
                </svg>
                Run Evaluation
              </button>
              {showMLSetupQuickLink ? (
                <button type="button" className={styles.quickLink} onClick={() => navigate('/ml-setup')}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" />
                  </svg>
                  ML Setup
                </button>
              ) : null}
              <button type="button" className={styles.quickLink} onClick={() => navigate('/config/models')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <rect x="2" y="3" width="20" height="18" rx="3" /><path d="M8 7v10M16 7v10" />
                </svg>
                Manage Models
              </button>
              <button type="button" className={styles.quickLink} onClick={() => navigate('/config/decisions')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M4 6h16M4 12h8M4 18h12" />
                </svg>
                Manage Decisions
              </button>
              <button type="button" className={styles.quickLink} onClick={() => navigate('/config/signals')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M12 20V10M18 20V4M6 20v-4" />
                </svg>
                Manage Signals
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <div>
            <h2 className={styles.cardTitle}>Loaded Models</h2>
            {knownModels > 0 && (
              <span className={styles.cardSubtitle}>{loadedModels}/{knownModels} ready</span>
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
          onSelectModel={(model) => navigate(`/status#${encodeURIComponent(getRouterModelAnchor(model))}`)}
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
                <div key={row.type} className={styles.breakdownRow} title={`${row.type}: ${row.count} signal(s)`}>
                  <span className={styles.breakdownLabel}>
                    <span className={styles.breakdownDot} style={{ background: row.color }} />
                    {row.type}
                  </span>
                  <div className={styles.breakdownBar}>
                    <div className={styles.breakdownFill} style={{ width: `${row.percent}%`, background: row.color }} />
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
              <button type="button" className={styles.cardAction} onClick={() => navigate('/config/decisions')}>
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
                  <span className={styles.decisionName} title={row.title}>
                    {row.name}
                  </span>
                  <span className={styles.decisionPriority}>{row.priorityLabel}</span>
                  <span className={`${styles.decisionBadge} ${
                    row.category === 'guardrail' ? styles.badgeGuardrail :
                    row.category === 'fallback' ? styles.badgeFallback :
                    styles.badgeRouting
                  }`}
                  >
                    {row.typeLabel}
                  </span>
                  <span className={styles.decisionModels} title={row.modelNames}>{row.modelNames}</span>
                </div>
              ))}
              {currentDecisions.length > 10 && (
                <button type="button" className={styles.decisionTableMore} onClick={() => navigate('/config/decisions')}>
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
