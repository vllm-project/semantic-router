import React, { useEffect, useState, useCallback, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import styles from './DashboardPage.module.css'

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface ServiceStatus {
  name: string
  status: string
  healthy: boolean
  message?: string
  component?: string
}

interface SystemStatus {
  overall: string
  deployment_type: string
  services: ServiceStatus[]
  version?: string
}

interface SignalConfig {
  name?: string
  type?: string
  [key: string]: unknown
}

interface DecisionRule {
  name?: string
  priority?: number
  rules?: unknown[]
  modelRefs?: unknown[]
  [key: string]: unknown
}

interface RouterConfig {
  signals?: Record<string, SignalConfig[]>
  decisions?: DecisionRule[]
  providers?: {
    default_model?: string
    models?: Array<{
      name?: string
      endpoints?: Array<{ name?: string }>
      preferred_endpoints?: string[]
      [key: string]: unknown
    }>
    vllm_endpoints?: unknown[]
    [key: string]: unknown
  }
  vllm_endpoints?: Array<{ name?: string }>
  plugins?: Record<string, unknown>
  global?: Record<string, unknown>
  [key: string]: unknown
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function countSignals(cfg: RouterConfig): { total: number; byType: Record<string, number> } {
  const byType: Record<string, number> = {}
  let total = 0
  if (cfg.signals) {
    for (const [type, arr] of Object.entries(cfg.signals)) {
      if (Array.isArray(arr)) {
        byType[type] = arr.length
        total += arr.length
      }
    }
  }
  return { total, byType }
}

function countRoutes(cfg: RouterConfig): number {
  return Array.isArray(cfg.decisions) ? cfg.decisions.length : 0
}

function countModels(cfg: RouterConfig): number {
  const models = cfg.providers?.models
  if (Array.isArray(models) && models.length > 0) {
    const endpointSet = new Set<string>()

    models.forEach((model, modelIndex) => {
      if (Array.isArray(model.endpoints)) {
        model.endpoints.forEach((ep, epIndex) => {
          const key = typeof ep?.name === 'string' && ep.name.trim().length > 0
            ? ep.name
            : `${model.name || `model-${modelIndex}`}-${epIndex}`
          endpointSet.add(key)
        })
      }

      if (Array.isArray(model.preferred_endpoints)) {
        model.preferred_endpoints.forEach((name) => {
          if (typeof name === 'string' && name.trim().length > 0) {
            endpointSet.add(name)
          }
        })
      }
    })

    return endpointSet.size
  }

  const legacyRootEndpoints = cfg.vllm_endpoints
  if (Array.isArray(legacyRootEndpoints)) return legacyRootEndpoints.length

  const legacyProviderEndpoints = cfg.providers?.vllm_endpoints
  return Array.isArray(legacyProviderEndpoints) ? legacyProviderEndpoints.length : 0
}

function countPlugins(cfg: RouterConfig): number {
  if (!cfg.plugins || typeof cfg.plugins !== 'object') return 0
  return Object.keys(cfg.plugins).length
}

/* ------------------------------------------------------------------ */
/*  Mini Flow Diagram (pure SVG, no dependency)                        */
/* ------------------------------------------------------------------ */

interface FlowProps {
  signals: { total: number; byType: Record<string, number> }
  routes: number
  models: number
  plugins: number
}

const SIGNAL_COLORS: Record<string, string> = {
  keyword: '#4EC9B0',
  embedding: '#9CDCFE',
  domain: '#DCDCAA',
  fact_check: '#CE9178',
  user_feedback: '#C586C0',
  preference: '#4FC1FF',
  language: '#B5CEA8',
  context: '#D7BA7D',
  complexity: '#569CD6',
  modality: '#D4D4D4',
  authz: '#F48771',
}

const MiniFlowDiagram: React.FC<FlowProps> = ({ signals, routes, models, plugins }) => {
  const signalTypes = Object.entries(signals.byType).sort((a, b) => b[1] - a[1]).slice(0, 6)
  const sH = Math.max(signalTypes.length * 36 + 20, 160)
  const height = Math.max(sH, 200)

  // Column x positions
  const colSignal = 90
  const colDecision = 310
  const colModel = 530
  const midY = height / 2

  return (
    <svg
      viewBox={`0 0 620 ${height}`}
      className={styles.flowSvg}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-text-muted)" />
        </marker>
      </defs>

      {/* Signal nodes */}
      {signalTypes.map(([type, count], i) => {
        const y = 20 + i * 36
        const color = SIGNAL_COLORS[type] || '#999'
        return (
          <g key={type}>
            <rect x={colSignal - 50} y={y} width={100} height={28} rx={6} fill={color + '22'} stroke={color} strokeWidth={1} />
            <text x={colSignal} y={y + 18} textAnchor="middle" fill={color} fontSize={11} fontFamily="var(--font-mono)">
              {type} ({count})
            </text>
            {/* connector */}
            <line
              x1={colSignal + 52} y1={y + 14}
              x2={colDecision - 50} y2={midY}
              stroke="var(--color-border-hover)" strokeWidth={1} opacity={0.4}
              markerEnd="url(#arrow)"
            />
          </g>
        )
      })}

      {/* Decision Engine box */}
      <rect x={colDecision - 50} y={midY - 28} width={100} height={56} rx={8} fill="var(--color-primary)" fillOpacity={0.15} stroke="var(--color-primary)" strokeWidth={1.5} />
      <text x={colDecision} y={midY - 6} textAnchor="middle" fill="var(--color-primary)" fontSize={11} fontWeight="bold">Decision</text>
      <text x={colDecision} y={midY + 12} textAnchor="middle" fill="var(--color-primary)" fontSize={11}>{routes} routes</text>

      {/* Connector Decision → Models */}
      <line
        x1={colDecision + 52} y1={midY}
        x2={colModel - 50} y2={midY}
        stroke="var(--color-border-hover)" strokeWidth={1.5}
        markerEnd="url(#arrow)"
      />

      {/* Model box */}
      <rect x={colModel - 50} y={midY - 28} width={100} height={56} rx={8} fill="var(--color-accent-cyan)" fillOpacity={0.12} stroke="var(--color-accent-cyan)" strokeWidth={1.5} />
      <text x={colModel} y={midY - 6} textAnchor="middle" fill="var(--color-accent-cyan)" fontSize={11} fontWeight="bold">Models</text>
      <text x={colModel} y={midY + 12} textAnchor="middle" fill="var(--color-accent-cyan)" fontSize={11}>{models} endpoints</text>

      {/* Plugins badge (bottom-right) */}
      {plugins > 0 && (
        <g>
          <rect x={colDecision - 30} y={midY + 38} width={60} height={22} rx={11} fill="var(--color-accent-purple)" fillOpacity={0.18} stroke="var(--color-accent-purple)" strokeWidth={1} />
          <text x={colDecision} y={midY + 53} textAnchor="middle" fill="var(--color-accent-purple)" fontSize={10}>{plugins} plugins</text>
        </g>
      )}

      {/* Column labels */}
      <text x={colSignal} y={height - 4} textAnchor="middle" fill="var(--color-text-muted)" fontSize={9}>SIGNALS</text>
      <text x={colDecision} y={height - 4} textAnchor="middle" fill="var(--color-text-muted)" fontSize={9}>DECISIONS</text>
      <text x={colModel} y={height - 4} textAnchor="middle" fill="var(--color-text-muted)" fontSize={9}>MODELS</text>
    </svg>
  )
}

/* ------------------------------------------------------------------ */
/*  Main Component                                                     */
/* ------------------------------------------------------------------ */

const DashboardPage: React.FC = () => {
  const navigate = useNavigate()

  const [config, setConfig] = useState<RouterConfig | null>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const fetchAll = useCallback(async () => {
    try {
      const [cfgRes, statusRes] = await Promise.all([
        fetch('/api/router/config/all'),
        fetch('/api/status'),
      ])
      if (cfgRes.ok) {
        const cfgData = await cfgRes.json()
        setConfig(cfgData)
      }
      if (statusRes.ok) {
        const statusData = await statusRes.json()
        setStatus(statusData)
      }
      setLastUpdated(new Date())
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAll()
    const interval = setInterval(fetchAll, 15000)
    return () => clearInterval(interval)
  }, [fetchAll])

  const signalStats = useMemo(() => config ? countSignals(config) : { total: 0, byType: {} }, [config])
  const routeCount = useMemo(() => config ? countRoutes(config) : 0, [config])
  const modelCount = useMemo(() => config ? countModels(config) : 0, [config])
  const pluginCount = useMemo(() => config ? countPlugins(config) : 0, [config])
  const healthyServices = useMemo(() => status?.services.filter(s => s.healthy).length ?? 0, [status])
  const totalServices = useMemo(() => status?.services.length ?? 0, [status])

  const goToRoutesConfig = () => navigate('/config/decisions')

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
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Dashboard</h1>
          <p className={styles.subtitle}>Configuration overview & system health</p>
        </div>
        <div className={styles.headerActions}>
          {lastUpdated && (
            <span className={styles.lastUpdated}>
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button className={styles.refreshBtn} onClick={fetchAll}>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M14 8A6 6 0 1 1 8 2" strokeLinecap="round" />
              <path d="M14 2v6h-6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className={styles.errorBanner}>
          <span>Failed to load data: {error}</span>
          <button onClick={fetchAll}>Retry</button>
        </div>
      )}

      {/* Stats Cards */}
      <div className={styles.statsGrid}>
        <button className={styles.statCard} onClick={() => navigate('/config/signals')}>
          <div className={styles.statIcon} style={{ background: 'var(--color-primary)', boxShadow: 'var(--glow-primary)' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round">
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

        <button className={styles.statCard} onClick={goToRoutesConfig}>
          <div className={styles.statIcon} style={{ background: 'var(--color-accent-cyan)', boxShadow: 'var(--glow-cyan)' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round">
              <path d="M4 6h16M4 12h8M4 18h12" />
            </svg>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{routeCount}</span>
            <span className={styles.statLabel}>Routes</span>
          </div>
          <span className={styles.statArrow}>&rsaquo;</span>
        </button>

        <button className={styles.statCard} onClick={() => navigate('/config/models')}>
          <div className={styles.statIcon} style={{ background: 'var(--color-accent-purple)', boxShadow: 'var(--glow-purple)' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round">
              <rect x="2" y="3" width="20" height="18" rx="3" />
              <path d="M8 7v10M12 7v10M16 7v10" />
            </svg>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{modelCount}</span>
            <span className={styles.statLabel}>Endpoints</span>
          </div>
          <span className={styles.statArrow}>&rsaquo;</span>
        </button>

        <button className={styles.statCard} onClick={() => navigate('/status')}>
          <div className={`${styles.statIcon} ${
            status?.overall === 'healthy' ? styles.statIconHealthy :
            status?.overall === 'degraded' ? styles.statIconDegraded :
            styles.statIconDown
          }`}>
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
      </div>

      {/* Main content: 2-column */}
      <div className={styles.mainGrid}>
        {/* Left: Flow Diagram */}
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Routing Pipeline</h2>
            <button className={styles.cardAction} onClick={() => navigate('/topology')}>
              View Full Topology &rsaquo;
            </button>
          </div>
          <div className={styles.flowContainer}>
            {config ? (
              <MiniFlowDiagram
                signals={signalStats}
                routes={routeCount}
                models={modelCount}
                plugins={pluginCount}
              />
            ) : (
              <div className={styles.emptyState}>No configuration loaded</div>
            )}
          </div>
        </div>

        {/* Right: Health + Quick Info */}
        <div className={styles.rightCol}>
          {/* Health Card */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2 className={styles.cardTitle}>System Health</h2>
              <button className={styles.cardAction} onClick={() => navigate('/status')}>
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
                    }`} />
                    <span className={styles.healthLabel}>
                      {status.overall === 'not_running' ? 'Not Running' :
                       status.overall.charAt(0).toUpperCase() + status.overall.slice(1)}
                    </span>
                    {status.version && <span className={styles.versionBadge}>v{status.version}</span>}
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

          {/* Quick Links */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2 className={styles.cardTitle}>Quick Actions</h2>
            </div>
            <div className={styles.quickLinks}>
              <button className={styles.quickLink} onClick={() => navigate('/playground')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                </svg>
                Test in Playground
              </button>
              <button className={styles.quickLink} onClick={() => navigate('/builder')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
                </svg>
                Open Builder
              </button>
              <button className={styles.quickLink} onClick={() => navigate('/topology')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" />
                </svg>
                View Topology
              </button>
              <button className={styles.quickLink} onClick={() => navigate('/evaluation')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M9 11l3 3L22 4" /><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" />
                </svg>
                Run Evaluation
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Signal Breakdown */}
      {signalStats.total > 0 && (
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Signal Breakdown</h2>
          </div>
          <div className={styles.signalBreakdown}>
            {Object.entries(signalStats.byType).sort((a, b) => b[1] - a[1]).map(([type, count]) => {
              const pct = Math.round((count / signalStats.total) * 100)
              const color = SIGNAL_COLORS[type] || '#999'
              return (
                <div key={type} className={styles.breakdownRow}>
                  <span className={styles.breakdownLabel}>
                    <span className={styles.breakdownDot} style={{ background: color }} />
                    {type}
                  </span>
                  <div className={styles.breakdownBar}>
                    <div className={styles.breakdownFill} style={{ width: `${pct}%`, background: color }} />
                  </div>
                  <span className={styles.breakdownCount}>{count}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Route Overview Table */}
      {config?.decisions && config.decisions.length > 0 && (
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Routes Overview</h2>
            <button className={styles.cardAction} onClick={() => navigate('/config/decisions')}>
              Manage &rsaquo;
            </button>
          </div>
          <div className={styles.routeTable}>
            <div className={styles.routeTableHead}>
              <span>Name</span>
              <span>Priority</span>
              <span>Models</span>
            </div>
            {config.decisions.slice(0, 8).map((d, i) => {
              const modelNames = Array.isArray(d.modelRefs)
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                ? d.modelRefs.map((m: any) => m?.model || '').filter(Boolean).join(', ')
                : '—'
              return (
                <div key={i} className={styles.routeTableRow}>
                  <span className={styles.routeName}>{d.name || `Route ${i + 1}`}</span>
                  <span className={styles.routePriority}>{d.priority ?? '—'}</span>
                  <span className={styles.routeModels}>{modelNames}</span>
                </div>
              )
            })}
            {config.decisions.length > 8 && (
              <div className={styles.routeTableMore}>
                +{config.decisions.length - 8} more routes
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default DashboardPage
