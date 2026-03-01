import React, { useEffect, useState, useCallback, useMemo, useRef } from 'react'
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
  description?: string
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

function countDecisions(cfg: RouterConfig): number {
  return Array.isArray(cfg.decisions) ? cfg.decisions.length : 0
}

function countModels(cfg: RouterConfig): number {
  const models = cfg.providers?.models
  if (Array.isArray(models)) {
    return models.length
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

/** Classify decision by priority range */
function getDecisionCategory(priority?: number): 'guardrail' | 'routing' | 'fallback' {
  if (priority == null) return 'routing'
  if (priority >= 999) return 'guardrail'
  if (priority <= 100) return 'fallback'
  return 'routing'
}

/* ------------------------------------------------------------------ */
/*  Mini Flow Diagram (pure SVG, no dependency)                        */
/* ------------------------------------------------------------------ */

interface FlowProps {
  signals: { total: number; byType: Record<string, number> }
  decisions: number
  models: number
  plugins: number
}

const SIGNAL_COLORS: Record<string, string> = {
  keywords: '#4EC9B0',
  embeddings: '#9CDCFE',
  domains: '#DCDCAA',
  fact_check: '#CE9178',
  user_feedbacks: '#C586C0',
  preferences: '#4FC1FF',
  language: '#B5CEA8',
  context: '#D7BA7D',
  complexity: '#569CD6',
  modality: '#D4D4D4',
  authz: '#F48771',
  jailbreak: '#F48771',
  pii: '#FF6B6B',
}

const MiniFlowDiagram: React.FC<FlowProps> = React.memo(({ signals, decisions, models, plugins }) => {
  const signalTypes = Object.entries(signals.byType).sort((a, b) => b[1] - a[1])
  const visibleSignals = signalTypes.slice(0, 7)
  const hiddenCount = signalTypes.length - visibleSignals.length
  const rowH = 34
  const sH = Math.max(visibleSignals.length * rowH + (hiddenCount > 0 ? 28 : 0) + 30, 180)
  const height = Math.max(sH, 220)

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
      {visibleSignals.map(([type, count], i) => {
        const y = 16 + i * rowH
        const color = SIGNAL_COLORS[type] || '#999'
        const endY = y + 14
        // bezier curve from signal to decision
        const cx1 = colSignal + 52 + 40
        const cx2 = colDecision - 50 - 40
        return (
          <g key={type} className={styles.flowNode}>
            <rect x={colSignal - 55} y={y} width={110} height={26} rx={6} fill={color + '18'} stroke={color} strokeWidth={1} />
            <text x={colSignal} y={y + 17} textAnchor="middle" fill={color} fontSize={10.5} fontFamily="var(--font-mono)">
              {type} ({count})
            </text>
            <path
              d={`M ${colSignal + 55} ${endY} C ${cx1} ${endY}, ${cx2} ${midY}, ${colDecision - 52} ${midY}`}
              fill="none" stroke="var(--color-border-hover)" strokeWidth={1} opacity={0.35}
              markerEnd="url(#arrow)"
            />
          </g>
        )
      })}

      {/* Truncation hint */}
      {hiddenCount > 0 && (
        <text
          x={colSignal} y={16 + visibleSignals.length * rowH + 14}
          textAnchor="middle" fill="var(--color-text-muted)" fontSize={10} fontStyle="italic"
        >
          +{hiddenCount} more
        </text>
      )}

      {/* Decision Engine box */}
      <rect x={colDecision - 52} y={midY - 30} width={104} height={60} rx={10}
        fill="var(--color-primary)" fillOpacity={0.12} stroke="var(--color-primary)" strokeWidth={1.5} />
      <text x={colDecision} y={midY - 6} textAnchor="middle" fill="var(--color-primary)" fontSize={11} fontWeight="bold">Decision</text>
      <text x={colDecision} y={midY + 12} textAnchor="middle" fill="var(--color-primary)" fontSize={10.5} opacity={0.85}>{decisions} layers</text>

      {/* Connector Decision → Models */}
      <line
        x1={colDecision + 54} y1={midY}
        x2={colModel - 54} y2={midY}
        stroke="var(--color-border-hover)" strokeWidth={1.5}
        markerEnd="url(#arrow)"
      />

      {/* Model box */}
      <rect x={colModel - 52} y={midY - 30} width={104} height={60} rx={10}
        fill="var(--color-accent-cyan)" fillOpacity={0.10} stroke="var(--color-accent-cyan)" strokeWidth={1.5} />
      <text x={colModel} y={midY - 6} textAnchor="middle" fill="var(--color-accent-cyan)" fontSize={11} fontWeight="bold">Models</text>
      <text x={colModel} y={midY + 12} textAnchor="middle" fill="var(--color-accent-cyan)" fontSize={10.5} opacity={0.85}>{models} models</text>

      {/* Plugins badge */}
      {plugins > 0 && (
        <g>
          <rect x={colDecision - 30} y={midY + 40} width={60} height={22} rx={11}
            fill="var(--color-accent-purple)" fillOpacity={0.15} stroke="var(--color-accent-purple)" strokeWidth={1} />
          <text x={colDecision} y={midY + 55} textAnchor="middle" fill="var(--color-accent-purple)" fontSize={10}>{plugins} plugins</text>
        </g>
      )}

      {/* Column labels */}
      <text x={colSignal} y={height - 4} textAnchor="middle" fill="var(--color-text-muted)" fontSize={9} letterSpacing="0.05em">SIGNALS</text>
      <text x={colDecision} y={height - 4} textAnchor="middle" fill="var(--color-text-muted)" fontSize={9} letterSpacing="0.05em">DECISIONS</text>
      <text x={colModel} y={height - 4} textAnchor="middle" fill="var(--color-text-muted)" fontSize={9} letterSpacing="0.05em">MODELS</text>
    </svg>
  )
})

MiniFlowDiagram.displayName = 'MiniFlowDiagram'

/* ------------------------------------------------------------------ */
/*  Main Component                                                     */
/* ------------------------------------------------------------------ */

const DashboardPage: React.FC = () => {
  const navigate = useNavigate()

  const [config, setConfig] = useState<RouterConfig | null>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const configTickRef = useRef(0)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/status')
      if (res.ok) {
        setStatus(await res.json())
      }
    } catch { /* silent */ }
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
    // Config changes rarely — poll every 30s; status every 10s
    const statusInterval = setInterval(fetchStatus, 10000)
    const configInterval = setInterval(() => {
      configTickRef.current += 1
      if (configTickRef.current % 3 === 0) {
        fetchAll()
      } else {
        fetchStatus()
      }
    }, 10000)
    return () => {
      clearInterval(statusInterval)
      clearInterval(configInterval)
    }
  }, [fetchAll, fetchStatus])

  const signalStats = useMemo(() => config ? countSignals(config) : { total: 0, byType: {} }, [config])
  const decisionCount = useMemo(() => config ? countDecisions(config) : 0, [config])
  const modelCount = useMemo(() => config ? countModels(config) : 0, [config])
  const pluginCount = useMemo(() => config ? countPlugins(config) : 0, [config])
  const healthyServices = useMemo(() => status?.services.filter(s => s.healthy).length ?? 0, [status])
  const totalServices = useMemo(() => status?.services.length ?? 0, [status])

  // Categorize decisions for the table
  const categorizedDecisions = useMemo(() => {
    if (!config?.decisions) return { guardrails: [], routing: [], fallbacks: [] }
    const guardrails: DecisionRule[] = []
    const routing: DecisionRule[] = []
    const fallbacks: DecisionRule[] = []
    for (const d of config.decisions) {
      const cat = getDecisionCategory(d.priority)
      if (cat === 'guardrail') guardrails.push(d)
      else if (cat === 'fallback') fallbacks.push(d)
      else routing.push(d)
    }
    return { guardrails, routing, fallbacks }
  }, [config])

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
          <p className={styles.subtitle}>Building the System Intelligence</p>
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

        <button className={styles.statCard} onClick={() => navigate('/config/decisions')}>
          <div className={styles.statIcon} style={{ background: 'var(--color-accent-cyan)', boxShadow: 'var(--glow-cyan)' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round">
              <path d="M4 6h16M4 12h8M4 18h12" />
            </svg>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{decisionCount}</span>
            <span className={styles.statLabel}>Decisions</span>
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
            <span className={styles.statLabel}>Models</span>
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
            <h2 className={styles.cardTitle}>Intelligence Layers</h2>
            <button className={styles.cardAction} onClick={() => navigate('/topology')}>
              View Full Layers &rsaquo;
            </button>
          </div>
          <div className={styles.flowContainer}>
            {config ? (
              <MiniFlowDiagram
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
              <button className={styles.quickLink} onClick={() => navigate('/config/signals')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M12 20V10M18 20V4M6 20v-4" />
                </svg>
                Manage Signals
              </button>
              <button className={styles.quickLink} onClick={() => navigate('/config/models')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <rect x="2" y="3" width="20" height="18" rx="3" /><path d="M8 7v10M16 7v10" />
                </svg>
                Manage Models
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Signal Breakdown + Decisions Overview — 2 column layout */}
      <div className={styles.bottomGrid}>
        {/* Signal Breakdown */}
        {signalStats.total > 0 && (
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2 className={styles.cardTitle}>Signal Breakdown</h2>
              <span className={styles.cardSubtitle}>{signalStats.total} total</span>
            </div>
            <div className={styles.signalBreakdown}>
              {Object.entries(signalStats.byType).sort((a, b) => b[1] - a[1]).map(([type, count]) => {
                const maxCount = Math.max(...Object.values(signalStats.byType))
                const pct = Math.round((count / maxCount) * 100)
                const color = SIGNAL_COLORS[type] || '#999'
                return (
                  <div key={type} className={styles.breakdownRow} title={`${type}: ${count} signal(s)`}>
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

        {/* Decisions Overview Table */}
        {config?.decisions && config.decisions.length > 0 && (
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2 className={styles.cardTitle}>Decisions Overview</h2>
              <button className={styles.cardAction} onClick={() => navigate('/config/decisions')}>
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
              {/* Guardrails first, then routing, then fallbacks — show top 10 */}
              {[...categorizedDecisions.guardrails, ...categorizedDecisions.routing, ...categorizedDecisions.fallbacks]
                .slice(0, 10)
                .map((d, i) => {
                  const modelNames = Array.isArray(d.modelRefs)
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    ? d.modelRefs.map((m: any) => m?.model || '').filter(Boolean).join(', ')
                    : '—'
                  const cat = getDecisionCategory(d.priority)
                  return (
                    <div key={i} className={styles.decisionTableRow}>
                      <span className={styles.decisionName} title={d.description || d.name || ''}>
                        {d.name || `Decision ${i + 1}`}
                      </span>
                      <span className={styles.decisionPriority}>{d.priority ?? '—'}</span>
                      <span className={`${styles.decisionBadge} ${
                        cat === 'guardrail' ? styles.badgeGuardrail :
                        cat === 'fallback' ? styles.badgeFallback :
                        styles.badgeRouting
                      }`}>
                        {cat === 'guardrail' ? 'Guard' : cat === 'fallback' ? 'Default' : 'Route'}
                      </span>
                      <span className={styles.decisionModels} title={modelNames}>{modelNames}</span>
                    </div>
                  )
                })}
              {config.decisions.length > 10 && (
                <button className={styles.decisionTableMore} onClick={() => navigate('/config/decisions')}>
                  +{config.decisions.length - 10} more decisions &rsaquo;
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
