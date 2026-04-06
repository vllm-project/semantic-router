import React from 'react'
import styles from './DashboardPage.module.css'
import {
  getDecisionCategory,
  SIGNAL_COLORS,
  type CategorizedDecisions,
  type DecisionRule,
  type SignalStats,
} from './dashboardPageSupport'

interface DashboardBottomPanelsProps {
  categorizedDecisions: CategorizedDecisions
  currentDecisions: DecisionRule[]
  signalStats: SignalStats
  onNavigate: (path: string) => void
}

function formatModelNames(modelRefs: DecisionRule['modelRefs']) {
  if (!Array.isArray(modelRefs)) {
    return '—'
  }

  const modelNames = modelRefs
    .map((modelRef) => {
      if (!modelRef || typeof modelRef !== 'object') {
        return ''
      }
      const candidate = modelRef as { model?: unknown }
      return typeof candidate.model === 'string' ? candidate.model : ''
    })
    .filter(Boolean)
    .join(', ')

  return modelNames || '—'
}

const DashboardBottomPanels: React.FC<DashboardBottomPanelsProps> = ({
  categorizedDecisions,
  currentDecisions,
  signalStats,
  onNavigate,
}) => (
  <div className={styles.bottomGrid}>
    {signalStats.total > 0 && (
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <h2 className={styles.cardTitle}>Signal Breakdown</h2>
          <span className={styles.cardSubtitle}>{signalStats.total} total</span>
        </div>
        <div className={styles.signalBreakdown}>
          {Object.entries(signalStats.byType)
            .sort((a, b) => b[1] - a[1])
            .map(([type, count]) => {
              const maxCount = Math.max(...Object.values(signalStats.byType))
              const pct = Math.round((count / maxCount) * 100)
              const color = SIGNAL_COLORS[type] || '#999'
              return (
                <div
                  key={type}
                  className={styles.breakdownRow}
                  title={`${type}: ${count} signal(s)`}
                >
                  <span className={styles.breakdownLabel}>
                    <span
                      className={styles.breakdownDot}
                      style={{ background: color }}
                    />
                    {type}
                  </span>
                  <div className={styles.breakdownBar}>
                    <div
                      className={styles.breakdownFill}
                      style={{ width: `${pct}%`, background: color }}
                    />
                  </div>
                  <span className={styles.breakdownCount}>{count}</span>
                </div>
              )
            })}
        </div>
      </div>
    )}

    {currentDecisions.length > 0 && (
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <h2 className={styles.cardTitle}>Decisions Overview</h2>
          <button
            className={styles.cardAction}
            onClick={() => onNavigate('/config/decisions')}
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
          {[
            ...categorizedDecisions.guardrails,
            ...categorizedDecisions.routing,
            ...categorizedDecisions.fallbacks,
          ]
            .slice(0, 10)
            .map((decision, index) => {
              const modelNames = formatModelNames(decision.modelRefs)
              const category = getDecisionCategory(decision.priority)
              return (
                <div key={index} className={styles.decisionTableRow}>
                  <span
                    className={styles.decisionName}
                    title={decision.description || decision.name || ''}
                  >
                    {decision.name || `Decision ${index + 1}`}
                  </span>
                  <span className={styles.decisionPriority}>
                    {decision.priority ?? '—'}
                  </span>
                  <span
                    className={`${styles.decisionBadge} ${
                      category === 'guardrail'
                        ? styles.badgeGuardrail
                        : category === 'fallback'
                          ? styles.badgeFallback
                          : styles.badgeRouting
                    }`}
                  >
                    {category === 'guardrail'
                      ? 'Guard'
                      : category === 'fallback'
                        ? 'Default'
                        : 'Route'}
                  </span>
                  <span
                    className={styles.decisionModels}
                    title={modelNames}
                  >
                    {modelNames}
                  </span>
                </div>
              )
            })}
          {currentDecisions.length > 10 && (
            <button
              className={styles.decisionTableMore}
              onClick={() => onNavigate('/config/decisions')}
            >
              +{currentDecisions.length - 10} more decisions &rsaquo;
            </button>
          )}
        </div>
      </div>
    )}
  </div>
)

export default DashboardBottomPanels
