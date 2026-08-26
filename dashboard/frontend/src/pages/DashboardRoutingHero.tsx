import ProductIcon from '../components/ProductIcon'
import styles from './DashboardRoutingHero.module.css'

interface DashboardRoutingHeroProps {
  modelCount: number
  signalCount: number
  decisionCount: number
  apiKeyCount: string | null
  showRoutingMetrics: boolean
  showAPIKeyMetric: boolean
  showPlaygroundAction: boolean
  showStatus: boolean
  overallStatus?: string
  refreshing: boolean
  lastUpdated: Date | null
  onRefresh: () => void
  onNavigate: (path: string) => void
}

export default function DashboardRoutingHero({
  modelCount,
  signalCount,
  decisionCount,
  apiKeyCount,
  showRoutingMetrics,
  showAPIKeyMetric,
  showPlaygroundAction,
  showStatus,
  overallStatus,
  refreshing,
  lastUpdated,
  onRefresh,
  onNavigate,
}: DashboardRoutingHeroProps) {
  const healthy = overallStatus === 'healthy'

  return (
    <section className={styles.hero} aria-labelledby="dashboard-routing-title">
      <header className={styles.banner}>
        <div className={styles.bannerCopy}>
          <div className={styles.eyebrowRow}>
            <span className={styles.eyebrow}>Mixture-of-Models control plane</span>
            {showStatus ? (
              <button
                type="button"
                className={`${styles.liveStatus} ${healthy ? styles.liveHealthy : ''}`}
                onClick={onRefresh}
                disabled={refreshing}
                aria-label="Refresh dashboard status"
                title={
                  lastUpdated
                    ? `Last checked ${lastUpdated.toLocaleTimeString()}`
                    : 'Check dashboard status'
                }
              >
                <i /> {refreshing ? 'Checking' : healthy ? 'Live' : overallStatus || 'Check'}
              </button>
            ) : null}
          </div>
          <h1 id="dashboard-routing-title">Your model system, at a glance.</h1>
          <p>One stable API. Every capability path visible, governed, and ready.</p>
        </div>
        {showPlaygroundAction ? (
          <div className={styles.bannerActions}>
            <button
              type="button"
              className={styles.primaryAction}
              onClick={() => onNavigate('/playground')}
            >
              Try a request <ProductIcon name="arrow-right" aria-hidden="true" />
            </button>
          </div>
        ) : null}
      </header>

      {showRoutingMetrics || showAPIKeyMetric ? (
        <div
          className={`${styles.metricStrip} ${!showRoutingMetrics ? styles.metricStripCompact : ''}`}
        >
          {showRoutingMetrics ? (
            <>
              <button type="button" onClick={() => onNavigate('/config/models')}>
                <strong>{modelCount}</strong>
                <span>Models</span>
              </button>
              <button type="button" onClick={() => onNavigate('/config/signals')}>
                <strong>{signalCount}</strong>
                <span>Signals</span>
              </button>
              <button type="button" onClick={() => onNavigate('/config/decisions')}>
                <strong>{decisionCount}</strong>
                <span>Decisions</span>
              </button>
            </>
          ) : null}
          {showAPIKeyMetric ? (
            <button type="button" onClick={() => onNavigate('/access/api-keys')}>
              <strong>
                {apiKeyCount === null
                  ? '—'
                  : new Intl.NumberFormat('en-US').format(BigInt(apiKeyCount))}
              </strong>
              <span>API Keys</span>
            </button>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}
