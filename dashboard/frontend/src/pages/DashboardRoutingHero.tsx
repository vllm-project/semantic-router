import styles from './DashboardRoutingHero.module.css'

interface DashboardRoutingHeroProps {
  modelCount: number
  signalCount: number
  decisionCount: number
  apiKeyCount: number
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
            <span className={`${styles.liveStatus} ${healthy ? styles.liveHealthy : ''}`}>
              <i /> {healthy ? 'Live' : overallStatus || 'Checking'}
            </span>
          </div>
          <h1 id="dashboard-routing-title">Your model system, at a glance.</h1>
          <p>One stable API. Every capability path visible, governed, and ready.</p>
        </div>
        <div className={styles.bannerActions}>
          <button
            type="button"
            className={styles.primaryAction}
            onClick={() => onNavigate('/playground')}
          >
            Try a request <span>→</span>
          </button>
          <button
            type="button"
            className={styles.refreshAction}
            onClick={onRefresh}
            disabled={refreshing}
          >
            <svg
              className={refreshing ? styles.spinning : ''}
              width="14"
              height="14"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <path d="M14 8A6 6 0 1 1 8 2" strokeLinecap="round" />
              <path d="M14 2v6H8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {refreshing ? 'Refreshing' : 'Refresh'}
          </button>
        </div>
      </header>

      <div className={styles.metricStrip}>
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
        <button type="button" onClick={() => onNavigate('/access/api-keys')}>
          <strong>{apiKeyCount}</strong>
          <span>API Keys</span>
        </button>
        <div className={styles.updated}>
          <span>Updated</span>
          <strong>
            {lastUpdated
              ? lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              : '—'}
          </strong>
        </div>
      </div>
    </section>
  )
}
