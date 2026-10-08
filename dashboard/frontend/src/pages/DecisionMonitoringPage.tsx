import DecisionTaskMonitoring from './DecisionTaskMonitoring'
import { Link } from 'react-router-dom'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import DecisionModelRuntimePanel from './DecisionModelRuntimePanel'
import { useDecisionModelMonitoring } from './useDecisionModelMonitoring'
import styles from './DecisionModelPage.module.css'

export default function DecisionMonitoringPage() {
  const runtime = useDecisionModelMonitoring()

  return (
    <ConfigPageManagerLayout
      eyebrow="Build / System One"
      title="Decision Monitoring"
      description="Watch deployment health, runtime traffic, latency, and cache behavior over time."
    >
      <div className={styles.page}>
        <div className={styles.toolbar}>
          <span>
            {runtime.updatedAt
              ? `Last checked ${runtime.updatedAt.toLocaleTimeString()} · refreshes every 10 seconds`
              : 'Loading runtime observations…'}
          </span>
          <div className={styles.toolbarActions}>
            <Link className={styles.testLink} to="/decision-model">
              Decision Models
            </Link>
            <Link className={styles.testLink} to="/decision-model/playground">
              Decision Playground
            </Link>
            <button
              type="button"
              onClick={() => void runtime.refresh()}
              disabled={runtime.refreshing}
            >
              {runtime.refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
        </div>
        {runtime.errors.length > 0 && (
          <div className={styles.notice} role="alert">
            {runtime.errors.map((error) => (
              <p key={error}>{error}</p>
            ))}
          </div>
        )}
        <DecisionModelRuntimePanel
          inventory={runtime.inventory}
          refreshedAt={runtime.updatedAt}
          loading={runtime.loading}
        />
        <DecisionTaskMonitoring refreshedAt={runtime.updatedAt} />
      </div>
    </ConfigPageManagerLayout>
  )
}
