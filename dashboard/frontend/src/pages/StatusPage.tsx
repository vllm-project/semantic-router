import React from 'react'
import ProductLoadingState from '../components/ProductLoadingState'
import { useSystemStatus } from '../contexts/SystemStatusContext'
import StatusAvailabilityPanel from './StatusAvailabilityPanel'
import styles from './StatusPage.module.css'

const StatusPage: React.FC = () => {
  const { status, isLoading, error, lastUpdated, refresh } = useSystemStatus()

  if (isLoading && !status) {
    return <ProductLoadingState label="Checking service availability" />
  }

  return (
    <div className={styles.container} data-testid="status-page">
      <header className={styles.statusMasthead}>
        <div>
          <div className={styles.eyebrowRow}>
            <span className={styles.pageEyebrow}>Status</span>
            <span className={styles.brandLockup}>
              <img src="/vllm.png" alt="" />
              vllm-sr
            </span>
          </div>
          <h1>System status</h1>
          <p>Models and services, live at a glance.</p>
        </div>
        <div className={styles.headerRight}>
          <button
            type="button"
            className={styles.liveRefreshButton}
            onClick={() => void refresh()}
            aria-label="Refresh system status"
            title={lastUpdated ? `Last checked ${lastUpdated.toLocaleTimeString()}` : 'Check now'}
          >
            <i
              className={`${styles.liveDot} ${
                lastUpdated && !error ? styles.liveDotHealthy : styles.liveDotUnavailable
              }`}
            />
            {lastUpdated && !error ? 'Live' : error ? 'Unavailable' : 'Checking'}
          </button>
        </div>
      </header>

      <StatusAvailabilityPanel status={status} lastUpdated={lastUpdated} />

      {error && (
        <div className={styles.error} role="alert">
          <span className={styles.errorIcon} aria-hidden="true">
            !
          </span>
          <span>{error}</span>
        </div>
      )}
    </div>
  )
}

export default StatusPage
