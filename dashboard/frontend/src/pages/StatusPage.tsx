import React, { useCallback, useEffect, useMemo, useState } from 'react'
import ProductLoadingState from '../components/ProductLoadingState'
import { fetchSystemStatus, type SystemStatus } from '../utils/routerRuntime'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'
import StatusAvailabilityPanel from './StatusAvailabilityPanel'
import styles from './StatusPage.module.css'

const StatusPage: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const fetchStatus = useCallback(async () => {
    try {
      setStatus(await fetchSystemStatus())
      setLastUpdated(new Date())
      setError(null)
    } catch (err) {
      setStatus(null)
      setLastUpdated(null)
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [])

  const statusRequest = useMemo(() => createVisibilityAwareRequest(fetchStatus), [fetchStatus])

  useEffect(() => {
    void statusRequest.run({ allowHidden: true })

    const refreshWhenVisible = () => {
      if (!document.hidden) void statusRequest.run()
    }
    document.addEventListener('visibilitychange', refreshWhenVisible)

    const interval = window.setInterval(() => {
      void statusRequest.run()
    }, 10000)

    return () => {
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [statusRequest])

  if (loading && !status) {
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
            onClick={() => void statusRequest.run({ allowHidden: true })}
            aria-label="Refresh system status"
            title={lastUpdated ? `Last checked ${lastUpdated.toLocaleTimeString()}` : 'Check now'}
          >
            <i
              className={`${styles.liveDot} ${
                lastUpdated ? styles.liveDotHealthy : styles.liveDotUnavailable
              }`}
            />
            {lastUpdated ? 'Live' : error ? 'Unavailable' : 'Checking'}
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
