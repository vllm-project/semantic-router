import React, { useCallback, useEffect, useMemo, useState } from 'react'

import ProductLoadingState from '../components/ProductLoadingState'
import type { SystemStatus } from '../utils/routerRuntime'
import StatusAvailabilityPanel from './StatusAvailabilityPanel'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'
import styles from './StatusPage.module.css'

const StatusPage: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/status', { headers: { Accept: 'application/json' } })
      if (!response.ok) throw new Error(`Status request failed (${response.status}).`)
      setStatus((await response.json()) as SystemStatus)
      setLastUpdated(new Date())
      setError(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'System status is unavailable.')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const statusRequest = useMemo(() => createVisibilityAwareRequest(fetchStatus), [fetchStatus])

  useEffect(() => {
    void statusRequest.run({ allowHidden: true })
    const refreshWhenVisible = () => {
      if (!document.hidden) void statusRequest.run()
    }
    document.addEventListener('visibilitychange', refreshWhenVisible)
    const interval = window.setInterval(refreshWhenVisible, 10_000)
    return () => {
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [statusRequest])

  if (isLoading && !status) return <ProductLoadingState label="Checking service availability" />

  return (
    <div className={styles.container} data-testid="status-page">
      <header className={styles.statusMasthead}>
        <div>
          <div className={styles.eyebrowRow}>
            <span className={styles.pageEyebrow}>System</span>
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
                lastUpdated && !error ? styles.liveDotHealthy : styles.liveDotUnavailable
              }`}
            />
            {lastUpdated && !error ? 'Live' : error ? 'Unavailable' : 'Checking'}
          </button>
        </div>
      </header>

      <StatusAvailabilityPanel status={status} lastUpdated={lastUpdated} />

      {error ? (
        <div className={styles.error} role="alert">
          <span className={styles.errorIcon} aria-hidden="true">
            !
          </span>
          <span>{error}</span>
        </div>
      ) : null}
    </div>
  )
}

export default StatusPage
