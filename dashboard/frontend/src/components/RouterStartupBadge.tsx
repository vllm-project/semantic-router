import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  describeRouterRuntime,
  getActiveRouterRuntime,
  type RouterRuntimeStatus,
  type StatusWithRouterRuntime,
} from '../utils/routerRuntime'
import styles from './RouterStartupBadge.module.css'

const RouterStartupBadge: React.FC = () => {
  const navigate = useNavigate()
  const [runtime, setRuntime] = useState<RouterRuntimeStatus | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/status')
      if (!response.ok) {
        return
      }

      const data = (await response.json()) as StatusWithRouterRuntime
      setRuntime(getActiveRouterRuntime(data))
    } catch {
      // Ignore transient polling errors in the header badge.
    }
  }, [])

  useEffect(() => {
    void fetchStatus()
    const interval = window.setInterval(() => {
      void fetchStatus()
    }, 10000)

    return () => {
      window.clearInterval(interval)
    }
  }, [fetchStatus])

  const detail = useMemo(() => {
    if (!runtime) {
      return ''
    }
    return describeRouterRuntime(runtime)
  }, [runtime])

  if (!runtime) {
    return null
  }

  return (
    <button
      type="button"
      className={styles.badge}
      title={runtime.message || detail}
      onClick={() => navigate('/status')}
    >
      <span className={styles.spinner} aria-hidden="true" />
      <span className={styles.text}>
        <span className={styles.label}>vLLM-SR starting</span>
        <span className={styles.detail}>{detail}</span>
      </span>
    </button>
  )
}

export default RouterStartupBadge
