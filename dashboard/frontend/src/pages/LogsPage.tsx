import React, { useEffect, useState, useCallback, useRef } from 'react'
import styles from './LogsPage.module.css'

interface LogEntry {
  line: string
  service?: string
}

interface LogsResponse {
  deployment_type: string
  service: string
  logs: LogEntry[]
  count: number
  error?: string
  message?: string
}

type ComponentType = 'router' | 'envoy' | 'dashboard' | 'all'

const LogsPage: React.FC = () => {
  const [selectedComponent, setSelectedComponent] = useState<ComponentType>('all')
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [deploymentType, setDeploymentType] = useState<string>('detecting...')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [lines, setLines] = useState(100)
  const logsContainerRef = useRef<HTMLDivElement>(null)

  const fetchLogs = useCallback(async () => {
    try {
      const response = await fetch(`/api/logs?component=${selectedComponent}&lines=${lines}`)
      if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.statusText}`)
      }
      const data: LogsResponse = await response.json()
      setLogs(data.logs || [])
      setDeploymentType(data.deployment_type)
      setError(data.error || null)
      setMessage(data.message || null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [selectedComponent, lines])

  useEffect(() => {
    setLoading(true)
    fetchLogs()

    if (autoRefresh) {
      const interval = setInterval(fetchLogs, 5000)
      return () => clearInterval(interval)
    }
  }, [fetchLogs, autoRefresh])

  useEffect(() => {
    if (autoScroll && logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const getDeploymentIcon = (type: string) => {
    const lowerType = type.toLowerCase()
    if (lowerType.includes('helm')) return '⎈'
    if (lowerType.includes('kubernetes') || lowerType.includes('k8s')) return '☸️'
    if (lowerType.includes('docker')) return '🐳'
    if (lowerType.includes('local')) return '💻'
    return '🚀'
  }

  const getLogLevel = (line: string): string => {
    const lowerLine = line.toLowerCase()
    if (lowerLine.includes('"level":"error"') || lowerLine.includes('[error]')) return 'error'
    if (lowerLine.includes('"level":"warn"') || lowerLine.includes('[warn]')) return 'warn'
    if (lowerLine.includes('"level":"info"') || lowerLine.includes('[info]')) return 'info'
    if (lowerLine.includes('"level":"debug"') || lowerLine.includes('[debug]')) return 'debug'
    return ''
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>
            <span className={styles.titleIcon}>📜</span>
            System Logs
          </h1>
          <p className={styles.subtitle}>
            View logs from vLLM Semantic Router services
            {deploymentType !== 'none' && deploymentType !== 'detecting...' && (
              <span className={styles.deploymentBadge}>
                {getDeploymentIcon(deploymentType)} {deploymentType}
              </span>
            )}
          </p>
        </div>
      </div>

      <div className={styles.controls}>
        <div className={styles.serviceSelector}>
          <button
            className={`${styles.serviceButton} ${selectedComponent === 'router' ? styles.active : ''}`}
            onClick={() => setSelectedComponent('router')}
          >
            🤖 Router
          </button>
          <button
            className={`${styles.serviceButton} ${selectedComponent === 'envoy' ? styles.active : ''}`}
            onClick={() => setSelectedComponent('envoy')}
          >
            🔀 Envoy
          </button>
          <button
            className={`${styles.serviceButton} ${selectedComponent === 'dashboard' ? styles.active : ''}`}
            onClick={() => setSelectedComponent('dashboard')}
          >
            📊 Dashboard
          </button>
          <button
            className={`${styles.serviceButton} ${selectedComponent === 'all' ? styles.active : ''}`}
            onClick={() => setSelectedComponent('all')}
          >
            📦 All
          </button>
        </div>

        <div className={styles.controlsRight}>
          <div className={styles.linesSelector}>
            <label>Lines:</label>
            <select
              value={lines}
              onChange={(e) => setLines(Number(e.target.value))}
              className={styles.linesSelect}
            >
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={500}>500</option>
            </select>
          </div>

          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            <span>Auto-refresh</span>
          </label>

          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            <span>Auto-scroll</span>
          </label>

          <button onClick={fetchLogs} className={styles.refreshButton}>
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className={styles.error}>
          <span className={styles.errorIcon}>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {message && !error && (
        <div className={styles.info}>
          <span className={styles.infoIcon}>ℹ️</span>
          <span>{message}</span>
        </div>
      )}

      <div className={styles.logsSection}>
        <div className={styles.logsHeader}>
          <span className={styles.logsTitle}>
            {selectedComponent.charAt(0).toUpperCase() + selectedComponent.slice(1)} Logs
          </span>
          <span className={styles.logsCount}>{logs.length} entries</span>
        </div>

        <div ref={logsContainerRef} className={styles.logsContainer}>
          {loading && logs.length === 0 ? (
            <div className={styles.loadingLogs}>
              <div className={styles.spinner}></div>
              <span>Fetching logs...</span>
            </div>
          ) : logs.length === 0 ? (
            <div className={styles.noLogs}>
              <span className={styles.noLogsIcon}>📭</span>
              <p>No logs available</p>
              {deploymentType === 'none' && (
                <p className={styles.noLogsHint}>
                  No running deployment detected. Start the router first.
                </p>
              )}
            </div>
          ) : (
            <div className={styles.logsList}>
              {logs.map((log, index) => {
                const level = getLogLevel(log.line)
                return (
                  <div 
                    key={index} 
                    className={`${styles.logEntry} ${level ? styles[`level${level.charAt(0).toUpperCase() + level.slice(1)}`] : ''}`}
                  >
                    <span className={styles.logLine}>{log.line}</span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default LogsPage
