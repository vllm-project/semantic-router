import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'
import { filterLogs, getLogLevel, type LogEntry, type LogLevel } from './logsPageSupport'
import styles from './LogsPage.module.css'

interface LogsResponse {
  deployment_type: string
  service: string
  logs: LogEntry[]
  count: number
  error?: string
  message?: string
}

type ComponentType = 'router' | 'envoy' | 'dashboard' | 'all'

const COMPONENT_OPTIONS: Array<{ value: ComponentType; label: string }> = [
  { value: 'router', label: 'Router' },
  { value: 'envoy', label: 'Envoy' },
  { value: 'dashboard', label: 'Dashboard' },
  { value: 'all', label: 'All services' },
]

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
  const [query, setQuery] = useState('')
  const [level, setLevel] = useState<LogLevel>('all')
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const logsContainerRef = useRef<HTMLDivElement>(null)
  const requestControllerRef = useRef<AbortController | null>(null)

  const fetchLogs = useCallback(async () => {
    requestControllerRef.current?.abort()
    const controller = new AbortController()
    requestControllerRef.current = controller
    try {
      const response = await fetch(`/api/logs?component=${selectedComponent}&lines=${lines}`, {
        signal: controller.signal,
      })
      if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.statusText}`)
      }
      const data: LogsResponse = await response.json()
      if (controller.signal.aborted) return
      setLogs(data.logs || [])
      setDeploymentType(data.deployment_type)
      setError(data.error || null)
      setMessage(data.message || null)
      setLastUpdated(new Date())
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      if (requestControllerRef.current === controller) setLoading(false)
    }
  }, [selectedComponent, lines])

  const logsRequest = useMemo(() => createVisibilityAwareRequest(fetchLogs), [fetchLogs])

  useEffect(() => {
    setLoading(true)
    void logsRequest.run({ allowHidden: true })

    const refreshWhenVisible = () => {
      if (!document.hidden) void logsRequest.run()
    }
    document.addEventListener('visibilitychange', refreshWhenVisible)

    if (autoRefresh) {
      const interval = window.setInterval(() => void logsRequest.run(), 5000)
      return () => {
        window.clearInterval(interval)
        document.removeEventListener('visibilitychange', refreshWhenVisible)
        requestControllerRef.current?.abort()
      }
    }

    return () => {
      document.removeEventListener('visibilitychange', refreshWhenVisible)
      requestControllerRef.current?.abort()
    }
  }, [autoRefresh, logsRequest])

  const filteredLogs = useMemo(() => filterLogs(logs, query, level), [level, logs, query])

  useEffect(() => {
    if (autoScroll && logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight
    }
  }, [filteredLogs, autoScroll])

  const formatDeploymentType = (type: string) => {
    if (type === 'none') return 'Not detected'
    if (type === 'detecting...') return 'Detecting'
    return type
      .split(/[-_\s]+/)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(' ')
  }

  const activeComponentLabel =
    COMPONENT_OPTIONS.find((option) => option.value === selectedComponent)?.label || 'All services'

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.eyebrow}>Operations</span>
          <h1 className={styles.title}>System Logs</h1>
          <p className={styles.subtitle}>
            View live output from vLLM Semantic Router services and runtime helpers.
          </p>
        </div>
        <div className={styles.headerRight}>
          <span className={styles.headerMeta}>Active stream: {activeComponentLabel}</span>
          <span className={styles.headerMeta}>
            Deployment: {formatDeploymentType(deploymentType)}
          </span>
          {lastUpdated && (
            <span className={styles.headerMeta}>Updated: {lastUpdated.toLocaleTimeString()}</span>
          )}
        </div>
      </div>

      <div className={styles.summaryGrid}>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Deployment</span>
          <strong className={styles.summaryValue}>{formatDeploymentType(deploymentType)}</strong>
          <span className={styles.summaryHint}>Resolved from the active runtime environment.</span>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Selected source</span>
          <strong className={styles.summaryValue}>{activeComponentLabel}</strong>
          <span className={styles.summaryHint}>
            Switch sources without changing the log viewport width.
          </span>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Entries loaded</span>
          <strong className={styles.summaryValue}>{logs.length}</strong>
          <span className={styles.summaryHint}>Showing the latest {lines} lines per request.</span>
        </article>
      </div>

      <section className={styles.controlPanel}>
        <div className={styles.controlPanelHeader}>
          <div>
            <h2 className={styles.panelTitle}>Stream controls</h2>
            <p className={styles.panelSubtitle}>
              Tune source selection, tail length, and live refresh behavior.
            </p>
          </div>
        </div>

        <div className={styles.controls}>
          <div className={styles.serviceSelector}>
            {COMPONENT_OPTIONS.map((option) => (
              <button
                key={option.value}
                className={`${styles.serviceButton} ${selectedComponent === option.value ? styles.active : ''}`}
                onClick={() => setSelectedComponent(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>

          <div className={styles.controlsRight}>
            <label className={styles.filterField}>
              <span>Search logs</span>
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Message or service"
              />
            </label>

            <label className={styles.filterField}>
              <span>Level</span>
              <select value={level} onChange={(event) => setLevel(event.target.value as LogLevel)}>
                <option value="all">All levels</option>
                <option value="error">Error</option>
                <option value="warn">Warning</option>
                <option value="info">Info</option>
                <option value="debug">Debug</option>
                <option value="other">Other</option>
              </select>
            </label>

            <div className={styles.linesSelector}>
              <label htmlFor="logs-lines">Lines</label>
              <select
                id="logs-lines"
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

            <button
              onClick={() => void logsRequest.run({ allowHidden: true })}
              className={styles.refreshButton}
            >
              Refresh
            </button>
          </div>
        </div>
      </section>

      {error && (
        <div className={styles.error}>
          <span className={styles.messageLabel}>Log error</span>
          <span>{error}</span>
        </div>
      )}

      {message && !error && (
        <div className={styles.info}>
          <span className={styles.messageLabel}>Notice</span>
          <span>{message}</span>
        </div>
      )}

      <div className={styles.logsSection}>
        <div className={styles.logsHeader}>
          <div className={styles.logsHeaderText}>
            <span className={styles.logsEyebrow}>Live stream</span>
            <span className={styles.logsTitle}>{activeComponentLabel}</span>
          </div>
          <span className={styles.logsCount} aria-live="polite">
            {filteredLogs.length === logs.length
              ? `${logs.length} entries`
              : `${filteredLogs.length} of ${logs.length} entries`}
          </span>
        </div>

        <div ref={logsContainerRef} className={styles.logsContainer}>
          {loading && logs.length === 0 ? (
            <div className={styles.loadingLogs}>
              <div className={styles.spinner}></div>
              <span>Fetching logs...</span>
            </div>
          ) : logs.length === 0 ? (
            <div className={styles.noLogs}>
              <p className={styles.noLogsTitle}>No logs available</p>
              {deploymentType === 'none' && (
                <p className={styles.noLogsHint}>
                  No running deployment detected. Start the router first.
                </p>
              )}
            </div>
          ) : filteredLogs.length === 0 ? (
            <div className={styles.noLogs}>
              <p className={styles.noLogsTitle}>No matching logs</p>
              <p className={styles.noLogsHint}>Change the search or level filter.</p>
            </div>
          ) : (
            <div className={styles.logsList}>
              {filteredLogs.map((log, index) => {
                const level = getLogLevel(log.line)
                return (
                  <div
                    key={`${log.service ?? 'service'}-${index}-${log.line.slice(0, 32)}`}
                    className={`${styles.logEntry} ${level !== 'other' ? styles[`level${level.charAt(0).toUpperCase() + level.slice(1)}`] : ''}`}
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
