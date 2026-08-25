import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { type SystemStatus } from '../utils/routerRuntime'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'
import styles from './StatusPage.module.css'
import ProductIcon from '../components/ProductIcon'

type StatusHistorySample = {
  at: number
  overall: string
  services: Record<string, boolean>
}

const STATUS_HISTORY_KEY = 'vllm-sr.status.history.v1'

const formatStatusLabel = (value: string) =>
  value.replace(/_/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())

const StatusPage: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [history, setHistory] = useState<StatusHistorySample[]>([])

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/status')
      if (!response.ok) {
        throw new Error(`Failed to fetch status: ${response.statusText}`)
      }

      const data = (await response.json()) as SystemStatus
      setStatus(data)
      setLastUpdated(new Date())
      setError(null)
    } catch (err) {
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

  const healthyServices = useMemo(
    () => status?.services.filter((service) => service.healthy).length ?? 0,
    [status],
  )

  useEffect(() => {
    if (!status) return
    const sample: StatusHistorySample = {
      at: Date.now(),
      overall: status.overall,
      services: Object.fromEntries(
        status.services.map((service) => [service.name, service.healthy]),
      ),
    }
    let previous: StatusHistorySample[] = []
    try {
      previous = JSON.parse(
        window.localStorage.getItem(STATUS_HISTORY_KEY) || '[]',
      ) as StatusHistorySample[]
    } catch {
      previous = []
    }
    const last = previous[previous.length - 1]
    const changed =
      !last ||
      last.overall !== sample.overall ||
      JSON.stringify(last.services) !== JSON.stringify(sample.services)
    const stale = !last || sample.at - last.at >= 5 * 60 * 1000
    const next = (changed || stale ? [...previous, sample] : previous).slice(-90)
    window.localStorage.setItem(STATUS_HISTORY_KEY, JSON.stringify(next))
    setHistory(next)
  }, [status])

  if (loading && !status) {
    return (
      <div className={styles.container} data-testid="status-page">
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>Checking service availability...</p>
        </div>
      </div>
    )
  }

  const healthLabel = status ? formatStatusLabel(status.overall) : 'Unavailable'
  const servicesReady = Boolean(
    status &&
      status.services.length > 0 &&
      status.overall === 'healthy' &&
      healthyServices === status.services.length,
  )
  const fullyOperational = servicesReady
  const noServices = Boolean(status && status.services.length === 0)
  const bannerTitle = fullyOperational
    ? 'All systems operational'
    : noServices
      ? 'No running services detected'
      : healthLabel
  const bannerCopy = fullyOperational
    ? 'Services are responding normally.'
    : noServices
      ? 'Start the Router to see live availability.'
      : 'One or more components need attention.'

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
            <i className={styles.liveDot} />
            {lastUpdated ? 'Live' : 'Checking'}
          </button>
        </div>
      </header>

      <section
        data-testid="status-overview"
        className={`${styles.overallBanner} ${fullyOperational ? styles.overallHealthy : styles.overallDegraded}`}
        aria-live="polite"
      >
        <span className={styles.overallIcon}>
          <ProductIcon name={fullyOperational ? 'check' : 'alert'} aria-hidden="true" />
        </span>
        <div>
          <h2>{bannerTitle}</h2>
          <p>{bannerCopy}</p>
        </div>
        <dl>
          <div>
            <dt>Services</dt>
            <dd>{status ? `${healthyServices}/${status.services.length}` : '—'}</dd>
          </div>
          <div>
            <dt>Router</dt>
            <dd>{status?.services.find((service) => service.name === 'Router')?.status || '—'}</dd>
          </div>
          <div>
            <dt>Last checked</dt>
            <dd>{lastUpdated?.toLocaleTimeString() || '—'}</dd>
          </div>
        </dl>
      </section>

      {status ? (
        <section
          className={styles.componentBoard}
          aria-labelledby="component-status-title"
          data-testid="status-services-section"
        >
          <div className={styles.componentBoardHeader}>
            <div>
              <span>Current availability</span>
              <h2 id="component-status-title">Components</h2>
            </div>
            <span>
              {history.length} recorded {history.length === 1 ? 'check' : 'checks'}
            </span>
          </div>
          <div>
            {status.services.length === 0 ? (
              <div className={styles.noServices}>
                <strong>No services reported</strong>
                <span>Availability appears here when the Router starts.</span>
              </div>
            ) : null}
            {status.services.map((service) => {
              const samples = history.slice(-30)
              return (
                <article key={service.name} className={styles.componentRow}>
                  <div>
                    <strong>{service.name}</strong>
                    <span>Live health check</span>
                  </div>
                  <div className={styles.uptimeBars} aria-label={`${service.name} recent checks`}>
                    {samples.length ? (
                      samples.map((sample) => (
                        <i
                          key={sample.at}
                          className={
                            sample.services[service.name] === false
                              ? styles.uptimeDown
                              : styles.uptimeUp
                          }
                          title={`${new Date(sample.at).toLocaleString()} · ${sample.services[service.name] === false ? 'Unavailable' : 'Operational'}`}
                        />
                      ))
                    ) : (
                      <i className={service.healthy ? styles.uptimeUp : styles.uptimeDown} />
                    )}
                  </div>
                  <span
                    className={
                      service.healthy ? styles.componentOperational : styles.componentIssue
                    }
                  >
                    {service.healthy ? 'Operational' : formatStatusLabel(service.status)}
                  </span>
                </article>
              )
            })}
          </div>
        </section>
      ) : null}

      <section className={styles.incidentHistory}>
        <div>
          <span>History</span>
          <h2>Recent incidents</h2>
        </div>
        {history.some((sample) => sample.overall !== 'healthy') ? (
          history
            .filter((sample) => sample.overall !== 'healthy')
            .slice(-5)
            .reverse()
            .map((sample) => (
              <article key={sample.at}>
                <i />
                <div>
                  <strong>{sample.overall.replace(/_/g, ' ')}</strong>
                  <span>{new Date(sample.at).toLocaleString()}</span>
                </div>
                <p>
                  {Object.entries(sample.services)
                    .filter(([, healthy]) => !healthy)
                    .map(([name]) => name)
                    .join(', ') || 'Runtime health degraded'}
                </p>
              </article>
            ))
        ) : (
          <div className={styles.noIncidents}>
            <strong>No incidents recorded</strong>
            <span>Incident history appears here.</span>
          </div>
        )}
      </section>

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
