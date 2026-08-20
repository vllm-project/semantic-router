import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import RouterModelInventory from '../components/RouterModelInventory'
import EmbeddingProviderStatusPanel from './EmbeddingProviderStatusPanel'
import {
  getActiveRouterRuntime,
  getLoadedModelCount,
  getModelStatusSummary,
  getTotalKnownModelCount,
  type SystemStatus,
} from '../utils/routerRuntime'
import StatusOverview from './StatusOverview'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'
import styles from './StatusPage.module.css'

type StatusHistorySample = {
  at: number
  overall: string
  services: Record<string, boolean>
}

const STATUS_HISTORY_KEY = 'vllm-sr.status.history.v1'

const StatusPage: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [history, setHistory] = useState<StatusHistorySample[]>([])
  const scrolledHashRef = useRef<string | null>(null)

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

    if (!autoRefresh) {
      return () => document.removeEventListener('visibilitychange', refreshWhenVisible)
    }

    const interval = window.setInterval(() => {
      void statusRequest.run()
    }, 10000)

    return () => {
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [autoRefresh, statusRequest])

  const modelStatus = useMemo(() => (status ? getModelStatusSummary(status) : null), [status])
  const runtime = useMemo(() => (status ? getActiveRouterRuntime(status) : null), [status])
  const healthyServices = useMemo(
    () => status?.services.filter((service) => service.healthy).length ?? 0,
    [status],
  )
  const loadedModels = useMemo(() => getLoadedModelCount(status?.models), [status])
  const knownModels = useMemo(() => getTotalKnownModelCount(status?.models), [status])
  useEffect(() => {
    if (!status?.models?.models?.length) {
      return
    }

    const currentHash = window.location.hash
    if (!currentHash || scrolledHashRef.current === currentHash) {
      return
    }

    const targetId = decodeURIComponent(currentHash.slice(1))
    const target = document.getElementById(targetId)
    if (!target) {
      return
    }

    scrolledHashRef.current = currentHash
    target.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [status?.models?.models?.length])

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
          <p>Detecting deployment and checking status...</p>
        </div>
      </div>
    )
  }

  const healthLabel = status
    ? status.overall.replace(/_/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())
    : 'Unavailable'

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
          <span className={styles.headerTimestamp}>
            <i className={styles.liveDot} />
            {lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : 'Checking now'}
          </span>
          <label className={styles.autoRefreshToggle}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(event) => setAutoRefresh(event.target.checked)}
            />
            <span>Auto-refresh</span>
          </label>
          <button
            onClick={() => void statusRequest.run({ allowHidden: true })}
            className={styles.refreshButton}
            aria-label="Refresh system status"
          >
            Refresh
          </button>
        </div>
      </header>

      <section
        className={`${styles.overallBanner} ${status?.overall === 'healthy' ? styles.overallHealthy : styles.overallDegraded}`}
      >
        <span className={styles.overallIcon}>{status?.overall === 'healthy' ? '✓' : '!'}</span>
        <div>
          <h2>{status?.overall === 'healthy' ? 'All systems operational' : `${healthLabel}`}</h2>
          <p>
            {status?.overall === 'healthy'
              ? 'Every monitored component is responding normally.'
              : 'One or more components need attention.'}
          </p>
        </div>
        <dl>
          <div>
            <dt>Services</dt>
            <dd>{status ? `${healthyServices}/${status.services.length}` : '—'}</dd>
          </div>
          <div>
            <dt>Models ready</dt>
            <dd>{knownModels > 0 ? `${loadedModels}/${knownModels}` : '—'}</dd>
          </div>
          <div>
            <dt>Deployment</dt>
            <dd>{status?.deployment_type || '—'}</dd>
          </div>
        </dl>
      </section>

      {status && modelStatus ? (
        <StatusOverview
          status={status}
          modelStatus={modelStatus}
          runtime={runtime}
          healthyServices={healthyServices}
          loadedModels={loadedModels}
          knownModels={knownModels}
        />
      ) : null}

      {status ? (
        <section className={styles.componentBoard} aria-labelledby="component-status-title">
          <div className={styles.componentBoardHeader}>
            <div>
              <span>Current availability</span>
              <h2 id="component-status-title">Components</h2>
            </div>
            <span>
              {history.length} recorded {history.length === 1 ? 'check' : 'checks'}
            </span>
          </div>
          <div className={styles.componentRows}>
            {status.services.map((service) => {
              const samples = history.slice(-30)
              return (
                <article key={service.name} className={styles.componentRow}>
                  <div>
                    <strong>{service.name}</strong>
                    <span>{service.message || service.component || 'Live health check'}</span>
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
                    {service.healthy ? 'Operational' : service.status}
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
          <span className={styles.errorIcon}>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {status && modelStatus && (
        <>
          {status.router_runtime?.embedding_provider ? (
            <EmbeddingProviderStatusPanel provider={status.router_runtime.embedding_provider} />
          ) : null}

          <section
            className={styles.servicesSection}
            data-testid="status-model-inventory-section"
            aria-labelledby="status-model-inventory-title"
          >
            <div className={styles.servicesSectionHeader}>
              <div>
                <h2 id="status-model-inventory-title" className={styles.servicesSectionTitle}>
                  Model inventory
                </h2>
                <p className={styles.servicesSectionDescription}>Models available to the router.</p>
              </div>
            </div>

            <div className={styles.sectionBody}>
              <RouterModelInventory
                mode="full"
                showSummary={false}
                modelsInfo={status.models}
                emptyMessage="The router has not exposed any model metadata yet."
              />
            </div>
          </section>
        </>
      )}
    </div>
  )
}

export default StatusPage
