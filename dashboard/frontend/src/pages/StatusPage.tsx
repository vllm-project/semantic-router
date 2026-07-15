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
import { clampPage, filterServices, type ServiceHealthFilter } from './statusPageSupport'
import styles from './StatusPage.module.css'

const StatusPage: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [serviceQuery, setServiceQuery] = useState('')
  const [serviceHealth, setServiceHealth] = useState<ServiceHealthFilter>('all')
  const [servicePage, setServicePage] = useState(1)
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
  const filteredServices = useMemo(
    () => filterServices(status?.services ?? [], serviceQuery, serviceHealth),
    [serviceHealth, serviceQuery, status?.services],
  )
  const servicePageSize = 9
  const currentServicePage = clampPage(servicePage, filteredServices.length, servicePageSize)
  const servicePageCount = Math.max(1, Math.ceil(filteredServices.length / servicePageSize))
  const visibleServices = filteredServices.slice(
    (currentServicePage - 1) * servicePageSize,
    currentServicePage * servicePageSize,
  )

  useEffect(() => {
    setServicePage(1)
  }, [serviceHealth, serviceQuery])

  useEffect(() => {
    if (!status?.models?.models.length) {
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
  }, [status?.models?.models.length])

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

  return (
    <div className={styles.container} data-testid="status-page">
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.pageEyebrow}>Operate / System health</span>
          <h1 className={styles.title}>System status</h1>
          <p className={styles.subtitle}>
            Live router health, model readiness, and deployment details in one operational view.
          </p>
        </div>
        <div className={styles.headerRight}>
          {lastUpdated && (
            <span className={styles.headerTimestamp}>
              <span className={styles.liveDot} aria-hidden="true" />
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <label className={styles.autoRefreshToggle}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
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
      </div>

      {error && (
        <div className={styles.error} role="alert">
          <span className={styles.errorIcon}>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {status && modelStatus && (
        <>
          <StatusOverview
            status={status}
            modelStatus={modelStatus}
            runtime={runtime}
            healthyServices={healthyServices}
            loadedModels={loadedModels}
            knownModels={knownModels}
          />

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
                  Model Inventory
                </h2>
                <p className={styles.servicesSectionDescription}>
                  The router-reported model list, load state, and metadata exposed by{' '}
                  <code>/info/models</code>.
                </p>
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

          <section
            className={styles.servicesSection}
            data-testid="status-services-section"
            aria-labelledby="status-services-title"
          >
            <div className={styles.servicesSectionHeader}>
              <div>
                <h2 id="status-services-title" className={styles.servicesSectionTitle}>
                  Services
                </h2>
                <p className={styles.servicesSectionDescription}>
                  Process-level health for the router, proxy, dashboard, and runtime helpers.
                </p>
              </div>
              <div className={styles.servicesHeaderMeta}>
                <span className={styles.servicesCountChip}>
                  {healthyServices}/{status.services.length} healthy
                </span>
              </div>
            </div>

            {status.services.length > 0 && (
              <div className={styles.servicesToolbar}>
                <label className={styles.serviceSearchField}>
                  <span className={styles.srOnly}>Search services</span>
                  <input
                    type="search"
                    value={serviceQuery}
                    onChange={(event) => setServiceQuery(event.target.value)}
                    placeholder="Search service, component, or status"
                  />
                </label>
                <label className={styles.serviceFilterField}>
                  <span>Health</span>
                  <select
                    value={serviceHealth}
                    onChange={(event) =>
                      setServiceHealth(event.target.value as ServiceHealthFilter)
                    }
                  >
                    <option value="all">All services</option>
                    <option value="healthy">Healthy</option>
                    <option value="unhealthy">Needs attention</option>
                  </select>
                </label>
                <span className={styles.serviceResultCount} aria-live="polite">
                  {filteredServices.length} of {status.services.length} services
                </span>
              </div>
            )}

            <div className={styles.servicesGrid}>
              {status.services.length > 0 && visibleServices.length > 0 ? (
                visibleServices.map((service, index) => (
                  <article
                    key={`${service.name}-${index}`}
                    className={`${styles.serviceCard} ${
                      service.healthy ? styles.serviceCardHealthy : styles.serviceCardUnhealthy
                    }`}
                  >
                    <div className={styles.serviceCardTop}>
                      <div className={styles.serviceNameWrap}>
                        <span
                          className={`${styles.serviceStateDot} ${
                            service.healthy
                              ? styles.serviceStateDotHealthy
                              : styles.serviceStateDotUnhealthy
                          }`}
                        />
                        <h3 className={styles.serviceName}>{service.name}</h3>
                        {service.component && (
                          <span className={styles.componentBadge}>{service.component}</span>
                        )}
                      </div>
                      <span
                        className={`${styles.serviceHealthChip} ${
                          service.healthy
                            ? styles.serviceHealthHealthy
                            : styles.serviceHealthUnhealthy
                        }`}
                      >
                        <span className={styles.serviceHealthDot} />
                        {service.status}
                      </span>
                    </div>

                    {service.message ? (
                      <p className={styles.serviceMessage}>{service.message}</p>
                    ) : (
                      <p className={styles.serviceMessageMuted}>No additional details reported.</p>
                    )}
                  </article>
                ))
              ) : status.services.length === 0 ? (
                <div className={styles.noServices}>
                  <span className={styles.noServicesIcon}>🔍</span>
                  <h3>No Running Services Detected</h3>
                  <p>Start the semantic router using one of these methods:</p>
                  <div className={styles.startOptions}>
                    <div className={styles.startOption}>
                      <strong>Local:</strong>
                      <code>vllm-sr serve</code>
                    </div>
                    <div className={styles.startOption}>
                      <strong>Docker:</strong>
                      <code>docker compose up</code>
                    </div>
                    <div className={styles.startOption}>
                      <strong>Kubernetes:</strong>
                      <code>kubectl apply -f deploy/kubernetes/</code>
                    </div>
                  </div>
                </div>
              ) : (
                <div className={styles.noServices}>
                  <h3>No matching services</h3>
                  <p>Change the search or health filter.</p>
                  <button
                    type="button"
                    className={styles.clearServiceFilters}
                    onClick={() => {
                      setServiceQuery('')
                      setServiceHealth('all')
                    }}
                  >
                    Clear filters
                  </button>
                </div>
              )}
            </div>

            {servicePageCount > 1 && (
              <nav className={styles.servicePagination} aria-label="Service inventory pages">
                <button
                  type="button"
                  disabled={currentServicePage === 1}
                  onClick={() => setServicePage((value) => Math.max(1, value - 1))}
                >
                  Previous
                </button>
                <span>
                  Page {currentServicePage} of {servicePageCount}
                </span>
                <button
                  type="button"
                  disabled={currentServicePage === servicePageCount}
                  onClick={() => setServicePage((value) => Math.min(servicePageCount, value + 1))}
                >
                  Next
                </button>
              </nav>
            )}
          </section>
        </>
      )}
    </div>
  )
}

export default StatusPage
