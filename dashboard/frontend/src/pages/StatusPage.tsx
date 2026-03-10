import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  describeRouterRuntime,
  getActiveRouterRuntime,
  getModelStatusSummary,
  type RouterRuntimeStatus,
} from '../utils/routerRuntime'
import styles from './StatusPage.module.css'

interface ServiceStatus {
  name: string
  status: string
  healthy: boolean
  message?: string
  component?: string
}

interface SystemStatus {
  overall: string
  deployment_type: string
  services: ServiceStatus[]
  version?: string
  router_runtime?: RouterRuntimeStatus
}

const StatusPage: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

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

  useEffect(() => {
    void fetchStatus()

    if (!autoRefresh) {
      return
    }

    const interval = window.setInterval(() => {
      void fetchStatus()
    }, 10000)

    return () => window.clearInterval(interval)
  }, [fetchStatus, autoRefresh])

  const getModelToneClass = (tone: 'ok' | 'warn' | 'down') => {
    if (tone === 'ok') return styles.modelStatusOk
    if (tone === 'down') return styles.modelStatusDown
    return styles.modelStatusWarn
  }

  const getOverallChipClass = (overall: string) => {
    if (overall === 'healthy') return styles.modelStatusOk
    if (overall === 'degraded') return styles.modelStatusWarn
    return styles.modelStatusDown
  }

  const getOverallTextClass = (overall: string) => {
    if (overall === 'healthy') return styles.toneOkText
    if (overall === 'degraded') return styles.toneWarnText
    return styles.toneDownText
  }

  const getOverallSurfaceClass = (overall: string) => {
    if (overall === 'healthy') return styles.routerHeroOk
    if (overall === 'degraded') return styles.routerHeroWarn
    return styles.routerHeroDown
  }

  const getOverallLabel = (overall: string) => {
    if (overall === 'not_running') return 'Not Running'
    if (overall === 'stopped') return 'Stopped'
    return overall.charAt(0).toUpperCase() + overall.slice(1)
  }

  const formatDeploymentType = (type: string) => {
    if (type === 'none') return 'Not Detected'
    return type.charAt(0).toUpperCase() + type.slice(1)
  }

  const modelStatus = useMemo(() => (status ? getModelStatusSummary(status) : null), [status])
  const runtime = useMemo(() => (status ? getActiveRouterRuntime(status) : null), [status])
  const healthyServices = useMemo(
    () => status?.services.filter((service) => service.healthy).length ?? 0,
    [status],
  )

  if (loading && !status) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>Detecting deployment and checking status...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>System Status</h1>
          <p className={styles.subtitle}>
            Real-time health status of vLLM Semantic Router services, model warmup, and deployment readiness.
          </p>
        </div>
        <div className={styles.headerRight}>
          {lastUpdated && (
            <span className={styles.headerTimestamp}>
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
          <button onClick={() => void fetchStatus()} className={styles.refreshButton}>
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

      {status && modelStatus && (
        <>
          <div className={styles.summaryGrid}>
            <section className={`${styles.summaryCard} ${styles.routerCard}`}>
              <div className={styles.cardHeader}>
                <div className={styles.cardTitleBlock}>
                  <h2 className={styles.cardTitle}>Router Status</h2>
                  <p className={styles.cardSubtitle}>Service health and deployment readiness.</p>
                </div>
                <span className={`${styles.statusChip} ${getOverallChipClass(status.overall)}`}>
                  {getOverallLabel(status.overall)}
                </span>
              </div>
              <div className={styles.cardBody}>
                <div className={`${styles.routerHero} ${getOverallSurfaceClass(status.overall)}`}>
                  <div className={styles.routerHeroHeader}>
                    <span className={styles.routerHeroLabel}>Current health</span>
                    <span className={styles.routerHeroMetric}>
                      {healthyServices}/{status.services.length} services
                    </span>
                  </div>
                  <div className={styles.routerCopy}>
                    <div className={`${styles.routerValue} ${getOverallTextClass(status.overall)}`}>
                      {getOverallLabel(status.overall)}
                    </div>
                    <p className={styles.routerNarrative}>
                      Service health and deployment readiness across the active router runtime.
                    </p>
                    <div className={styles.routerHighlights}>
                      <div className={styles.routerHighlight}>
                        <span className={styles.routerHighlightLabel}>Deployment</span>
                        <span className={styles.routerHighlightValue}>
                          {formatDeploymentType(status.deployment_type)}
                        </span>
                      </div>
                      <div className={styles.routerHighlight}>
                        <span className={styles.routerHighlightLabel}>Runtime</span>
                        <span className={styles.routerHighlightValue}>
                          {runtime ? runtime.phase.replace(/_/g, ' ') : modelStatus.value}
                        </span>
                      </div>
                      <div className={styles.routerHighlight}>
                        <span className={styles.routerHighlightLabel}>Coverage</span>
                        <span className={styles.routerHighlightValue}>
                          {healthyServices === status.services.length ? 'All services ready' : 'Attention required'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section className={`${styles.summaryCard} ${styles.modelCard}`}>
              <div className={styles.cardHeader}>
                <div className={styles.cardTitleBlock}>
                  <h2 className={styles.cardTitle}>Model Status</h2>
                  <p className={styles.cardSubtitle}>Router warmup, download, and readiness.</p>
                </div>
                <span className={`${styles.statusChip} ${getModelToneClass(modelStatus.tone)}`}>
                  {modelStatus.value}
                </span>
              </div>
              <div className={styles.cardBody}>
                <p className={styles.cardDescription}>{modelStatus.detail}</p>

                {runtime ? (
                  <div className={styles.modelFacts}>
                    <div className={styles.modelFactRow}>
                      <span className={styles.factLabel}>Phase</span>
                      <span className={styles.factValue}>{runtime.phase}</span>
                    </div>

                    {runtime.downloading_model && (
                      <div className={styles.modelFactRow}>
                        <span className={styles.factLabel}>Current model</span>
                        <span className={styles.factValue}>{runtime.downloading_model}</span>
                      </div>
                    )}

                    {typeof runtime.ready_models === 'number' &&
                      typeof runtime.total_models === 'number' &&
                      runtime.total_models > 0 && (
                        <div className={styles.modelFactRow}>
                          <span className={styles.factLabel}>Ready</span>
                          <span className={styles.factValue}>
                            {runtime.ready_models}/{runtime.total_models}
                          </span>
                        </div>
                      )}

                    <div className={styles.modelHint}>{describeRouterRuntime(runtime)}</div>
                  </div>
                ) : (
                  <div className={styles.modelReadyPanel}>All required models are ready.</div>
                )}
              </div>
            </section>

            <section className={`${styles.summaryCard} ${styles.metaCard}`}>
              <div className={styles.cardHeader}>
                <div className={styles.cardTitleBlock}>
                  <h2 className={styles.cardTitle}>Runtime Facts</h2>
                  <p className={styles.cardSubtitle}>Deployment and live runtime metadata.</p>
                </div>
              </div>
              <div className={styles.cardBody}>
                <div className={styles.metaGrid}>
                  <div className={styles.metaItem}>
                    <span className={styles.metaLabel}>Deployment</span>
                    <span className={styles.metaValue}>{formatDeploymentType(status.deployment_type)}</span>
                  </div>

                  <div className={styles.metaItem}>
                    <span className={styles.metaLabel}>Version</span>
                    <span className={styles.metaValue}>{status.version || 'Unknown'}</span>
                  </div>

                  <div className={styles.metaItem}>
                    <span className={styles.metaLabel}>Healthy services</span>
                    <span className={styles.metaValue}>
                      {healthyServices}/{status.services.length}
                    </span>
                  </div>

                  {lastUpdated && (
                    <div className={styles.metaItem}>
                      <span className={styles.metaLabel}>Last update</span>
                      <span className={styles.metaValue}>{lastUpdated.toLocaleTimeString()}</span>
                    </div>
                  )}
                </div>
              </div>
            </section>
          </div>

          <section className={styles.servicesSection}>
            <div className={styles.servicesSectionHeader}>
              <div>
                <span className={styles.servicesSectionTitle}>Services</span>
                <p className={styles.servicesSectionDescription}>
                  Process-level health for the router, proxy, dashboard, and runtime helpers.
                </p>
              </div>
              <div className={styles.servicesHeaderMeta}>
                <span className={styles.servicesCountChip}>
                  {healthyServices}/{status.services.length} healthy
                </span>
                <span className={styles.servicesCountChip}>
                  {status.services.length} {status.services.length === 1 ? 'service' : 'services'}
                </span>
              </div>
            </div>

            <div className={styles.servicesGrid}>
              {status.services.length > 0 ? (
                status.services.map((service, index) => (
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
                            service.healthy ? styles.serviceStateDotHealthy : styles.serviceStateDotUnhealthy
                          }`}
                        />
                        <h3 className={styles.serviceName}>{service.name}</h3>
                        {service.component && (
                          <span className={styles.componentBadge}>{service.component}</span>
                        )}
                      </div>
                      <span
                        className={`${styles.serviceHealthChip} ${
                          service.healthy ? styles.serviceHealthHealthy : styles.serviceHealthUnhealthy
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
              ) : (
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
              )}
            </div>
          </section>
        </>
      )}
    </div>
  )
}

export default StatusPage
