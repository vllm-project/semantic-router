import React, { useEffect, useState, useCallback } from 'react'
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
      const data = await response.json()
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
    fetchStatus()
    
    if (autoRefresh) {
      const interval = setInterval(fetchStatus, 10000)
      return () => clearInterval(interval)
    }
  }, [fetchStatus, autoRefresh])

  const getStatusIcon = (healthy: boolean) => {
    return healthy ? '✅' : '❌'
  }

  const getStatusClass = (healthy: boolean) => {
    return healthy ? styles.healthy : styles.unhealthy
  }

  const getOverallIcon = (overall: string) => {
    switch (overall) {
      case 'healthy':
        return '🟢'
      case 'degraded':
        return '🟡'
      case 'not_running':
        return '⚫'
      default:
        return '🔴'
    }
  }

  const getDeploymentIcon = (type: string) => {
    const lowerType = type.toLowerCase()
    if (lowerType.includes('helm')) return '⎈'
    if (lowerType.includes('kubernetes') || lowerType.includes('k8s')) return '☸️'
    if (lowerType.includes('docker')) return '🐳'
    if (lowerType.includes('local')) return '💻'
    return '🚀'
  }

  if (loading && !status) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>
          <div className={styles.spinner}></div>
          <p>Detecting deployment and checking status...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>
            <span className={styles.titleIcon}>🩺</span>
            System Status
          </h1>
          <p className={styles.subtitle}>
            Real-time health status of vLLM Semantic Router services
          </p>
        </div>
        <div className={styles.headerRight}>
          <label className={styles.autoRefreshToggle}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            <span>Auto-refresh</span>
          </label>
          <button onClick={fetchStatus} className={styles.refreshButton}>
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

      {status && (
        <>
          <div className={styles.overallStatus}>
            <div className={styles.overallCard}>
              <span className={styles.overallIcon}>{getOverallIcon(status.overall)}</span>
              <div className={styles.overallInfo}>
                <span className={styles.overallLabel}>Overall Status</span>
                <span className={`${styles.overallValue} ${styles[status.overall] || ''}`}>
                  {status.overall === 'not_running' ? 'Not Running' : 
                   status.overall.charAt(0).toUpperCase() + status.overall.slice(1)}
                </span>
              </div>
              <div className={styles.deploymentType}>
                <span className={styles.deploymentIcon}>{getDeploymentIcon(status.deployment_type)}</span>
                <div className={styles.deploymentInfo}>
                  <span className={styles.deploymentLabel}>Deployment</span>
                  <span className={styles.deploymentValue}>
                    {status.deployment_type === 'none' ? 'Not Detected' :
                     status.deployment_type.charAt(0).toUpperCase() + status.deployment_type.slice(1)}
                  </span>
                </div>
              </div>
              {status.version && (
                <div className={styles.version}>
                  <span className={styles.versionLabel}>Version</span>
                  <span className={styles.versionValue}>{status.version}</span>
                </div>
              )}
            </div>
          </div>

          <div className={styles.servicesSection}>
            <div className={styles.servicesSectionHeader}>
              <span className={styles.servicesSectionTitle}>Services</span>
              <span className={styles.servicesCount}>
                {status.services.length} {status.services.length === 1 ? 'service' : 'services'}
              </span>
            </div>
            <div className={styles.servicesList}>
              {status.services.length > 0 ? (
                status.services.map((service, index) => (
                  <div
                    key={`${service.name}-${index}`}
                    className={`${styles.serviceCard} ${getStatusClass(service.healthy)}`}
                  >
                    <span className={styles.serviceIcon}>{getStatusIcon(service.healthy)}</span>
                    <div className={styles.serviceInfo}>
                      <div className={styles.serviceHeader}>
                        <h3 className={styles.serviceName}>{service.name}</h3>
                        {service.component && (
                          <span className={styles.componentBadge}>{service.component}</span>
                        )}
                      </div>
                      <div className={styles.serviceBody}>
                        <div className={styles.statusRow}>
                          <span className={styles.statusLabel}>Status:</span>
                          <span className={`${styles.statusValue} ${getStatusClass(service.healthy)}`}>
                            {service.status}
                          </span>
                        </div>
                        {service.message && (
                          <div className={styles.messageRow}>
                            <span className={styles.messageLabel}>Details:</span>
                            <span className={styles.messageValue}>{service.message}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
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
          </div>

          {lastUpdated && (
            <div className={styles.footer}>
              <span className={styles.lastUpdated}>
                Last updated: {lastUpdated.toLocaleTimeString()}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default StatusPage
