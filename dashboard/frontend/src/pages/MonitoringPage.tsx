import React, { useState, useEffect, useRef, useCallback } from 'react'
import styles from './MonitoringPage.module.css'
import ServiceNotConfigured, {
  ServiceConfig,
} from '../components/ServiceNotConfigured'

// Grafana service configuration
const GRAFANA_SERVICE: ServiceConfig = {
  name: 'Grafana',
  envVar: 'TARGET_GRAFANA_URL',
  description:
    'Grafana is used to display metrics and dashboards for the semantic router. Please configure the Grafana URL to enable monitoring capabilities.',
  docsUrl:
    'https://vllm-semantic-router.com/docs/tutorials/observability/dashboard',
  exampleValue: 'http://localhost:3000',
}

const MonitoringPage: React.FC = () => {
  // Get theme from document attribute
  const getTheme = () => {
    return document.documentElement.getAttribute('data-theme') || 'dark'
  }

  const [theme, setTheme] = useState(getTheme())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [serviceAvailable, setServiceAvailable] = useState<boolean | null>(
    null
  )
  const iframeRef = useRef<HTMLIFrameElement>(null)

  // Check if Grafana service is available
  const checkServiceAvailability = useCallback(async () => {
    try {
      const response = await fetch('/embedded/grafana/', {
        method: 'HEAD',
      })
      // 503 means service not configured
      if (response.status === 503) {
        setServiceAvailable(false)
        setLoading(false)
        return
      }
      setServiceAvailable(true)
    } catch {
      // Network error - service might be available but request failed
      setServiceAvailable(true)
    }
  }, [])

  // Check service availability on mount
  useEffect(() => {
    checkServiceAvailability()
  }, [checkServiceAvailability])

  // Listen to theme changes
  useEffect(() => {
    const observer = new MutationObserver(() => {
      const newTheme = getTheme()
      if (newTheme !== theme) {
        setTheme(newTheme)
        // Reload iframe with new theme by toggling loading state
        if (serviceAvailable) {
          setLoading(true)
        }
      }
    })

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    })

    return () => observer.disconnect()
  }, [theme, serviceAvailable])

  // Build Grafana dashboard URL directly
  const buildGrafanaUrl = () => {
    // Load the dashboard directly using the goto endpoint
    // This is the cleanest approach and avoids redirect loops
    const url = `/embedded/grafana/goto/llm-router-metrics?orgId=1&theme=${theme}&refresh=30s`
    console.log('Grafana URL:', url)
    return url
  }

  const handleIframeLoad = () => {
    console.log('Iframe loaded')
    setLoading(false)
    setError(null)
  }

  const handleIframeError = () => {
    setLoading(false)
    setError(
      'Failed to load Grafana dashboard. Please check the dashboard path and try again.'
    )
  }

  const handleRetry = () => {
    setServiceAvailable(null)
    setLoading(true)
    setError(null)
    checkServiceAvailability()
  }

  // Show service not configured page
  if (serviceAvailable === false) {
    return (
      <div className={styles.container}>
        <ServiceNotConfigured service={GRAFANA_SERVICE} onRetry={handleRetry} />
      </div>
    )
  }

  return (
    <div className={styles.container}>
      {error && (
        <div className={styles.errorBanner}>
          <span className={styles.errorIcon}>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      <div className={styles.iframeContainer}>
        {loading && (
          <div className={styles.loadingOverlay}>
            <div className={styles.spinner}></div>
            <p>Loading Grafana dashboard...</p>
          </div>
        )}
        {serviceAvailable && (
          <iframe
            ref={iframeRef}
            key={`grafana-${theme}`}
            src={buildGrafanaUrl()}
            className={styles.iframe}
            title="Grafana Dashboard"
            allowFullScreen
            onLoad={handleIframeLoad}
            onError={handleIframeError}
          />
        )}
      </div>
    </div>
  )
}

export default MonitoringPage
