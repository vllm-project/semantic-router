import React, { useEffect, useRef, useState, useCallback } from 'react'
import styles from './MonitoringPage.module.css'
import ServiceNotConfigured, {
  ServiceConfig,
} from '../components/ServiceNotConfigured'

// Jaeger service configuration
const JAEGER_SERVICE: ServiceConfig = {
  name: 'Jaeger',
  envVar: 'TARGET_JAEGER_URL',
  description:
    'Jaeger is used for distributed tracing to help you monitor and troubleshoot the semantic router. Please configure the Jaeger URL to enable tracing capabilities.',
  docsUrl:
    'https://vllm-semantic-router.com/docs/tutorials/observability/dashboard',
  exampleValue: 'http://localhost:16686',
}

const TracingPage: React.FC = () => {
  const [theme, setTheme] = useState(
    document.documentElement.getAttribute('data-theme') || 'dark'
  )
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [serviceAvailable, setServiceAvailable] = useState<boolean | null>(
    null
  )
  const iframeRef = useRef<HTMLIFrameElement>(null)

  // Check if Jaeger service is available
  const checkServiceAvailability = useCallback(async () => {
    try {
      const response = await fetch('/embedded/jaeger/', {
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

  useEffect(() => {
    const observer = new MutationObserver(() => {
      const t = document.documentElement.getAttribute('data-theme') || 'dark'
      if (t !== theme) {
        setTheme(t)
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

  const buildJaegerUrl = () => {
    // Default Jaeger landing page; could navigate to search with params later
    return `/embedded/jaeger/search?lookback=1h&limit=20&service=vllm-semantic-router`
  }

  useEffect(() => {
    // Slight delay to ensure iframe renders (only if service is available)
    if (serviceAvailable) {
      const timer = setTimeout(() => setLoading(false), 100)
      return () => clearTimeout(timer)
    }
  }, [theme, serviceAvailable])

  const handleIframeLoad = () => {
    setLoading(false)
    setError(null)
  }

  const handleIframeError = () => {
    setLoading(false)
    setError(
      'Failed to load Jaeger UI. Please check that Jaeger is running and the proxy is configured.'
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
        <ServiceNotConfigured service={JAEGER_SERVICE} onRetry={handleRetry} />
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
            <p>Loading Jaeger UI...</p>
          </div>
        )}
        {serviceAvailable && (
          <iframe
            ref={iframeRef}
            key={`jaeger-${theme}`}
            src={buildJaegerUrl()}
            className={styles.iframe}
            title="Jaeger Tracing"
            allowFullScreen
            onLoad={handleIframeLoad}
            onError={handleIframeError}
          />
        )}
      </div>
    </div>
  )
}

export default TracingPage
