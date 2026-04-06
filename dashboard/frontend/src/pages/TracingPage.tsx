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
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [serviceAvailable, setServiceAvailable] = useState<boolean | null>(
    null
  )
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [themeSet, setThemeSet] = useState(false)

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

  // Theme monitoring removed since we only support dark mode
  // If light mode support is added in the future, uncomment this code

  const buildJaegerUrl = () => {
    // Jaeger UI theme is controlled by localStorage, not URL parameters
    // We can only set the service and lookback time via URL
    // The theme will be inherited from the parent page's theme
    // Note: Jaeger may show light theme initially if it hasn't loaded the theme preference yet
    return `/embedded/jaeger/search?lookback=1h&limit=20&service=vllm-sr`
  }

  useEffect(() => {
    // Set a timeout to hide loading overlay if iframe doesn't trigger onLoad
    // This prevents the loading overlay from staying forever if iframe load event doesn't fire
    if (serviceAvailable && loading) {
      const timer = setTimeout(() => {
        setLoading(false)
      }, 5000) // 5 second timeout
      return () => clearTimeout(timer)
    }
  }, [serviceAvailable, loading])

  const handleIframeLoad = () => {
    setLoading(false)
    setError(null)

    // Force Jaeger UI to use LIGHT theme for consistent display
    // This avoids theme conflicts that cause "patchy" appearance (深一块浅一块)
    // Since Jaeger is served from the same origin (via proxy), we can access its localStorage
    try {
      const iframe = iframeRef.current
      if (iframe && iframe.contentWindow) {
        // Wait a bit for Jaeger to initialize
        setTimeout(() => {
          try {
            const iframeWindow = iframe.contentWindow
            if (iframeWindow && iframeWindow.localStorage) {
              // Check current theme
              const currentTheme = iframeWindow.localStorage.getItem('jaeger-ui-theme') ||
                                   iframeWindow.localStorage.getItem('theme')

              // If not already light, set it to light and reload
              if (currentTheme !== 'light' && !themeSet) {
                iframeWindow.localStorage.setItem('jaeger-ui-theme', 'light')
                iframeWindow.localStorage.setItem('theme', 'light')
                setThemeSet(true)
                // Reload iframe to apply the theme
                iframeWindow.location.reload()
                return
              }

              // Also set data-theme attribute for immediate effect
              if (iframeWindow.document && iframeWindow.document.documentElement) {
                iframeWindow.document.documentElement.setAttribute('data-theme', 'light')
                iframeWindow.document.documentElement.setAttribute('data-bs-theme', 'light')
                iframeWindow.document.documentElement.style.colorScheme = 'light'
              }
            }
          } catch (e) {
            // Cross-origin restrictions - this is expected if Jaeger is on a different domain
            console.log('Note: Could not access Jaeger iframe (cross-origin):', e)
          }
        }, 100)
      }
    } catch (e) {
      console.log('Note: Could not access iframe:', e)
    }
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
