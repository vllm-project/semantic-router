import React, { useState, useEffect, useRef } from 'react'
import styles from './MonitoringPage.module.css'

const MonitoringPage: React.FC = () => {
  // Get theme from document attribute
  const getTheme = () => {
    return document.documentElement.getAttribute('data-theme') || 'dark'
  }

  const [theme, setTheme] = useState(getTheme())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const iframeRef = useRef<HTMLIFrameElement>(null)

  // Listen to theme changes
  useEffect(() => {
    const observer = new MutationObserver(() => {
      const newTheme = getTheme()
      if (newTheme !== theme) {
        setTheme(newTheme)
        // Reload iframe with new theme by toggling loading state
        setLoading(true)
      }
    })

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    })

    return () => observer.disconnect()
  }, [theme])

  // Build initial Grafana URL - load the root path first
  const buildInitialGrafanaUrl = () => {
    // Start with Grafana root path
    const url = `/embedded/grafana/?orgId=1&theme=${theme}`
    console.log('Initial Grafana URL:', url)
    return url
  }

  // Build dashboard URL using goto endpoint - this is what Grafana uses internally
  const buildDashboardUrl = () => {
    // Use Grafana's goto endpoint to navigate by UID
    // This mimics the internal navigation when clicking Home
    const url = `/embedded/grafana/goto/llm-router-metrics?orgId=1&theme=${theme}&refresh=30s`
    console.log('Dashboard goto URL:', url)
    return url
  }

  // Add effect to handle automatic redirect to dashboard
  useEffect(() => {
    // After initial page load, wait a bit then redirect to dashboard using goto
    const timer = setTimeout(() => {
      if (iframeRef.current) {
        console.log('Redirecting to dashboard using goto...')
        // Use goto endpoint to navigate to dashboard (mimics clicking Home)
        iframeRef.current.src = buildDashboardUrl()
      }
      setLoading(false)
    }, 100) // Wait 0.1 seconds after initial load

    return () => clearTimeout(timer)
  }, [theme])



  const handleIframeLoad = () => {
    console.log('Iframe loaded')
    setLoading(false)
    setError(null)
  }

  const handleIframeError = () => {
    setLoading(false)
    setError('Failed to load Grafana dashboard. Please check the dashboard path and try again.')
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
        <iframe
          ref={iframeRef}
          key={`grafana-${theme}`}
          src={buildInitialGrafanaUrl()}
          className={styles.iframe}
          title="Grafana Dashboard"
          allowFullScreen
          onLoad={handleIframeLoad}
          onError={handleIframeError}
        />
      </div>
    </div>
  )
}

export default MonitoringPage
