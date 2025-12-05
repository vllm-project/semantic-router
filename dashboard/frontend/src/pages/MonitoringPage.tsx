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
          src={buildGrafanaUrl()}
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
