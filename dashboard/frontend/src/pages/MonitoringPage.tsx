import React, { useState, useEffect } from 'react'
import styles from './MonitoringPage.module.css'

const MonitoringPage: React.FC = () => {
  // Get theme from document attribute
  const getTheme = () => {
    return document.documentElement.getAttribute('data-theme') || 'dark'
  }

  const [grafanaPath, setGrafanaPath] = useState('/d/semantic-router-dashboard/semantic-router')
  const [currentPath, setCurrentPath] = useState(grafanaPath)
  const [theme, setTheme] = useState(getTheme())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loadTimeout, setLoadTimeout] = useState<ReturnType<typeof setTimeout> | null>(null)

  // Listen to theme changes
  useEffect(() => {
    const observer = new MutationObserver(() => {
      const newTheme = getTheme()
      if (newTheme !== theme) {
        setTheme(newTheme)
        // Reload iframe with new theme
        if (currentPath) {
          setCurrentPath(currentPath) // Trigger re-render
          setLoading(true)
        }
      }
    })

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    })

    return () => observer.disconnect()
  }, [theme, currentPath])

  // Build complete Grafana URL with necessary parameters
  const buildGrafanaUrl = (path: string) => {
    const cleanPath = path.startsWith('/') ? path : `/${path}`
    // Add kiosk mode, theme, and other necessary parameters
    const params = new URLSearchParams({
      kiosk: 'tv', // tv mode hides some UI elements but keeps time range picker
      theme: theme,
      refresh: '30s',
    })
    return `/embedded/grafana${cleanPath}?${params.toString()}`
  }

  const handlePathChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setGrafanaPath(e.target.value)
  }

  const handleApply = () => {
    setError(null)
    setLoading(true)
    setCurrentPath(grafanaPath)

    // Set timeout for loading
    if (loadTimeout) clearTimeout(loadTimeout)
    const timeout = setTimeout(() => {
      setLoading(false)
      setError('Dashboard loading timeout. Please check if the dashboard path is correct.')
    }, 15000) // 15 second timeout
    setLoadTimeout(timeout)
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleApply()
    }
  }

  const handleIframeLoad = () => {
    if (loadTimeout) clearTimeout(loadTimeout)
    setLoading(false)
    setError(null)
  }

  const handleIframeError = () => {
    if (loadTimeout) clearTimeout(loadTimeout)
    setLoading(false)
    setError('Failed to load Grafana dashboard. Please check the dashboard path and try again.')
  }

  return (
    <div className={styles.container}>
      <div className={styles.controls}>
        <div className={styles.controlGroup}>
          <label htmlFor="grafana-path" className={styles.label}>
            Grafana Dashboard Path:
          </label>
          <input
            id="grafana-path"
            type="text"
            value={grafanaPath}
            onChange={handlePathChange}
            onKeyPress={handleKeyPress}
            className={styles.input}
            placeholder="/d/semantic-router-dashboard/semantic-router"
          />
          <button onClick={handleApply} className={styles.button}>
            Apply
          </button>
        </div>
        <div className={styles.hints}>
          <span className={styles.hint}>💡 Tip: Press Enter to apply changes</span>
          <span className={styles.hint}>
            🎨 Theme: <strong>{theme}</strong> (synced with dashboard)
          </span>
        </div>
      </div>

      {error && (
        <div className={styles.errorBanner}>
          <span className={styles.errorIcon}>⚠️</span>
          <span>{error}</span>
          <button onClick={handleApply} className={styles.retryButton}>
            Retry
          </button>
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
          src={buildGrafanaUrl(currentPath)}
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
