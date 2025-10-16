import React, { useEffect, useRef, useState } from 'react'
import styles from './MonitoringPage.module.css'

const TracingPage: React.FC = () => {
  const [theme, setTheme] = useState(
    document.documentElement.getAttribute('data-theme') || 'dark'
  )
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const iframeRef = useRef<HTMLIFrameElement>(null)

  useEffect(() => {
    const observer = new MutationObserver(() => {
      const t = document.documentElement.getAttribute('data-theme') || 'dark'
      if (t !== theme) {
        setTheme(t)
        setLoading(true)
      }
    })
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => observer.disconnect()
  }, [theme])

  const buildJaegerUrl = () => {
    // Default Jaeger landing page; could navigate to search with params later
    return `/embedded/jaeger/search?lookback=1h&limit=20&service=vllm-semantic-router`
  }

  useEffect(() => {
    // Slight delay to ensure iframe renders
    const timer = setTimeout(() => setLoading(false), 100)
    return () => clearTimeout(timer)
  }, [theme])

  const handleIframeLoad = () => {
    setLoading(false)
    setError(null)
  }

  const handleIframeError = () => {
    setLoading(false)
    setError('Failed to load Jaeger UI. Please check that Jaeger is running and the proxy is configured.')
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
      </div>
    </div>
  )
}

export default TracingPage
