import React, { useState, useEffect } from 'react'
import styles from './ConfigPage.module.css'

interface ConfigData {
  [key: string]: unknown
}

const ConfigPage: React.FC = () => {
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedEndpoint, setSelectedEndpoint] = useState('/api/router/config/classification')

  const endpoints = [
    { value: '/api/router/config/classification', label: 'Classification Config' },
    { value: '/api/router/config/router', label: 'Router Config' },
    { value: '/api/router/config/all', label: 'All Config' },
  ]

  useEffect(() => {
    fetchConfig()
  }, [selectedEndpoint])

  const fetchConfig = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(selectedEndpoint)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch config')
      setConfig(null)
    } finally {
      setLoading(false)
    }
  }

  const handleRefresh = () => {
    fetchConfig()
  }

  const handleEndpointChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedEndpoint(e.target.value)
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h2 className={styles.title}>Configuration Viewer</h2>
          <select
            value={selectedEndpoint}
            onChange={handleEndpointChange}
            className={styles.select}
          >
            {endpoints.map((endpoint) => (
              <option key={endpoint.value} value={endpoint.value}>
                {endpoint.label}
              </option>
            ))}
          </select>
        </div>
        <button onClick={handleRefresh} className={styles.button} disabled={loading}>
          🔄 Refresh
        </button>
      </div>

      <div className={styles.content}>
        {loading && (
          <div className={styles.loading}>
            <div className={styles.spinner}></div>
            <p>Loading configuration...</p>
          </div>
        )}

        {error && !loading && (
          <div className={styles.error}>
            <span className={styles.errorIcon}>⚠️</span>
            <div>
              <h3>Error Loading Config</h3>
              <p>{error}</p>
            </div>
          </div>
        )}

        {config && !loading && !error && (
          <pre className={styles.codeBlock}>
            <code>{JSON.stringify(config, null, 2)}</code>
          </pre>
        )}
      </div>
    </div>
  )
}

export default ConfigPage
