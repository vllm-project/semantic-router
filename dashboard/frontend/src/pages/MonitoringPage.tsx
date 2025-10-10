import React, { useState } from 'react'
import styles from './MonitoringPage.module.css'

const MonitoringPage: React.FC = () => {
  const [grafanaPath, setGrafanaPath] = useState('/d/semantic-router/semantic-router-dashboard')
  const [currentPath, setCurrentPath] = useState(grafanaPath)

  const handlePathChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setGrafanaPath(e.target.value)
  }

  const handleApply = () => {
    setCurrentPath(grafanaPath)
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleApply()
    }
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
            placeholder="/d/semantic-router/semantic-router-dashboard"
          />
          <button onClick={handleApply} className={styles.button}>
            Apply
          </button>
        </div>
        <div className={styles.hints}>
          <span className={styles.hint}>💡 Tip: Press Enter to apply changes</span>
        </div>
      </div>
      <div className={styles.iframeContainer}>
        <iframe
          src={`/embedded/grafana${currentPath}`}
          className={styles.iframe}
          title="Grafana Dashboard"
          allowFullScreen
        />
      </div>
    </div>
  )
}

export default MonitoringPage
