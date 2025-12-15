import React, { useState, useEffect } from 'react'
import styles from './PlaygroundPage.module.css'

const PlaygroundPage: React.FC = () => {
  // Detect OpenWebUI URL based on current hostname
  const getOpenWebUIUrl = () => {
    const hostname = window.location.hostname
    const protocol = window.location.protocol

    // Build-time configurable port (e.g., for All-in-One deployment)
    const configuredPort = import.meta.env.VITE_OPENWEBUI_PORT
    if (configuredPort) {
      return `${protocol}//${hostname}:${configuredPort}`
    }

    // Assumes openwebui and dashboard have matching hostname patterns
    const openwebuiHost = hostname.replace('dashboard', 'openwebui')
    if (openwebuiHost === hostname) {
      // hostname doesn't contain 'dashboard', cannot determine Open WebUI URL
      return ''
    }
    return `${protocol}//${openwebuiHost}`
  }

  const [openWebUIUrl] = useState(() => getOpenWebUIUrl())
  const [currentUrl, setCurrentUrl] = useState('')

  // Auto-load on mount
  useEffect(() => {
    // Default to loading the configured URL on mount
    setCurrentUrl(openWebUIUrl)
  }, [openWebUIUrl]) // Load when URL changes

  return (
    <div className={styles.container}>
      <div className={styles.iframeContainer}>
        {!currentUrl && (
          <div className={styles.placeholder}>
            <span className={styles.placeholderIcon}>🎮</span>
            <h3>Open WebUI Playground</h3>
            <p>
              Test your LLM models and semantic routing with Open WebUI.
            </p>
            <p className={styles.note}>
              If unable to load, please check Open WebUI deployment and port configuration.
            </p>
          </div>
        )}

        {currentUrl && (
          <iframe
            src={currentUrl}
            className={styles.iframe}
            title="Open WebUI Playground"
            allowFullScreen
          />
        )}
      </div>
    </div>
  )
}

export default PlaygroundPage
