import React, { useState, useEffect } from 'react'
import styles from './PlaygroundPage.module.css'

const PlaygroundPage: React.FC = () => {
  const [openWebUIUrl, setOpenWebUIUrl] = useState('http://localhost:3001')
  const [currentUrl, setCurrentUrl] = useState('')

  // Auto-load on mount
  useEffect(() => {
    // Default to loading the configured URL on mount
    setCurrentUrl(openWebUIUrl)
  }, []) // Empty dependency array means this runs once on mount

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setOpenWebUIUrl(e.target.value)
  }

  const handleApply = () => {
    setCurrentUrl(openWebUIUrl)
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleApply()
    }
  }

  return (
    <div className={styles.container}>
      <div className={styles.iframeContainer}>
        {!currentUrl && (
          <div className={styles.placeholder}>
            <span className={styles.placeholderIcon}>🎮</span>
            <h3>Open WebUI Playground</h3>
            <p>
              Configure the URL below and click &quot;Load&quot; to embed Open WebUI interface.
            </p>
            <p className={styles.note}>
              Note: Open WebUI needs to be deployed separately. Check the dashboard README for instructions.
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

      <div className={styles.controls}>
        <div className={styles.controlGroup}>
          <label htmlFor="openwebui-url" className={styles.label}>
            Open WebUI URL:
          </label>
          <input
            id="openwebui-url"
            type="text"
            value={openWebUIUrl}
            onChange={handleUrlChange}
            onKeyPress={handleKeyPress}
            className={styles.input}
            placeholder="http://localhost:3001"
          />
          <button onClick={handleApply} className={styles.button}>
            Load
          </button>
        </div>

        <div className={styles.hints}>
            <span className={styles.hint}>
            🎮 Open WebUI Playground: Test your LLM models and semantic routing.
            </span>
        </div>
      </div>
    </div>
  )
}

export default PlaygroundPage
