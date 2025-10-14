import React, { useState, useEffect } from 'react'
import styles from './PlaygroundPage.module.css'

const PlaygroundPage: React.FC = () => {
  const [openWebUIUrl] = useState('http://localhost:3001')
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
    </div>
  )
}

export default PlaygroundPage
