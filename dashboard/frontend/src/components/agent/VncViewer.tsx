/**
 * VncViewer - Component for displaying E2B desktop VNC stream
 */

import { useState, useEffect, useRef } from 'react'
import styles from './VncViewer.module.css'

interface VncViewerProps {
  /** VNC stream URL from E2B */
  vncUrl: string | null
  /** Whether the connection is active */
  isConnected?: boolean
  /** Whether the agent is running */
  isRunning?: boolean
  /** Error message to display */
  error?: string | null
  /** Width of the viewer (default: 100%) */
  width?: string | number
  /** Height of the viewer (default: auto) */
  height?: string | number
  /** Whether to show the status bar */
  showStatusBar?: boolean
}

const VncViewer = ({
  vncUrl,
  isConnected = false,
  isRunning = false,
  error = null,
  width = '100%',
  height = 'auto',
  showStatusBar = true,
}: VncViewerProps) => {
  const [isLoading, setIsLoading] = useState(false)
  const [iframeError, setIframeError] = useState<string | null>(null)
  const iframeRef = useRef<HTMLIFrameElement>(null)

  // Handle iframe load
  const handleIframeLoad = () => {
    setIsLoading(false)
    setIframeError(null)
  }

  // Handle iframe error
  const handleIframeError = () => {
    setIsLoading(false)
    setIframeError('Failed to load VNC stream')
  }

  // Reset loading state when URL changes
  useEffect(() => {
    if (vncUrl) {
      setIsLoading(true)
      setIframeError(null)
    } else {
      setIsLoading(false)
    }
  }, [vncUrl])

  // Determine the display state
  const showPlaceholder = !vncUrl || !isConnected
  const showError = error || iframeError
  const showLoadingSpinner = isLoading && vncUrl && isConnected

  return (
    <div
      className={styles.container}
      style={{
        width: typeof width === 'number' ? `${width}px` : width,
        height: typeof height === 'number' ? `${height}px` : height,
      }}
    >
      {/* Status bar */}
      {showStatusBar && (
        <div className={styles.statusBar}>
          <div className={styles.statusLeft}>
            <div
              className={`${styles.statusDot} ${
                isConnected ? (isRunning ? styles.running : styles.connected) : styles.disconnected
              }`}
            />
            <span className={styles.statusText}>
              {isConnected
                ? isRunning
                  ? 'Agent Running'
                  : 'Connected'
                : 'Disconnected'}
            </span>
          </div>
          <div className={styles.statusRight}>
            {vncUrl && (
              <span className={styles.vncBadge}>
                E2B Desktop
              </span>
            )}
          </div>
        </div>
      )}

      {/* Main content area */}
      <div className={styles.viewerArea}>
        {/* Loading spinner */}
        {showLoadingSpinner && (
          <div className={styles.loadingOverlay}>
            <div className={styles.spinner} />
            <span>Loading desktop stream...</span>
          </div>
        )}

        {/* Error state */}
        {showError && (
          <div className={styles.errorOverlay}>
            <svg
              className={styles.errorIcon}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span className={styles.errorText}>{showError}</span>
          </div>
        )}

        {/* Placeholder when not connected */}
        {showPlaceholder && !showError && (
          <div className={styles.placeholder}>
            <svg
              className={styles.placeholderIcon}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <rect x="2" y="3" width="20" height="14" rx="2" />
              <line x1="8" y1="21" x2="16" y2="21" />
              <line x1="12" y1="17" x2="12" y2="21" />
            </svg>
            <span className={styles.placeholderTitle}>Desktop Not Connected</span>
            <span className={styles.placeholderText}>
              Submit a task to start an E2B desktop session
            </span>
          </div>
        )}

        {/* VNC iframe */}
        {vncUrl && isConnected && !showError && (
          <iframe
            ref={iframeRef}
            className={styles.vncFrame}
            src={vncUrl}
            title="E2B Desktop"
            sandbox="allow-scripts allow-same-origin allow-popups"
            onLoad={handleIframeLoad}
            onError={handleIframeError}
          />
        )}
      </div>
    </div>
  )
}

export default VncViewer
