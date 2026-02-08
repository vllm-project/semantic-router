/**
 * ScreenshotViewer - Component to display browser screenshots with action overlays
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import styles from './ScreenshotViewer.module.css'

interface ClickTarget {
  x: number
  y: number
  label?: string
}

interface ScreenshotViewerProps {
  screenshot: string | null  // Base64 encoded image
  url?: string
  title?: string
  width?: number
  height?: number
  isLoading?: boolean
  error?: string
  clickTargets?: ClickTarget[]
  onClickCoordinate?: (x: number, y: number) => void
  showCoordinates?: boolean
}

const ScreenshotViewer = ({
  screenshot,
  url,
  title,
  width = 1280,
  height = 800,
  isLoading = false,
  error,
  clickTargets = [],
  onClickCoordinate,
  showCoordinates = false,
}: ScreenshotViewerProps) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  const [scale, setScale] = useState(1)

  // Calculate scale to fit container
  useEffect(() => {
    if (containerRef.current && width && height) {
      const containerWidth = containerRef.current.clientWidth
      const containerHeight = containerRef.current.clientHeight
      const scaleX = containerWidth / width
      const scaleY = containerHeight / height
      setScale(Math.min(scaleX, scaleY, 1))
    }
  }, [width, height])

  // Handle mouse move for coordinate display
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!showCoordinates || !containerRef.current) return

      const rect = containerRef.current.getBoundingClientRect()
      const x = Math.round((e.clientX - rect.left) / scale)
      const y = Math.round((e.clientY - rect.top) / scale)
      setMousePos({ x, y })
    },
    [showCoordinates, scale]
  )

  // Handle click for coordinate selection
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!onClickCoordinate || !containerRef.current) return

      const rect = containerRef.current.getBoundingClientRect()
      const x = Math.round((e.clientX - rect.left) / scale)
      const y = Math.round((e.clientY - rect.top) / scale)
      onClickCoordinate(x, y)
    },
    [onClickCoordinate, scale]
  )

  const handleMouseLeave = useCallback(() => {
    setMousePos(null)
  }, [])

  // Render loading state
  if (isLoading) {
    return (
      <div className={styles.container}>
        <div className={styles.loadingState}>
          <div className={styles.spinner} />
          <span>Loading browser...</span>
        </div>
      </div>
    )
  }

  // Render error state
  if (error) {
    return (
      <div className={styles.container}>
        <div className={styles.errorState}>
          <svg
            width="48"
            height="48"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  // Render empty state
  if (!screenshot) {
    return (
      <div className={styles.container}>
        <div className={styles.emptyState}>
          <svg
            width="48"
            height="48"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <rect x="2" y="3" width="20" height="14" rx="2" />
            <line x1="8" y1="21" x2="16" y2="21" />
            <line x1="12" y1="17" x2="12" y2="21" />
          </svg>
          <p>No screenshot available</p>
          <span>Start a browser session to see the page</span>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      {/* URL bar */}
      {url && (
        <div className={styles.urlBar}>
          <div className={styles.urlIcon}>
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="2" y1="12" x2="22" y2="12" />
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          </div>
          <span className={styles.urlText}>{url}</span>
          {title && <span className={styles.pageTitle}>{title}</span>}
        </div>
      )}

      {/* Screenshot container */}
      <div
        ref={containerRef}
        className={`${styles.screenshotContainer} ${onClickCoordinate ? styles.clickable : ''}`}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <img
          src={`data:image/png;base64,${screenshot}`}
          alt="Browser screenshot"
          className={styles.screenshot}
          style={{
            maxWidth: width * scale,
            maxHeight: height * scale,
          }}
          draggable={false}
        />

        {/* Click targets overlay */}
        {clickTargets.map((target, index) => (
          <div
            key={index}
            className={styles.clickTarget}
            style={{
              left: target.x * scale - 10,
              top: target.y * scale - 10,
            }}
          >
            <div className={styles.clickTargetRing} />
            {target.label && (
              <span className={styles.clickTargetLabel}>{target.label}</span>
            )}
          </div>
        ))}

        {/* Coordinate display */}
        {showCoordinates && mousePos && (
          <div
            className={styles.coordinateTooltip}
            style={{
              left: mousePos.x * scale + 15,
              top: mousePos.y * scale - 10,
            }}
          >
            ({mousePos.x}, {mousePos.y})
          </div>
        )}

        {/* Crosshair cursor when clickable */}
        {onClickCoordinate && mousePos && (
          <>
            <div
              className={styles.crosshairH}
              style={{ top: mousePos.y * scale }}
            />
            <div
              className={styles.crosshairV}
              style={{ left: mousePos.x * scale }}
            />
          </>
        )}
      </div>

      {/* Dimensions display */}
      {width && height && (
        <div className={styles.dimensions}>
          {width} x {height}
        </div>
      )}
    </div>
  )
}

export default ScreenshotViewer
