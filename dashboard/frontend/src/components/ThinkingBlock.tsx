import { useState, useEffect } from 'react'
import styles from './ThinkingBlock.module.css'
import MarkdownRenderer from './MarkdownRenderer'
import { getTranslateAttr } from '../hooks/useNoTranslate'

interface ThinkingBlockProps {
  content: string
  isStreaming?: boolean
  thinkingTime?: number // in seconds
}

const AUTO_COLLAPSE_THRESHOLD = 2000 // Auto collapse when content exceeds 500 characters

const ThinkingBlock = ({ content, isStreaming = false, thinkingTime }: ThinkingBlockProps) => {
  const [displayTime, setDisplayTime] = useState(0)
  const [isExpanded, setIsExpanded] = useState(true)
  const [hasAutoCollapsed, setHasAutoCollapsed] = useState(false)

  // Auto-collapse when content exceeds threshold (even during streaming)
  useEffect(() => {
    const shouldAutoCollapse = content.length > AUTO_COLLAPSE_THRESHOLD

    console.log('ThinkingBlock state:', {
      contentLength: content.length,
      threshold: AUTO_COLLAPSE_THRESHOLD,
      shouldAutoCollapse,
      isStreaming,
      hasAutoCollapsed,
      isExpanded
    })

    // Auto-collapse as soon as content exceeds threshold, even while streaming
    if (shouldAutoCollapse && !hasAutoCollapsed) {
      console.log(`✅ Auto-collapsing: content length ${content.length} > ${AUTO_COLLAPSE_THRESHOLD}`)
      setIsExpanded(false)
      setHasAutoCollapsed(true)
    }
  }, [content.length, hasAutoCollapsed])

  // Simulate thinking time counter when streaming
  useEffect(() => {
    if (isStreaming) {
      const interval = setInterval(() => {
        setDisplayTime(prev => prev + 0.1)
      }, 100)
      return () => clearInterval(interval)
    } else if (thinkingTime !== undefined) {
      setDisplayTime(thinkingTime)
    }
  }, [isStreaming, thinkingTime])

  if (!content || content.trim().length === 0) {
    return null
  }

  const formatTime = (seconds: number) => {
    if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
    return `${seconds.toFixed(1)}s`
  }

  return (
    <div className={styles.container} translate={getTranslateAttr(isStreaming)}>
      <button
        className={styles.header}
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <div className={styles.headerLeft}>
          <svg
            className={`${styles.icon} ${isExpanded ? styles.iconExpanded : ''}`}
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
          <svg
            className={styles.thinkingIcon}
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            {/* Light bulb icon */}
            <path d="M9 18h6" />
            <path d="M10 22h4" />
            <path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8c0-3.31-2.69-6-6-6S6 4.69 6 8c0 1.33.47 2.55 1.5 3.5.76.76 1.23 1.52 1.41 2.5" />
          </svg>
          <span className={`${styles.title} ${isStreaming ? styles.titleStreaming : ''}`}>
            {isStreaming ? 'Thinking' : 'Completed Deep Thinking'}
          </span>
        </div>
        <div className={styles.headerRight}>
          {displayTime > 0 && (
            <span className={styles.time}>{formatTime(displayTime)}</span>
          )}
          <svg
            className={styles.expandIcon}
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="7 13 12 18 17 13" />
            <polyline points="7 6 12 11 17 6" />
          </svg>
        </div>
      </button>
      {isStreaming && (
        <div className={styles.progressBar}>
          <div className={styles.progressFill} />
        </div>
      )}
      {isExpanded && (
        <div className={styles.content}>
          <MarkdownRenderer content={content} />
        </div>
      )}
    </div>
  )
}

export default ThinkingBlock

