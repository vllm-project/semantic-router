import { useEffect, useRef } from 'react'
import styles from './ThinkingAnimation.module.css'
import PlatformBranding from './PlatformBranding'

interface ThinkingAnimationProps {
  onComplete?: () => void
  thinkingProcess?: string
}

const ROUTING_STAGES = ['Classifying intent', 'Selecting route', 'Preparing response']

const ThinkingAnimation = ({ onComplete, thinkingProcess }: ThinkingAnimationProps) => {
  const thinkingContentRef = useRef<HTMLDivElement>(null)

  // Call onComplete when component unmounts (when parent hides it)
  useEffect(() => {
    return () => {
      onComplete?.()
    }
  }, [onComplete])

  // Auto-scroll to bottom when thinking process updates
  useEffect(() => {
    if (thinkingContentRef.current && thinkingProcess) {
      thinkingContentRef.current.scrollTop = thinkingContentRef.current.scrollHeight
    }
  }, [thinkingProcess])

  return (
    <div className={styles.overlay}>
      <div className={styles.container}>
        <div className={styles.routeHeader}>
          <span className={styles.routeEyebrow}>Semantic Router</span>
          <div className={styles.routeTrack} aria-hidden="true">
            <span className={styles.routeTrackProgress} />
          </div>
        </div>
        <div className={styles.routeStages} aria-label="Routing request">
          {ROUTING_STAGES.map((stage, index) => (
            <div key={stage} className={styles.routeStage} style={{ animationDelay: `${index * 0.36}s` }}>
              <span className={styles.routeStageIndex}>{String(index + 1).padStart(2, '0')}</span>
              <span>{stage}</span>
            </div>
          ))}
        </div>
        <div className={styles.statusText}>
          Routing request through the local model fleet
        </div>

        {thinkingProcess && (
          <div ref={thinkingContentRef} className={styles.thinkingContent}>
            <div className={styles.thinkingLabel}>Thinking Process:</div>
            <pre className={styles.thinkingText}>{thinkingProcess}</pre>
          </div>
        )}

        <PlatformBranding variant="default" />
      </div>
    </div>
  )
}

export default ThinkingAnimation
