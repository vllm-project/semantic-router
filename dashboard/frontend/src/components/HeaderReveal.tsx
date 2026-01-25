import { useEffect, useState } from 'react'
import styles from './HeaderReveal.module.css'

interface HeaderRevealProps {
  headers: Record<string, string>
  onComplete?: () => void
  displayDuration?: number // How long to show headers before auto-closing
}

// Header metadata for display
const HEADER_INFO: Record<string, { label: string; description: string }> = {
  'x-vsr-selected-model': {
    label: 'Selected Model',
    description: 'The model chosen by the router',
  },
  'x-vsr-selected-decision': {
    label: 'Routing Decision',
    description: 'The decision rule that was applied',
  },
  'x-vsr-cache-hit': {
    label: 'Cache Status',
    description: 'Whether the response was served from cache',
  },
  'x-vsr-selected-reasoning': {
    label: 'Reasoning Mode',
    description: 'The reasoning strategy applied',
  },
  'x-vsr-jailbreak-blocked': {
    label: 'Security: Jailbreak',
    description: 'Jailbreak attempt detected and blocked',
  },
  'x-vsr-pii-violation': {
    label: 'Security: PII',
    description: 'Personal information detected',
  },
  'x-vsr-hallucination-detected': {
    label: 'Quality: Hallucination',
    description: 'Potential hallucination detected',
  },
  'x-vsr-fact-check-needed': {
    label: 'Quality: Fact Check',
    description: 'Fact checking recommended',
  },
  'x-vsr-matched-keywords': {
    label: 'Signal: Keywords',
    description: 'Matched keyword patterns',
  },
  'x-vsr-matched-embeddings': {
    label: 'Signal: Embeddings',
    description: 'Semantic similarity match',
  },
  'x-vsr-matched-domains': {
    label: 'Signal: Domain',
    description: 'Domain classification result',
  },
  'x-vsr-matched-fact-check': {
    label: 'Signal: Fact Check',
    description: 'Fact check signal triggered',
  },
  'x-vsr-matched-user-feedback': {
    label: 'Signal: User Feedback',
    description: 'Based on user feedback patterns',
  },
  'x-vsr-matched-preference': {
    label: 'Signal: Preference',
    description: 'User preference match',
  },
  'x-vsr-matched-language': {
    label: 'Signal: Language',
    description: 'Detected language match',
  },
  'x-vsr-matched-latency': {
    label: 'Signal: Latency',
    description: 'Matched latency rule based on model TPOT',
  },
  'x-vsr-matched-context': {
    label: 'Signal: Context',
    description: 'Token count-based context classification',
  },
  'x-vsr-context-token-count': {
    label: 'Context Count',
    description: 'Estimated token count for the request',
  },
  // Looper headers
  'x-vsr-looper-model': {
    label: 'Final Model',
    description: 'The model that produced the final response',
  },
  'x-vsr-looper-models-used': {
    label: 'Collaborative Models',
    description: 'All models called during multi-model routing',
  },
  'x-vsr-looper-iterations': {
    label: 'Iterations',
    description: 'Number of model calls made',
  },
  'x-vsr-looper-algorithm': {
    label: 'Algorithm',
    description: 'The multi-model algorithm used (confidence, ratings)',
  },
}

const HeaderReveal = ({ headers, onComplete, displayDuration = 2000 }: HeaderRevealProps) => {
  const [isVisible, setIsVisible] = useState(true)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false)
      setTimeout(() => onComplete?.(), 300) // Wait for fade out animation
    }, displayDuration)

    return () => clearTimeout(timer)
  }, [displayDuration, onComplete])

  if (!isVisible) {
    return null
  }

  const displayHeaders = Object.entries(headers).filter(([key]) => key in HEADER_INFO)

  return (
    <div className={`${styles.overlay} ${!isVisible ? styles.fadeOut : ''}`}>
      <div className={styles.container}>
        <div className={styles.title}>Signal Driven Decision</div>
        <div className={styles.headerGrid}>
          {displayHeaders.map(([key, value]) => {
            const info = HEADER_INFO[key]
            return (
              <div key={key} className={styles.headerItem}>
                <div className={styles.headerLabel}>{info.label}</div>
                <div className={styles.headerValue}>{value}</div>
                <div className={styles.headerDescription}>{info.description}</div>
              </div>
            )
          })}
        </div>
        <div className={styles.hint}>
          Response will appear shortly...
        </div>
      </div>
    </div>
  )
}

export default HeaderReveal

