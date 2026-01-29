import { useEffect, useState } from 'react'
import styles from './HeaderReveal.module.css'

interface HeaderRevealProps {
  headers: Record<string, string>
  onComplete?: () => void
  displayDuration?: number // How long to show headers before auto-closing
}

// Header metadata for display
const HEADER_INFO: Record<string, { label: string; description: string }> = {
  // Signal headers
  'x-vsr-matched-keywords': {
    label: 'Keywords',
    description: 'Matched keyword patterns',
  },
  'x-vsr-matched-embeddings': {
    label: 'Embeddings',
    description: 'Semantic similarity match',
  },
  'x-vsr-matched-domains': {
    label: 'Domain',
    description: 'Domain classification result',
  },
  'x-vsr-matched-fact-check': {
    label: 'Fact Check',
    description: 'Fact check signal triggered',
  },
  'x-vsr-matched-user-feedback': {
    label: 'User Feedback',
    description: 'Based on user feedback patterns',
  },
  'x-vsr-matched-preference': {
    label: 'Preference',
    description: 'User preference match',
  },
  'x-vsr-matched-language': {
    label: 'Language',
    description: 'Detected language match',
  },
  'x-vsr-matched-latency': {
    label: 'Latency',
    description: 'Matched latency rule based on model TPOT',
  },
  'x-vsr-matched-context': {
    label: 'Context',
    description: 'Token count-based context classification',
  },
  'x-vsr-matched-complexity': {
    label: 'Complexity',
    description: 'Query complexity classification (hard/easy/medium)',
  },
  // Decision headers
  'x-vsr-selected-decision': {
    label: 'Routing Decision',
    description: 'The decision rule that was applied',
  },
  // Model selection headers
  'x-vsr-selected-model': {
    label: 'Selected Model',
    description: 'The model chosen by the router',
  },
  // Plugin status headers
  'x-vsr-cache-hit': {
    label: 'Cache Status',
    description: 'Whether the response was served from cache',
  },
  'x-vsr-selected-reasoning': {
    label: 'Reasoning Mode',
    description: 'The reasoning strategy applied',
  },
  'x-vsr-context-token-count': {
    label: 'Context Count',
    description: 'Estimated token count for the request',
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

  // Group headers by category based on the comments in HEADER_INFO
  const groupedHeaders = {
    // Signal headers: all x-vsr-matched-*
    signals: displayHeaders.filter(([key]) =>
      key.startsWith('x-vsr-matched-')
    ),
    // Decision headers: selected-decision
    decision: displayHeaders.filter(([key]) =>
      key === 'x-vsr-selected-decision'
    ),
    // Model selection headers: selected-model
    model: displayHeaders.filter(([key]) =>
      key === 'x-vsr-selected-model'
    ),
    // Plugin status headers: cache, reasoning, context, security, quality
    plugin: displayHeaders.filter(([key]) =>
      key === 'x-vsr-cache-hit' ||
      key === 'x-vsr-selected-reasoning' ||
      key === 'x-vsr-context-token-count' ||
      key === 'x-vsr-jailbreak-blocked' ||
      key === 'x-vsr-pii-violation' ||
      key === 'x-vsr-hallucination-detected' ||
      key === 'x-vsr-fact-check-needed'
    ),
    // Looper headers: all x-vsr-looper-*
    looper: displayHeaders.filter(([key]) =>
      key.startsWith('x-vsr-looper-')
    ),
  }

  const renderSection = (title: string, items: [string, string][], isPrimary = false) => {
    if (items.length === 0) return null
    return (
      <div key={title} className={`${styles.section} ${isPrimary ? styles.sectionPrimary : ''}`}>
        <div className={styles.sectionTitle}>{title}</div>
        <div className={styles.sectionItems}>
          {items.map(([key, value]) => {
            const info = HEADER_INFO[key]
            return (
              <div key={key} className={`${styles.headerItem} ${isPrimary ? styles.headerItemPrimary : ''}`}>
                <div className={styles.headerLabel}>{info.label}</div>
                <div className={styles.headerValue}>{value}</div>
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  return (
    <div className={`${styles.overlay} ${!isVisible ? styles.fadeOut : ''}`}>
      <div className={styles.container}>
        <div className={styles.title}>Signal Driven Decision</div>
        <div className={styles.sections}>
          {renderSection('MODEL', groupedHeaders.model, true)}
          {renderSection('DECISION', groupedHeaders.decision, true)}
          {renderSection('SIGNALS', groupedHeaders.signals)}
          {renderSection('PLUGIN', groupedHeaders.plugin)}
          {renderSection('LOOPER', groupedHeaders.looper)}
        </div>
        <div className={styles.hint}>
          Response will appear shortly...
        </div>
      </div>
    </div>
  )
}

export default HeaderReveal

