import { useEffect, useState } from 'react'
import styles from './HeaderReveal.module.css'
import { formatLearningHeaderValue, isLearningHeader } from './headerLearningDisplay'

interface HeaderRevealProps {
  headers: Record<string, string>
  onComplete?: () => void
  displayDuration?: number // How long to show headers before auto-closing
}

// Header metadata for display
const HEADER_INFO: Record<string, { label: string; description: string }> = {
  // Response contract (v0.4 keystone) headers (#2203)
  'x-vsr-schema-version': {
    label: 'Schema Version',
    description: 'Response header contract revision',
  },
  'x-vsr-response-path': {
    label: 'Response Path',
    description: 'How the response was produced (upstream, cache, fast_response, looper, ...)',
  },
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
  'x-vsr-matched-reask': {
    label: 'Reask',
    description: 'Detected repeated-question dissatisfaction across recent user turns',
  },
  'x-vsr-matched-preference': {
    label: 'Preference',
    description: 'User preference match',
  },
  'x-vsr-matched-language': {
    label: 'Language',
    description: 'Detected language match',
  },
  'x-vsr-matched-context': {
    label: 'Context',
    description: 'Token count-based context classification',
  },
  'x-vsr-matched-structure': {
    label: 'Structure',
    description: 'Matched request-shape and structural heuristic signals',
  },
  'x-vsr-matched-complexity': {
    label: 'Complexity',
    description: 'Query complexity classification (hard/easy/medium)',
  },
  'x-vsr-matched-modality': {
    label: 'Modality',
    description: 'Matched modality classification signal',
  },
  'x-vsr-matched-authz': {
    label: 'Authz',
    description: 'Matched authorization routing signal',
  },
  'x-vsr-matched-jailbreak': {
    label: 'Jailbreak',
    description: 'Jailbreak detection signal matched',
  },
  'x-vsr-matched-pii': {
    label: 'PII',
    description: 'PII detection signal matched',
  },
  'x-vsr-matched-kb': {
    label: 'Knowledge Base',
    description: 'Matched knowledge-base label or group routing signal',
  },
  'x-vsr-matched-conversation': {
    label: 'Conversation',
    description: 'Conversation structure signal matched (e.g. multi-turn, tool usage)',
  },
  'x-vsr-matched-event': {
    label: 'Event',
    description: 'Event signal matched',
  },
  'x-vsr-matched-projections': {
    label: 'Projection',
    description: 'Projection mapping output matched',
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
  'x-vsr-selected-modality': {
    label: 'Selected Modality',
    description: 'The modality chosen by the router',
  },
  // Router Learning headers
  'x-vsr-learning-methods': {
    label: 'Learning',
    description: 'Router Learning methods summarized by this response',
  },
  'x-vsr-learning-actions': {
    label: 'Learning Action',
    description: 'Method-keyed Router Learning actions',
  },
  'x-vsr-learning-scopes': {
    label: 'Learning Scope',
    description: 'Method-keyed identity scopes used by Router Learning',
  },
  'x-vsr-learning-reasons': {
    label: 'Learning Reason',
    description: 'Method-keyed compact reasons for Router Learning actions',
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
  'x-vsr-fast-response': {
    label: 'Fast Response',
    description: 'Request short-circuited by fast_response plugin',
  },
  'x-vsr-response-warnings': {
    label: 'Quality: Response Warnings',
    description:
      'Comma-separated response-quality warnings (hallucination, unverified_factual, response_jailbreak)',
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
    description: 'The multi-model algorithm used (confidence, ratings, remom, fusion)',
  },
  // Retention directive headers (issue #2009)
  'x-vsr-retention-drop': {
    label: 'Retention: Drop',
    description: 'Matched decision asked to skip the semantic-cache write',
  },
  'x-vsr-retention-ttl-turns': {
    label: 'Retention TTL (turns)',
    description: 'Per-entry semantic-cache lifetime override, in conversation turns',
  },
  'x-vsr-retention-keep-current-model': {
    label: 'Keep Current Model',
    description: 'Forces the model-switch gate to stay on the current model',
  },
  'x-vsr-retention-prefer-prefix': {
    label: 'Prefer Prefix',
    description: 'Hints the inference pool to keep the prompt prefix / KV cache warm',
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
    // Response contract (keystone) headers: schema-version + response-path
    response: displayHeaders.filter(
      ([key]) => key === 'x-vsr-schema-version' || key === 'x-vsr-response-path',
    ),
    // Signal headers: all x-vsr-matched-*
    signals: displayHeaders.filter(([key]) => key.startsWith('x-vsr-matched-')),
    // Decision headers: selected-decision
    decision: displayHeaders.filter(([key]) => key === 'x-vsr-selected-decision'),
    // Model selection headers: selected-model
    model: displayHeaders.filter(
      ([key]) => key === 'x-vsr-selected-model' || key === 'x-vsr-selected-modality',
    ),
    // Router Learning headers: bounded primary adaptation summary
    learning: displayHeaders.filter(([key]) => key.startsWith('x-vsr-learning')),
    // Plugin status headers: cache, reasoning, context, security, quality
    plugin: displayHeaders.filter(
      ([key]) =>
        key === 'x-vsr-cache-hit' ||
        key === 'x-vsr-selected-reasoning' ||
        key === 'x-vsr-fast-response' ||
        key === 'x-vsr-context-token-count' ||
        key === 'x-vsr-response-warnings',
    ),
    // Looper headers: all x-vsr-looper-*
    looper: displayHeaders.filter(([key]) => key.startsWith('x-vsr-looper-')),
    // Retention directive headers: all x-vsr-retention-*
    retention: displayHeaders.filter(([key]) => key.startsWith('x-vsr-retention-')),
  }

  const renderSection = (title: string, items: [string, string][], isPrimary = false) => {
    if (items.length === 0) return null
    return (
      <div key={title} className={`${styles.section} ${isPrimary ? styles.sectionPrimary : ''}`}>
        <div className={styles.sectionTitle}>{title}</div>
        <div className={styles.sectionItems}>
          {items.map(([key, value]) => {
            const info = HEADER_INFO[key]
            const displayValue = isLearningHeader(key)
              ? formatLearningHeaderValue(key, value)
              : value
            return (
              <div
                key={key}
                className={`${styles.headerItem} ${isPrimary ? styles.headerItemPrimary : ''}`}
              >
                <div className={styles.headerLabel}>{info.label}</div>
                <div className={styles.headerValue}>{displayValue}</div>
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
        <div className={styles.title}>Router Decision</div>
        <div className={styles.sections}>
          {renderSection('RESPONSE', groupedHeaders.response, true)}
          {renderSection('MODEL', groupedHeaders.model, true)}
          {renderSection('DECISION', groupedHeaders.decision, true)}
          {renderSection('LEARNING', groupedHeaders.learning)}
          {renderSection('SIGNALS', groupedHeaders.signals)}
          {renderSection('PLUGIN', groupedHeaders.plugin)}
          {renderSection('LOOPER', groupedHeaders.looper)}
          {renderSection('RETENTION', groupedHeaders.retention)}
        </div>
        <div className={styles.hint}>Response will appear shortly...</div>
      </div>
    </div>
  )
}

export default HeaderReveal
