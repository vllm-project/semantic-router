import styles from './HeaderDisplay.module.css'
import { formatLearningHeaderValue, isLearningHeader } from './headerLearningDisplay'

interface HeaderDisplayProps {
  headers: Record<string, string>
}

// Header metadata for display
const HEADER_INFO: Record<
  string,
  { label: string; type: 'info' | 'success' | 'warning' | 'danger' }
> = {
  // Response outcome. The schema version is intentionally not user-facing.
  'x-vsr-response-path': {
    label: 'Response Path',
    type: 'info',
  },
  'x-vsr-selected-model': {
    label: 'Model',
    type: 'info',
  },
  'x-vsr-selected-algorithm': {
    label: 'Algorithm',
    type: 'info',
  },
  'x-vsr-selected-decision': {
    label: 'Decision',
    type: 'info',
  },
  'x-vsr-selected-modality': {
    label: 'Modality',
    type: 'info',
  },
  'x-vsr-cache-hit': {
    label: 'Cache',
    type: 'success',
  },
  'x-vsr-selected-reasoning': {
    label: 'Reasoning',
    type: 'info',
  },
  'x-vsr-learning-methods': {
    label: 'Learning',
    type: 'info',
  },
  'x-vsr-learning-actions': {
    label: 'Learning Action',
    type: 'info',
  },
  'x-vsr-learning-scopes': {
    label: 'Learning Scope',
    type: 'info',
  },
  'x-vsr-learning-reasons': {
    label: 'Learning Reason',
    type: 'info',
  },
  'x-vsr-fast-response': {
    label: 'Fast Response',
    type: 'success',
  },
  'x-vsr-response-warnings': {
    label: 'Response Warnings',
    type: 'warning',
  },
  'x-vsr-matched-keywords': {
    label: 'Keywords',
    type: 'info',
  },
  'x-vsr-matched-embeddings': {
    label: 'Embeddings',
    type: 'info',
  },
  'x-vsr-matched-domains': {
    label: 'Domain',
    type: 'info',
  },
  'x-vsr-matched-fact-check': {
    label: 'Fact Check Signal',
    type: 'info',
  },
  'x-vsr-matched-user-feedback': {
    label: 'User Feedback',
    type: 'info',
  },
  'x-vsr-matched-reask': {
    label: 'Reask',
    type: 'info',
  },
  'x-vsr-matched-preference': {
    label: 'Preference',
    type: 'info',
  },
  'x-vsr-matched-language': {
    label: 'Language',
    type: 'info',
  },
  'x-vsr-matched-context': {
    label: 'Context',
    type: 'info',
  },
  'x-vsr-matched-structure': {
    label: 'Structure',
    type: 'info',
  },
  'x-vsr-context-token-count': {
    label: 'Context Count',
    type: 'info',
  },
  'x-vsr-matched-complexity': {
    label: 'Complexity',
    type: 'info',
  },
  'x-vsr-matched-modality': {
    label: 'Modality',
    type: 'info',
  },
  'x-vsr-matched-authz': {
    label: 'Authz',
    type: 'info',
  },
  'x-vsr-matched-jailbreak': {
    label: 'Jailbreak Signal',
    type: 'danger',
  },
  'x-vsr-matched-pii': {
    label: 'PII Signal',
    type: 'warning',
  },
  'x-vsr-matched-kb': {
    label: 'Knowledge Base',
    type: 'info',
  },
  'x-vsr-matched-conversation': {
    label: 'Conversation Signal',
    type: 'info',
  },
  'x-vsr-matched-event': {
    label: 'Event Signal',
    type: 'info',
  },
  'x-vsr-matched-projections': {
    label: 'Projection',
    type: 'info',
  },
  // Looper headers
  'x-vsr-looper-model': {
    label: 'Final Model',
    type: 'info',
  },
  'x-vsr-looper-models-used': {
    label: 'Collaborative Models',
    type: 'success',
  },
  'x-vsr-looper-iterations': {
    label: 'Iterations',
    type: 'info',
  },
  'x-vsr-looper-algorithm': {
    label: 'Algorithm',
    type: 'info',
  },
  // Retention directive headers (issue #2009)
  'x-vsr-retention-drop': {
    label: 'Retention: Drop',
    type: 'warning',
  },
  'x-vsr-retention-ttl-turns': {
    label: 'Retention TTL (turns)',
    type: 'info',
  },
  'x-vsr-retention-keep-current-model': {
    label: 'Keep Current Model',
    type: 'info',
  },
  'x-vsr-retention-prefer-prefix': {
    label: 'Prefer Prefix',
    type: 'info',
  },
}

function shouldSummarizeHeaderValue(key: string, values: string[]): boolean {
  return (
    values.length > 1 && (key.startsWith('x-vsr-matched-') || key === 'x-vsr-looper-models-used')
  )
}

function summarizeHeaderValue(key: string, rawValue: string): string {
  if (isLearningHeader(key)) {
    return formatLearningHeaderValue(key, rawValue)
  }

  const values = rawValue
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)

  if (!shouldSummarizeHeaderValue(key, values)) {
    return rawValue
  }

  return `${values[0]} +${values.length - 1}`
}

function routingHeaderPriority(key: string): number {
  if (key.startsWith('x-vsr-matched-')) return 0
  if (key === 'x-vsr-selected-decision') return 1
  if (key === 'x-vsr-selected-algorithm' || key === 'x-vsr-looper-algorithm') return 2
  if (
    key === 'x-vsr-selected-model' ||
    key === 'x-vsr-looper-model' ||
    key === 'x-vsr-looper-models-used'
  )
    return 3
  if (key === 'x-vsr-response-path') return 4
  return 5
}

const HeaderDisplay = ({ headers }: HeaderDisplayProps) => {
  // Filter to only show headers that exist
  const hasSelectedAlgorithm = Boolean(headers['x-vsr-selected-algorithm'])
  const displayHeaders = Object.entries(headers)
    .filter(
      ([key]) => key in HEADER_INFO && !(hasSelectedAlgorithm && key === 'x-vsr-looper-algorithm'),
    )
    .sort(
      ([leftKey], [rightKey]) => routingHeaderPriority(leftKey) - routingHeaderPriority(rightKey),
    )

  if (displayHeaders.length === 0) {
    return null
  }

  return (
    <div className={styles.container}>
      <div className={styles.headers}>
        {displayHeaders.map(([key, value]) => {
          const info = HEADER_INFO[key]
          const displayValue = summarizeHeaderValue(key, value)
          return (
            <div
              key={key}
              className={`${styles.header} ${styles[info.type]}`}
              title={`${info.label}: ${value}`}
            >
              <span className={styles.label}>{info.label}</span>
              <span className={styles.value}>{displayValue}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default HeaderDisplay
