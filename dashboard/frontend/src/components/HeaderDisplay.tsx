import styles from './HeaderDisplay.module.css'

interface HeaderDisplayProps {
  headers: Record<string, string>
}

// Header metadata for display
const HEADER_INFO: Record<string, { label: string; type: 'info' | 'success' | 'warning' | 'danger' }> = {
  'x-vsr-selected-model': {
    label: 'Model',
    type: 'info',
  },
  'x-vsr-selected-decision': {
    label: 'Decision',
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
  'x-vsr-jailbreak-blocked': {
    label: 'Jailbreak Blocked',
    type: 'danger',
  },
  'x-vsr-pii-violation': {
    label: 'PII Violation',
    type: 'danger',
  },
  'x-vsr-hallucination-detected': {
    label: 'Hallucination',
    type: 'warning',
  },
  'x-vsr-fact-check-needed': {
    label: 'Fact Check',
    type: 'info',
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
  'x-vsr-matched-preference': {
    label: 'Preference',
    type: 'info',
  },
  'x-vsr-matched-language': {
    label: 'Language',
    type: 'info',
  },
  'x-vsr-matched-latency': {
    label: 'Latency',
    type: 'info',
  },
  'x-vsr-matched-context': {
    label: 'Context',
    type: 'info',
  },
  'x-vsr-context-token-count': {
    label: 'Context Count',
    type: 'info',
  },
  // Looper headers
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
}

const HeaderDisplay = ({ headers }: HeaderDisplayProps) => {
  // Filter to only show headers that exist
  const displayHeaders = Object.entries(headers).filter(([key]) => key in HEADER_INFO)

  if (displayHeaders.length === 0) {
    return null
  }

  return (
    <div className={styles.container}>
      <div className={styles.headers}>
        {displayHeaders.map(([key, value]) => {
          const info = HEADER_INFO[key]
          return (
            <div key={key} className={`${styles.header} ${styles[info.type]}`} title={`${info.label}: ${value}`}>
              <span className={styles.label}>{info.label}</span>
              <span className={styles.value}>{value}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default HeaderDisplay

