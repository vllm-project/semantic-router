import {
  X_VSR_CACHE_HIT,
  X_VSR_CONTEXT_TOKEN_COUNT,
  X_VSR_EFFECTIVE_INPUT_TOKENS,
  X_VSR_EFFECTIVE_MAX_OUTPUT_TOKENS,
  X_VSR_FAST_RESPONSE,
  X_VSR_LEARNING_ACTIONS,
  X_VSR_LEARNING_METHODS,
  X_VSR_LEARNING_REASONS,
  X_VSR_LEARNING_SCOPES,
  X_VSR_LOOPER_ALGORITHM,
  X_VSR_LOOPER_COMPLETION_TOKENS,
  X_VSR_LOOPER_ITERATIONS,
  X_VSR_LOOPER_LATENCY_MS,
  X_VSR_LOOPER_MODEL,
  X_VSR_LOOPER_MODELS_USED,
  X_VSR_LOOPER_PROMPT_TOKENS,
  X_VSR_LOOPER_TOTAL_TOKENS,
  X_VSR_MATCHED_AUTHZ,
  X_VSR_MATCHED_COMPLEXITY,
  X_VSR_MATCHED_CONTEXT,
  X_VSR_MATCHED_CONVERSATION,
  X_VSR_MATCHED_DECISION_MODEL,
  X_VSR_MATCHED_DOMAINS,
  X_VSR_MATCHED_EMBEDDINGS,
  X_VSR_MATCHED_EVENT,
  X_VSR_MATCHED_FACT_CHECK,
  X_VSR_MATCHED_HALLUCINATION,
  X_VSR_MATCHED_INPUT_MODALITY,
  X_VSR_MATCHED_JAILBREAK,
  X_VSR_MATCHED_KB,
  X_VSR_MATCHED_KEYWORDS,
  X_VSR_MATCHED_LANGUAGE,
  X_VSR_MATCHED_MODALITY,
  X_VSR_MATCHED_PII,
  X_VSR_MATCHED_PREFERENCE,
  X_VSR_MATCHED_PROJECTIONS,
  X_VSR_MATCHED_REASK,
  X_VSR_MATCHED_SAFETY,
  X_VSR_MATCHED_STRUCTURE,
  X_VSR_MATCHED_USER_FEEDBACK,
  X_VSR_RESPONSE_PATH,
  X_VSR_RESPONSE_WARNINGS,
  X_VSR_RETENTION_DROP,
  X_VSR_RETENTION_KEEP_CURRENT_MODEL,
  X_VSR_RETENTION_PREFER_PREFIX,
  X_VSR_RETENTION_TTL_TURNS,
  X_VSR_ROUTING_LATENCY_MS,
  X_VSR_SELECTED_ALGORITHM,
  X_VSR_SELECTED_CONFIDENCE,
  X_VSR_SELECTED_DECISION,
  X_VSR_SELECTED_MODALITY,
  X_VSR_SELECTED_MODEL,
  X_VSR_SELECTED_REASONING,
  X_VSR_SELECTED_RECIPE,
} from '../generated/vsrHeaders'
import styles from './HeaderDisplay.module.css'
import ProductIcon from './ProductIcon'
import { formatLearningHeaderValue, isLearningHeader } from './headerLearningDisplay'
import { PLAYGROUND_LATENCY_MS, PLAYGROUND_TTFT_MS } from './playgroundTimingHeaders'
import { formatRoutingMetadataValue, MATCHED_SIGNAL_HEADER_PREFIX } from './routingMetadataDisplay'

interface HeaderDisplayProps {
  headers: Record<string, string>
}

// Header metadata for display
const HEADER_INFO: Record<
  string,
  { label: string; type: 'info' | 'success' | 'warning' | 'danger' }
> = {
  // Response outcome. The schema version is intentionally not user-facing.
  [X_VSR_RESPONSE_PATH]: {
    label: 'Response Path',
    type: 'info',
  },
  [X_VSR_SELECTED_MODEL]: {
    label: 'Model',
    type: 'info',
  },
  [X_VSR_EFFECTIVE_INPUT_TOKENS]: {
    label: 'Rendered input tokens',
    type: 'info',
  },
  [X_VSR_EFFECTIVE_MAX_OUTPUT_TOKENS]: {
    label: 'Output token limit',
    type: 'info',
  },
  [X_VSR_SELECTED_ALGORITHM]: {
    label: 'Algorithm',
    type: 'info',
  },
  [X_VSR_SELECTED_DECISION]: {
    label: 'Decision',
    type: 'info',
  },
  [X_VSR_SELECTED_RECIPE]: {
    label: 'Recipe',
    type: 'info',
  },
  [X_VSR_SELECTED_CONFIDENCE]: {
    label: 'Decision Confidence',
    type: 'info',
  },
  [X_VSR_SELECTED_MODALITY]: {
    label: 'Modality',
    type: 'info',
  },
  [X_VSR_CACHE_HIT]: {
    label: 'Cache',
    type: 'success',
  },
  [X_VSR_SELECTED_REASONING]: {
    label: 'Reasoning',
    type: 'info',
  },
  [X_VSR_LEARNING_METHODS]: {
    label: 'Learning',
    type: 'info',
  },
  [X_VSR_LEARNING_ACTIONS]: {
    label: 'Learning Action',
    type: 'info',
  },
  [X_VSR_LEARNING_SCOPES]: {
    label: 'Learning Scope',
    type: 'info',
  },
  [X_VSR_LEARNING_REASONS]: {
    label: 'Learning Reason',
    type: 'info',
  },
  [X_VSR_FAST_RESPONSE]: {
    label: 'Fast Response',
    type: 'success',
  },
  [X_VSR_RESPONSE_WARNINGS]: {
    label: 'Response Warnings',
    type: 'warning',
  },
  [X_VSR_MATCHED_KEYWORDS]: {
    label: 'Keywords',
    type: 'info',
  },
  [X_VSR_MATCHED_EMBEDDINGS]: {
    label: 'Embeddings',
    type: 'info',
  },
  [X_VSR_MATCHED_DOMAINS]: {
    label: 'Domain',
    type: 'info',
  },
  [X_VSR_MATCHED_FACT_CHECK]: {
    label: 'Fact Check Signal',
    type: 'info',
  },
  [X_VSR_MATCHED_USER_FEEDBACK]: {
    label: 'User Feedback',
    type: 'info',
  },
  [X_VSR_MATCHED_REASK]: {
    label: 'Reask',
    type: 'info',
  },
  [X_VSR_MATCHED_PREFERENCE]: {
    label: 'Preference',
    type: 'info',
  },
  [X_VSR_MATCHED_LANGUAGE]: {
    label: 'Language',
    type: 'info',
  },
  [X_VSR_MATCHED_CONTEXT]: {
    label: 'Context',
    type: 'info',
  },
  [X_VSR_MATCHED_STRUCTURE]: {
    label: 'Structure',
    type: 'info',
  },
  [X_VSR_CONTEXT_TOKEN_COUNT]: {
    label: 'Context Count',
    type: 'info',
  },
  [X_VSR_MATCHED_COMPLEXITY]: {
    label: 'Complexity',
    type: 'info',
  },
  [X_VSR_MATCHED_MODALITY]: {
    label: 'Modality',
    type: 'info',
  },
  [X_VSR_MATCHED_AUTHZ]: {
    label: 'Authz',
    type: 'info',
  },
  [X_VSR_MATCHED_JAILBREAK]: {
    label: 'Jailbreak Signal',
    type: 'danger',
  },
  [X_VSR_MATCHED_SAFETY]: {
    label: 'Safety Signal',
    type: 'warning',
  },
  [X_VSR_MATCHED_HALLUCINATION]: {
    label: 'Hallucination Signal',
    type: 'warning',
  },
  [X_VSR_MATCHED_PII]: {
    label: 'PII Signal',
    type: 'warning',
  },
  [X_VSR_MATCHED_KB]: {
    label: 'Knowledge Base',
    type: 'info',
  },
  [X_VSR_MATCHED_CONVERSATION]: {
    label: 'Conversation Signal',
    type: 'info',
  },
  [X_VSR_MATCHED_EVENT]: {
    label: 'Event Signal',
    type: 'info',
  },
  [X_VSR_MATCHED_INPUT_MODALITY]: {
    label: 'Input Modality Signal',
    type: 'info',
  },
  [X_VSR_MATCHED_DECISION_MODEL]: {
    label: 'System One Answers',
    type: 'info',
  },
  [X_VSR_MATCHED_PROJECTIONS]: {
    label: 'Projection',
    type: 'info',
  },
  // Looper headers
  [X_VSR_LOOPER_MODEL]: {
    label: 'Final Model',
    type: 'info',
  },
  [X_VSR_LOOPER_MODELS_USED]: {
    label: 'Collaborative Models',
    type: 'success',
  },
  [X_VSR_LOOPER_ITERATIONS]: {
    label: 'Iterations',
    type: 'info',
  },
  [X_VSR_LOOPER_ALGORITHM]: {
    label: 'Algorithm',
    type: 'info',
  },
  [X_VSR_LOOPER_LATENCY_MS]: {
    label: 'Looper latency',
    type: 'info',
  },
  [X_VSR_LOOPER_PROMPT_TOKENS]: {
    label: 'Looper Prompt Tokens',
    type: 'info',
  },
  [X_VSR_LOOPER_COMPLETION_TOKENS]: {
    label: 'Looper Completion Tokens',
    type: 'info',
  },
  [X_VSR_LOOPER_TOTAL_TOKENS]: {
    label: 'Looper Total Tokens',
    type: 'info',
  },
  [X_VSR_ROUTING_LATENCY_MS]: {
    label: 'Routing Latency',
    type: 'info',
  },
  [PLAYGROUND_LATENCY_MS]: {
    label: 'Latency',
    type: 'info',
  },
  [PLAYGROUND_TTFT_MS]: {
    label: 'TTFT',
    type: 'info',
  },
  // Retention directive headers (issue #2009)
  [X_VSR_RETENTION_DROP]: {
    label: 'Retention: Drop',
    type: 'warning',
  },
  [X_VSR_RETENTION_TTL_TURNS]: {
    label: 'Retention TTL (turns)',
    type: 'info',
  },
  [X_VSR_RETENTION_KEEP_CURRENT_MODEL]: {
    label: 'Keep Current Model',
    type: 'info',
  },
  [X_VSR_RETENTION_PREFER_PREFIX]: {
    label: 'Prefer Prefix',
    type: 'info',
  },
}

function shouldSummarizeHeaderValue(key: string, values: string[]): boolean {
  return (
    values.length > 1 &&
    (key.startsWith(MATCHED_SIGNAL_HEADER_PREFIX) || key === X_VSR_LOOPER_MODELS_USED)
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
    .map((value) => formatRoutingMetadataValue(key, value))

  if (!shouldSummarizeHeaderValue(key, values)) {
    return values.join(', ')
  }

  return `${values[0]} +${values.length - 1}`
}

function routingHeaderPriority(key: string): number {
  if (key.startsWith(MATCHED_SIGNAL_HEADER_PREFIX)) return 0
  if (key === X_VSR_SELECTED_DECISION) return 1
  if (key === X_VSR_SELECTED_ALGORITHM || key === X_VSR_LOOPER_ALGORITHM) return 2
  if (
    key === X_VSR_SELECTED_MODEL ||
    key === X_VSR_LOOPER_MODEL ||
    key === X_VSR_LOOPER_MODELS_USED
  )
    return 3
  if (key === X_VSR_RESPONSE_PATH) return 4
  return 5
}

const HeaderDisplay = ({ headers }: HeaderDisplayProps) => {
  // Filter to only show headers that exist
  const hasSelectedAlgorithm = Boolean(headers[X_VSR_SELECTED_ALGORITHM])
  const displayHeaders = Object.entries(headers)
    .filter(
      ([key]) => key in HEADER_INFO && !(hasSelectedAlgorithm && key === X_VSR_LOOPER_ALGORITHM),
    )
    .sort(
      ([leftKey], [rightKey]) => routingHeaderPriority(leftKey) - routingHeaderPriority(rightKey),
    )

  if (displayHeaders.length === 0) {
    return null
  }

  const primaryKeys = [
    X_VSR_SELECTED_DECISION,
    X_VSR_SELECTED_ALGORITHM,
    X_VSR_LOOPER_ALGORITHM,
    X_VSR_SELECTED_MODEL,
    X_VSR_LOOPER_MODEL,
  ]
  const performanceKeys = [
    X_VSR_ROUTING_LATENCY_MS,
    PLAYGROUND_LATENCY_MS,
    PLAYGROUND_TTFT_MS,
    X_VSR_LOOPER_LATENCY_MS,
  ]
  const primaryHeaders = displayHeaders.filter(([key]) => primaryKeys.includes(key))
  const performanceHeaders = displayHeaders.filter(([key]) => performanceKeys.includes(key))
  const detailHeaders = displayHeaders.filter(
    ([key]) => !primaryKeys.includes(key) && !performanceKeys.includes(key),
  )

  const renderHeader = ([key, value]: [string, string]) => {
    const info = HEADER_INFO[key]
    const displayValue = summarizeHeaderValue(key, value)
    const unit = key.endsWith('-ms') ? ' ms' : ''
    return (
      <div
        key={key}
        className={`${styles.header} ${styles[info.type]}`}
        title={`${info.label}: ${displayValue}${unit}`}
      >
        <span className={styles.label}>{info.label}</span>
        <span className={styles.value}>
          {displayValue}
          {unit}
        </span>
      </div>
    )
  }

  const renderPerformance = ([key, value]: [string, string]) => {
    const info = HEADER_INFO[key]
    const displayValue = summarizeHeaderValue(key, value)
    return (
      <div
        key={key}
        className={styles.performanceMetric}
        title={`${info.label}: ${displayValue} ms`}
      >
        <span>{info.label}</span>
        <strong>{displayValue}</strong>
        <small>ms</small>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.headers}>
        {primaryHeaders.map(renderHeader)}
        {performanceHeaders.length > 0 ? (
          <div className={styles.performance}>{performanceHeaders.map(renderPerformance)}</div>
        ) : null}
      </div>
      {detailHeaders.length > 0 ? (
        <details className={styles.details}>
          <summary className={styles.detailsToggle} aria-label="Show response details">
            <span>Response details</span>
            <ProductIcon name="chevron-down" width={13} height={13} />
          </summary>
          <div className={styles.detailsPanel}>{detailHeaders.map(renderHeader)}</div>
        </details>
      ) : null}
    </div>
  )
}

export default HeaderDisplay
