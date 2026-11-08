import {
  TRACE_ALGORITHM,
  TRACE_CACHE_SIMILARITY,
  TRACE_CONFIDENCE,
  TRACE_CONTEXT_TOKENS,
  TRACE_EFFECTIVE_INPUT_TOKENS,
  TRACE_EFFECTIVE_MAX_OUTPUT_TOKENS,
  TRACE_FALLBACK_ATTEMPTS,
  TRACE_LOOPER,
  TRACE_MODALITY,
  TRACE_REASONING,
  TRACE_RETENTION,
  TRACE_ROUTING_LATENCY_MS,
  TRACE_TRUNCATED,
  X_VSR_RESPONSE_PATH,
  X_VSR_SELECTED_DECISION,
  X_VSR_SELECTED_MODEL,
  X_VSR_SELECTED_RECIPE,
  X_VSR_WARNINGS,
} from '../generated/vsrHeaders'
import styles from './HeaderDisplay.module.css'
import ProductIcon from './ProductIcon'
import {
  formatLearningHeaderValue,
  isLearningHeader,
  LEARNING_ACTIONS,
  LEARNING_METHODS,
  LEARNING_REASONS,
  LEARNING_SCOPES,
} from './headerLearningDisplay'
import { PLAYGROUND_LATENCY_MS, PLAYGROUND_TTFT_MS } from './playgroundTimingHeaders'
import {
  formatRoutingMetadataValue,
  MATCHED_SIGNAL_KEY_PREFIX,
  matchedSignalKey,
  traceKey,
} from './routingMetadataDisplay'

interface HeaderDisplayProps {
  headers: Record<string, string>
}

const ALGORITHM = traceKey(TRACE_ALGORITHM)
const LOOPER_ALGORITHM = traceKey(`${TRACE_LOOPER}.algorithm`)
const LOOPER_MODEL = traceKey(`${TRACE_LOOPER}.model`)
const LOOPER_MODELS = traceKey(`${TRACE_LOOPER}.models`)
const LOOPER_LATENCY_MS = traceKey(`${TRACE_LOOPER}.latency_ms`)
const ROUTING_LATENCY_MS = traceKey(TRACE_ROUTING_LATENCY_MS)

type HeaderType = 'info' | 'success' | 'warning' | 'danger'

const SIGNAL_INFO: Record<string, { label: string; type: HeaderType }> = {
  keyword: { label: 'Keywords', type: 'info' },
  domain: { label: 'Domain', type: 'info' },
  fact_check: { label: 'Fact Check Signal', type: 'info' },
  user_feedback: { label: 'User Feedback', type: 'info' },
  turn_relation: { label: 'Turn Relation', type: 'info' },
  language: { label: 'Language', type: 'info' },
  context: { label: 'Context', type: 'info' },
  complexity: { label: 'Complexity', type: 'info' },
  modality: { label: 'Modality', type: 'info' },
  authz: { label: 'Authz', type: 'info' },
  jailbreak: { label: 'Jailbreak Signal', type: 'danger' },
  safety: { label: 'Safety Signal', type: 'warning' },
  hallucination: { label: 'Hallucination Signal', type: 'warning' },
  pii: { label: 'PII Signal', type: 'warning' },
  conversation: { label: 'Conversation Signal', type: 'info' },
  metadata: { label: 'Metadata Signal', type: 'info' },
  input_modality: { label: 'Input Modality Signal', type: 'info' },
  systemone: { label: 'System One Answers', type: 'info' },
  projection: { label: 'Projection', type: 'info' },
}

// What each display entry shows: the public headers, and the details of the
// trace a debug response carries.
const HEADER_INFO: Record<string, { label: string; type: HeaderType }> = {
  // Response outcome. The schema version is intentionally not user-facing.
  [X_VSR_RESPONSE_PATH]: { label: 'Response Path', type: 'info' },
  [X_VSR_SELECTED_MODEL]: { label: 'Model', type: 'info' },
  [traceKey(TRACE_EFFECTIVE_INPUT_TOKENS)]: { label: 'Rendered input tokens', type: 'info' },
  [traceKey(TRACE_EFFECTIVE_MAX_OUTPUT_TOKENS)]: { label: 'Output token limit', type: 'info' },
  [ALGORITHM]: { label: 'Algorithm', type: 'info' },
  [X_VSR_SELECTED_DECISION]: { label: 'Decision', type: 'info' },
  [X_VSR_SELECTED_RECIPE]: { label: 'Recipe', type: 'info' },
  [traceKey(TRACE_CONFIDENCE)]: { label: 'Decision Confidence', type: 'info' },
  [traceKey(TRACE_MODALITY)]: { label: 'Output', type: 'info' },
  [traceKey(TRACE_CACHE_SIMILARITY)]: { label: 'Cache Similarity', type: 'success' },
  [traceKey(TRACE_REASONING)]: { label: 'Reasoning', type: 'info' },
  [traceKey(TRACE_FALLBACK_ATTEMPTS)]: { label: 'Fallback Attempts', type: 'info' },
  [LEARNING_METHODS]: { label: 'Learning', type: 'info' },
  [LEARNING_ACTIONS]: { label: 'Learning Action', type: 'info' },
  [LEARNING_SCOPES]: { label: 'Learning Scope', type: 'info' },
  [LEARNING_REASONS]: { label: 'Learning Reason', type: 'info' },
  [X_VSR_WARNINGS]: { label: 'Warnings', type: 'warning' },
  ...Object.fromEntries(
    Object.entries(SIGNAL_INFO).map(([signalType, info]) => [matchedSignalKey(signalType), info]),
  ),
  [traceKey(TRACE_CONTEXT_TOKENS)]: { label: 'Context Count', type: 'info' },
  // The Looper's execution
  [LOOPER_MODEL]: { label: 'Final Model', type: 'info' },
  [LOOPER_MODELS]: { label: 'Collaborative Models', type: 'success' },
  [traceKey(`${TRACE_LOOPER}.iterations`)]: { label: 'Iterations', type: 'info' },
  [LOOPER_ALGORITHM]: { label: 'Algorithm', type: 'info' },
  [LOOPER_LATENCY_MS]: { label: 'Looper latency', type: 'info' },
  [traceKey(`${TRACE_LOOPER}.prompt_tokens`)]: { label: 'Looper Prompt Tokens', type: 'info' },
  [traceKey(`${TRACE_LOOPER}.completion_tokens`)]: {
    label: 'Looper Completion Tokens',
    type: 'info',
  },
  [traceKey(`${TRACE_LOOPER}.total_tokens`)]: { label: 'Looper Total Tokens', type: 'info' },
  [ROUTING_LATENCY_MS]: { label: 'Routing Latency', type: 'info' },
  [PLAYGROUND_LATENCY_MS]: { label: 'Latency', type: 'info' },
  [PLAYGROUND_TTFT_MS]: { label: 'TTFT', type: 'info' },
  // The matched decision's retention directive (issue #2009)
  [traceKey(`${TRACE_RETENTION}.drop`)]: { label: 'Retention: Drop', type: 'warning' },
  [traceKey(`${TRACE_RETENTION}.ttl_turns`)]: { label: 'Retention TTL (turns)', type: 'info' },
  [traceKey(`${TRACE_RETENTION}.keep_current_model`)]: {
    label: 'Keep Current Model',
    type: 'info',
  },
  [traceKey(`${TRACE_RETENTION}.prefer_prefix`)]: { label: 'Prefer Prefix', type: 'info' },
  [traceKey(TRACE_TRUNCATED)]: { label: 'Trace Truncated', type: 'warning' },
}

function shouldSummarizeHeaderValue(key: string, values: string[]): boolean {
  return values.length > 1 && (key.startsWith(MATCHED_SIGNAL_KEY_PREFIX) || key === LOOPER_MODELS)
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
  if (key.startsWith(MATCHED_SIGNAL_KEY_PREFIX)) return 0
  if (key === X_VSR_SELECTED_DECISION) return 1
  if (key === ALGORITHM || key === LOOPER_ALGORITHM) return 2
  if (key === X_VSR_SELECTED_MODEL || key === LOOPER_MODEL || key === LOOPER_MODELS) return 3
  if (key === X_VSR_RESPONSE_PATH) return 4
  return 5
}

const HeaderDisplay = ({ headers }: HeaderDisplayProps) => {
  // Filter to only show headers that exist
  const hasSelectedAlgorithm = Boolean(headers[ALGORITHM])
  const displayHeaders = Object.entries(headers)
    .filter(([key]) => key in HEADER_INFO && !(hasSelectedAlgorithm && key === LOOPER_ALGORITHM))
    .sort(
      ([leftKey], [rightKey]) => routingHeaderPriority(leftKey) - routingHeaderPriority(rightKey),
    )

  if (displayHeaders.length === 0) {
    return null
  }

  const primaryKeys = [
    X_VSR_SELECTED_DECISION,
    ALGORITHM,
    LOOPER_ALGORITHM,
    X_VSR_SELECTED_MODEL,
    LOOPER_MODEL,
  ]
  const performanceKeys = [
    ROUTING_LATENCY_MS,
    PLAYGROUND_LATENCY_MS,
    PLAYGROUND_TTFT_MS,
    LOOPER_LATENCY_MS,
  ]
  const primaryHeaders = displayHeaders.filter(([key]) => primaryKeys.includes(key))
  const performanceHeaders = displayHeaders.filter(([key]) => performanceKeys.includes(key))
  const detailHeaders = displayHeaders.filter(
    ([key]) => !primaryKeys.includes(key) && !performanceKeys.includes(key),
  )

  const renderHeader = ([key, value]: [string, string]) => {
    const info = HEADER_INFO[key]
    const displayValue = summarizeHeaderValue(key, value)
    const unit = key.endsWith('-ms') || key.endsWith('_ms') ? ' ms' : ''
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
