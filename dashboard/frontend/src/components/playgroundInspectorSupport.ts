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
  TRACE_ROUTING_LATENCY_MS,
  TRACE_TRUNCATED,
  X_VSR_COST,
  X_VSR_COST_CURRENCY,
  X_VSR_REPLAY_ID,
  X_VSR_RESPONSE_PATH,
  X_VSR_SELECTED_DECISION,
  X_VSR_SELECTED_MODEL,
  X_VSR_SELECTED_RECIPE,
  X_VSR_WARNINGS,
} from '../generated/vsrHeaders'
import { signalTypeLabel } from '../pages/configPageSignalCatalog'
import { PLAYGROUND_LATENCY_MS, PLAYGROUND_TTFT_MS } from './playgroundTimingHeaders'
import {
  formatRoutingMetadataValue,
  MATCHED_SIGNAL_KEY_PREFIX,
  traceKey,
} from './routingMetadataDisplay'

export interface InspectedSignal {
  type: string
  label: string
  names: string[]
}

export interface InspectedMetric {
  id: string
  label: string
  value: string
}

export interface InspectedRoute {
  decision?: { id: string; label: string }
  recipe?: string
  model?: string
  algorithm?: string
  confidence?: string
  responsePath?: string
  signals: InspectedSignal[]
  looper?: { models: string[]; iterations?: string }
  metrics: InspectedMetric[]
  details: InspectedMetric[]
  warnings: string[]
  truncated: boolean
  replayId?: string
}

const splitList = (value?: string) =>
  (value ?? '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

const ms = (value?: string) => {
  const number = Number(value)
  if (!value || !Number.isFinite(number)) return undefined
  return number >= 1000 ? `${(number / 1000).toFixed(2)} s` : `${Math.round(number)} ms`
}

const count = (value?: string) => {
  const number = Number(value)
  return value && Number.isFinite(number) ? number.toLocaleString('en-US') : undefined
}

/**
 * What the routing trace shows for one answer, from the entries the chat
 * collects: the public x-vsr headers, the debug trace and the playground's own
 * timings.
 */
export function readInspectedRoute(headers: Record<string, string>): InspectedRoute {
  const get = (key: string) => headers[key]?.trim() || undefined
  const decisionId = get(X_VSR_SELECTED_DECISION)
  const algorithmKey = traceKey(TRACE_ALGORITHM)
  const looperAlgorithmKey = traceKey(`${TRACE_LOOPER}.algorithm`)
  const algorithm = get(algorithmKey) ?? get(looperAlgorithmKey)
  const looperModels = splitList(get(traceKey(`${TRACE_LOOPER}.models`)))

  const signals = Object.entries(headers)
    .filter(([key, value]) => key.startsWith(MATCHED_SIGNAL_KEY_PREFIX) && value.trim())
    .map(([key, value]) => {
      const type = key.slice(MATCHED_SIGNAL_KEY_PREFIX.length)
      return { type, label: signalTypeLabel(type), names: splitList(value) }
    })

  const cost = get(X_VSR_COST)
  const metric = (id: string, label: string, value?: string): InspectedMetric[] =>
    value ? [{ id, label, value }] : []
  const metrics = [
    ...metric('latency', 'Latency', ms(get(PLAYGROUND_LATENCY_MS))),
    ...metric('ttft', 'First token', ms(get(PLAYGROUND_TTFT_MS))),
    ...metric('routing', 'Routing', ms(get(traceKey(TRACE_ROUTING_LATENCY_MS)))),
    ...metric('input', 'Input tokens', count(get(traceKey(TRACE_EFFECTIVE_INPUT_TOKENS)))),
    ...metric(
      'output-limit',
      'Output limit',
      count(get(traceKey(TRACE_EFFECTIVE_MAX_OUTPUT_TOKENS))),
    ),
    ...metric(
      'cost',
      'Cost',
      cost ? `${cost}${get(X_VSR_COST_CURRENCY) ? ` ${get(X_VSR_COST_CURRENCY)}` : ''}` : undefined,
    ),
    ...metric('looper-latency', 'Looper', ms(get(traceKey(`${TRACE_LOOPER}.latency_ms`)))),
    ...metric(
      'looper-tokens',
      'Looper tokens',
      count(get(traceKey(`${TRACE_LOOPER}.total_tokens`))),
    ),
  ]

  const details = [
    ...metric('context', 'Context tokens', count(get(traceKey(TRACE_CONTEXT_TOKENS)))),
    ...metric('cache', 'Cache similarity', get(traceKey(TRACE_CACHE_SIMILARITY))),
    ...metric('reasoning', 'Reasoning', get(traceKey(TRACE_REASONING))),
    ...metric('fallbacks', 'Fallback attempts', get(traceKey(TRACE_FALLBACK_ATTEMPTS))),
    ...metric(
      'modality',
      'Output',
      splitList(get(traceKey(TRACE_MODALITY))).join(', ') || undefined,
    ),
  ]

  return {
    decision: decisionId
      ? { id: decisionId, label: formatRoutingMetadataValue(X_VSR_SELECTED_DECISION, decisionId) }
      : undefined,
    recipe: get(X_VSR_SELECTED_RECIPE),
    model: get(traceKey(`${TRACE_LOOPER}.model`)) ?? get(X_VSR_SELECTED_MODEL),
    algorithm: algorithm ? formatRoutingMetadataValue(algorithmKey, algorithm) : undefined,
    confidence: get(traceKey(TRACE_CONFIDENCE)),
    responsePath: get(X_VSR_RESPONSE_PATH),
    signals,
    looper: looperModels.length
      ? { models: looperModels, iterations: get(traceKey(`${TRACE_LOOPER}.iterations`)) }
      : undefined,
    metrics,
    details,
    warnings: splitList(get(X_VSR_WARNINGS)),
    truncated: get(traceKey(TRACE_TRUNCATED)) === 'true',
    replayId: get(X_VSR_REPLAY_ID),
  }
}

/** Every collected entry for the raw view: public headers first, then trace details. */
export function sortedHeaderEntries(headers: Record<string, string>): Array<[string, string]> {
  const rank = (key: string) =>
    key.startsWith('x-vsr-') ? 0 : key.startsWith(MATCHED_SIGNAL_KEY_PREFIX) ? 1 : 2
  return Object.entries(headers).sort(
    ([left], [right]) => rank(left) - rank(right) || left.localeCompare(right),
  )
}
