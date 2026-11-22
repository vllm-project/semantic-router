import {
  TRACE_ALGORITHM,
  TRACE_CACHE_SIMILARITY,
  TRACE_CONFIDENCE,
  TRACE_EFFECTIVE_INPUT_TOKENS,
  TRACE_EFFECTIVE_MAX_OUTPUT_TOKENS,
  TRACE_LEARNING,
  TRACE_LOOPER,
  TRACE_PROMPT_CACHE,
  TRACE_RANKING,
  TRACE_REASONING,
  TRACE_RESPONSE_GUARD,
  TRACE_RETENTION,
  TRACE_ROUTING_LATENCY_MS,
  TRACE_SYSTEM_PROMPT,
  TRACE_TOOLS,
  TRACE_TRUNCATED,
  X_VSR_COST,
  X_VSR_COST_CURRENCY,
  X_VSR_REPLAY_ID,
  X_VSR_SELECTED_DECISION,
  X_VSR_SELECTED_MODEL,
  X_VSR_SELECTED_RECIPE,
  X_VSR_WARNINGS,
} from '../generated/vsrHeaders'
import { signalTypeLabel } from '../pages/configPageSignalCatalog'
import type { InsightsRecord } from '../pages/insightsPageTypes'
import { formatInsightsCost } from '../utils/insightsCost'
import { PLAYGROUND_LATENCY_MS, PLAYGROUND_TTFT_MS } from './playgroundTimingHeaders'
import {
  formatRoutingMetadataValue,
  MATCHED_SIGNAL_KEY_PREFIX,
  traceKey,
} from './routingMetadataDisplay'

export interface InspectedSignal {
  /** `type:name`, as the replay record keys its confidences. */
  key: string
  type: string
  typeLabel: string
  name: string
  matched: boolean
  /** The rule's confidence from 0 to 1, when the replay record carries one. */
  confidence?: number
}

export interface InspectedCandidate {
  model: string
  score?: number
  chosen: boolean
}

export interface InspectedPlugin {
  id: string
  label: string
  detail?: string
  tone: 'rose' | 'success' | 'warning'
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
  /** The matched decision the selected one beat, with the reason. */
  runnerUp?: string
  signals: InspectedSignal[]
  candidates: InspectedCandidate[]
  plugins: InspectedPlugin[]
  looper?: { models: string[]; iterations?: string }
  latencyMs?: number
  cost?: { value: number; currency?: string }
  metrics: InspectedMetric[]
  warnings: string[]
  truncated: boolean
  replayId?: string
}

// Rules evaluated without matching say little past the strongest few.
const UNMATCHED_SIGNAL_LIMIT = 6

const splitList = (value?: string) =>
  (value ?? '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

const number = (value?: string) => {
  const parsed = Number(value)
  return value && Number.isFinite(parsed) ? parsed : undefined
}

export const formatInspectedMs = (value?: number) => {
  if (value === undefined) return undefined
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`
}

export const formatInspectedCost = (cost?: InspectedRoute['cost']) => {
  if (!cost) return undefined
  return cost.currency ? formatInsightsCost(cost.value, cost.currency) : String(cost.value)
}

const count = (value?: number) => (value === undefined ? undefined : value.toLocaleString('en-US'))

function readSignals(
  headers: Record<string, string>,
  record?: InsightsRecord | null,
): InspectedSignal[] {
  const signals = new Map<string, InspectedSignal>()
  const entry = (type: string, name: string) => {
    const key = `${type}:${name}`
    let signal = signals.get(key)
    if (!signal) {
      signal = { key, type, typeLabel: signalTypeLabel(type), name, matched: false }
      signals.set(key, signal)
    }
    return signal
  }
  for (const [key, value] of Object.entries(headers)) {
    if (!key.startsWith(MATCHED_SIGNAL_KEY_PREFIX)) continue
    const type = key.slice(MATCHED_SIGNAL_KEY_PREFIX.length)
    splitList(value).forEach((name) => (entry(type, name).matched = true))
  }
  for (const [type, names] of Object.entries(record?.signals ?? {})) {
    names?.forEach((name) => (entry(type, name).matched = true))
  }
  for (const [key, confidence] of Object.entries(record?.signal_confidences ?? {})) {
    const separator = key.indexOf(':')
    if (separator <= 0 || !Number.isFinite(confidence)) continue
    entry(key.slice(0, separator), key.slice(separator + 1)).confidence = Math.min(
      1,
      Math.max(0, confidence),
    )
  }
  const all = [...signals.values()]
  const unmatched = all
    .filter((signal) => !signal.matched)
    .sort((left, right) => (right.confidence ?? 0) - (left.confidence ?? 0))
    .slice(0, UNMATCHED_SIGNAL_LIMIT)
  return [...all.filter((signal) => signal.matched), ...unmatched]
}

/**
 * The models the selection weighed, with their scores: Router Learning's
 * adaptation scores, else session protection's, else the candidates the
 * objective stages kept. Empty when the record names fewer than two.
 */
function readCandidates(record: InsightsRecord | null | undefined, chosen?: string) {
  const fromScores = (scores?: Record<string, number | undefined>) =>
    Object.entries(scores ?? {})
      .filter(([model]) => model)
      .map(([model, score]) => ({
        model,
        score: typeof score === 'number' && Number.isFinite(score) ? score : undefined,
        chosen: model === chosen,
      }))
      .sort((left, right) => (right.score ?? -Infinity) - (left.score ?? -Infinity))
  const adaptation = record?.learning?.adaptation?.scores
  const policyScores = record?.session_policy?.final_scores ?? record?.session_policy?.base_scores
  const candidates: InspectedCandidate[] = adaptation
    ? fromScores(
        Object.fromEntries(
          Object.entries(adaptation).map(([model, score]) => [
            model,
            score?.score ?? score?.predicted_quality,
          ]),
        ),
      )
    : policyScores
      ? fromScores(policyScores)
      : (record?.route_diagnostics?.selection_trace?.final_survivors ?? []).map((candidate) => ({
          model: candidate.Model,
          chosen: candidate.Model === chosen,
        }))
  return candidates.length > 1 ? candidates : []
}

function readPlugins(
  get: (key: string) => string | undefined,
  headers: Record<string, string>,
  record?: InsightsRecord | null,
): InspectedPlugin[] {
  const trace = (member: string) => get(traceKey(member))
  const plugins: InspectedPlugin[] = []
  const add = (
    id: string,
    label: string,
    detail?: string,
    tone: InspectedPlugin['tone'] = 'rose',
  ) => plugins.push({ id, label, detail, tone })

  if (trace(TRACE_SYSTEM_PROMPT)) add('system_prompt', 'System prompt')
  const promptCache = trace(TRACE_PROMPT_CACHE)
  if (promptCache) add('prompt_cache', 'Prompt cache', promptCache)
  const tools = trace(TRACE_TOOLS)
  if (tools) {
    const kept = trace(`${TRACE_TOOLS}.kept`)
    const candidates = trace(`${TRACE_TOOLS}.candidates`)
    add('tools', 'Tools', kept && candidates ? `${kept} of ${candidates}` : tools)
  }
  const guard = trace(TRACE_RESPONSE_GUARD)
  if (guard)
    add('response_guard', 'Response guard', guard, guard === 'blocked' ? 'warning' : 'rose')
  const similarity =
    trace(TRACE_CACHE_SIMILARITY) ??
    (typeof record?.cache_similarity === 'number' ? record.cache_similarity.toFixed(2) : undefined)
  if (record?.from_cache) add('semantic_cache', 'Semantic cache', 'hit', 'success')
  else if (similarity) add('semantic_cache', 'Semantic cache', `miss · ${similarity}`)
  if (Object.keys(headers).some((key) => key.startsWith(traceKey(`${TRACE_RETENTION}.`)))) {
    add('retention', 'Retention')
  }
  const learning = trace(`${TRACE_LEARNING}.methods`)
  if (learning) add('learning', 'Learning', splitList(learning).join(', '))
  if (trace(TRACE_REASONING) === 'on') add('reasoning', 'Reasoning')
  if (record?.rag_enabled) add('rag', 'RAG', record.rag_backend)
  if (record?.jailbreak_enabled) {
    add(
      'jailbreak',
      'Jailbreak guard',
      record.jailbreak_detected ? 'detected' : undefined,
      record.jailbreak_detected ? 'warning' : 'rose',
    )
  }
  if (record?.pii_enabled) {
    const flagged = record.pii_blocked ? 'blocked' : record.pii_detected ? 'detected' : undefined
    add('pii', 'PII guard', flagged, flagged ? 'warning' : 'rose')
  }
  if (record?.hallucination_enabled) {
    add(
      'hallucination',
      'Hallucination check',
      record.hallucination_detected ? 'flagged' : undefined,
      record.hallucination_detected ? 'warning' : 'rose',
    )
  }
  return plugins
}

/**
 * What the routing trace shows for one answer, from the entries the chat
 * collects (the public x-vsr headers, the debug trace and the playground's own
 * timings) and, once it loads, the answer's Router Replay record, which adds
 * signal confidences, scored candidates and the plugins that ran.
 */
export function readInspectedRoute(
  headers: Record<string, string>,
  record?: InsightsRecord | null,
): InspectedRoute {
  const get = (key: string) => headers[key]?.trim() || undefined
  const decisionId = get(X_VSR_SELECTED_DECISION) ?? record?.decision
  const algorithmKey = traceKey(TRACE_ALGORITHM)
  const algorithm = get(algorithmKey) ?? get(traceKey(`${TRACE_LOOPER}.algorithm`))
  const looperModels = splitList(get(traceKey(`${TRACE_LOOPER}.models`)))
  const model = get(traceKey(`${TRACE_LOOPER}.model`)) ?? get(X_VSR_SELECTED_MODEL)
  const latencyMs = number(get(PLAYGROUND_LATENCY_MS)) ?? record?.duration_ms
  const costValue = number(get(X_VSR_COST)) ?? record?.actual_cost
  const currency = get(X_VSR_COST_CURRENCY) ?? record?.currency
  const tokens =
    typeof record?.prompt_tokens === 'number' && typeof record?.completion_tokens === 'number'
      ? record.prompt_tokens + record.completion_tokens
      : undefined
  const cost = costValue === undefined ? undefined : { value: costValue, currency }

  const metric = (id: string, label: string, value?: string): InspectedMetric[] =>
    value ? [{ id, label, value }] : []
  const metrics = [
    ...metric('latency', 'Latency', formatInspectedMs(latencyMs)),
    ...metric('ttft', 'First token', formatInspectedMs(number(get(PLAYGROUND_TTFT_MS)))),
    ...metric(
      'routing',
      'Routing',
      formatInspectedMs(number(get(traceKey(TRACE_ROUTING_LATENCY_MS)))),
    ),
    ...metric('tokens', 'Tokens', count(tokens)),
    ...metric('cost', 'Cost', formatInspectedCost(cost)),
    ...metric('input', 'Input tokens', count(number(get(traceKey(TRACE_EFFECTIVE_INPUT_TOKENS))))),
    ...metric(
      'output-limit',
      'Output limit',
      count(number(get(traceKey(TRACE_EFFECTIVE_MAX_OUTPUT_TOKENS)))),
    ),
    ...metric(
      'looper-latency',
      'Looper',
      formatInspectedMs(number(get(traceKey(`${TRACE_LOOPER}.latency_ms`)))),
    ),
    ...metric(
      'looper-tokens',
      'Looper tokens',
      count(number(get(traceKey(`${TRACE_LOOPER}.total_tokens`)))),
    ),
  ]

  return {
    decision: decisionId
      ? { id: decisionId, label: formatRoutingMetadataValue(X_VSR_SELECTED_DECISION, decisionId) }
      : undefined,
    recipe: get(X_VSR_SELECTED_RECIPE) ?? record?.recipe,
    model: model ?? record?.selected_model,
    algorithm: algorithm
      ? formatRoutingMetadataValue(algorithmKey, algorithm)
      : record?.selection_method,
    confidence:
      get(traceKey(TRACE_CONFIDENCE)) ??
      (typeof record?.confidence_score === 'number'
        ? record.confidence_score.toFixed(2)
        : undefined),
    runnerUp: get(traceKey(TRACE_RANKING)),
    signals: readSignals(headers, record),
    candidates: readCandidates(record, model ?? record?.selected_model),
    plugins: readPlugins(get, headers, record),
    looper: looperModels.length
      ? { models: looperModels, iterations: get(traceKey(`${TRACE_LOOPER}.iterations`)) }
      : undefined,
    latencyMs,
    cost,
    metrics,
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
