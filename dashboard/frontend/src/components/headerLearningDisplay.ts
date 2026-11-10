import { TRACE_LEARNING } from '../generated/vsrHeaders'
import { traceKey } from './routingMetadataDisplay'

// The display entries of the trace's learning member: the methods, and each
// method's action, scope and reason as method=value pairs.
export const LEARNING_METHODS = traceKey(`${TRACE_LEARNING}.methods`)
export const LEARNING_ACTIONS = traceKey(`${TRACE_LEARNING}.actions`)
export const LEARNING_SCOPES = traceKey(`${TRACE_LEARNING}.scopes`)
export const LEARNING_REASONS = traceKey(`${TRACE_LEARNING}.reasons`)

const LEARNING_METHOD_LABELS: Record<string, string> = {
  adaptation: 'adaptation',
  protection: 'protection',
}

const LEARNING_ACTION_LABELS: Record<string, string> = {
  keep_base: 'kept base model',
  propose_switch: 'proposed switch',
  observe: 'observe only',
  bypass: 'bypassed',
  establish_baseline: 'baseline established',
  allow_sampling: 'sampling allowed',
  suppress_sampling: 'sampling suppressed',
  hold_current: 'held current model',
  allow_switch: 'switch allowed',
  rescue_switch: 'rescue switch',
  release_switch: 'explicit release',
}

const LEARNING_REASON_LABELS: Record<string, string> = {
  cold_start: 'cold start',
  posterior_win: 'posterior winner',
  sampled_win: 'sampled winner',
  base_best: 'base model best',
  candidate_ineligible: 'candidate ineligible',
  observe_only: 'observe only',
  decision_bypass: 'decision bypass',
  new_conversation: 'conversation baseline',
  new_session: 'session baseline',
  idle_reset: 'idle reset',
  missing_identity: 'missing identity',
  tool_or_protocol_state: 'tool or protocol state',
  cache_cost_high: 'cache cost high',
  handoff_cost_high: 'handoff cost high',
  switch_margin_not_met: 'switch margin not met',
  switch_allowed: 'switch allowed',
  rescue_evidence: 'rescue evidence',
  explicit_release: 'explicit release',
}

const HUMANIZED_VALUE_BY_HEADER: Record<string, Record<string, string>> = {
  [LEARNING_ACTIONS]: LEARNING_ACTION_LABELS,
  [LEARNING_REASONS]: LEARNING_REASON_LABELS,
}

const LEARNING_HEADERS: string[] = [
  LEARNING_METHODS,
  LEARNING_ACTIONS,
  LEARNING_SCOPES,
  LEARNING_REASONS,
]

function splitHeaderList(rawValue: string): string[] {
  return rawValue
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)
}

function humanizeToken(value: string): string {
  return value.replace(/[_-]+/g, ' ').trim()
}

function methodLabel(method: string): string {
  return LEARNING_METHOD_LABELS[method] ?? humanizeToken(method)
}

function humanizeHeaderValue(headerKey: string, value: string): string {
  const normalized = value.trim()
  return HUMANIZED_VALUE_BY_HEADER[headerKey]?.[normalized] ?? humanizeToken(normalized)
}

function formatLearningPairs(headerKey: string, rawValue: string): string {
  return splitHeaderList(rawValue)
    .map((entry) => {
      const [method, value] = entry.split('=', 2).map((part) => part.trim())
      if (!method || !value) {
        return humanizeHeaderValue(headerKey, entry)
      }
      return `${methodLabel(method)}: ${humanizeHeaderValue(headerKey, value)}`
    })
    .join(' · ')
}

function formatLearningMethods(rawValue: string): string {
  return splitHeaderList(rawValue).map(methodLabel).join(' + ')
}

export function isLearningHeader(key: string): boolean {
  return LEARNING_HEADERS.includes(key)
}

export function formatLearningHeaderValue(key: string, rawValue: string): string {
  if (key === LEARNING_METHODS) {
    return formatLearningMethods(rawValue)
  }
  if (isLearningHeader(key)) {
    return formatLearningPairs(key, rawValue)
  }
  return rawValue
}
