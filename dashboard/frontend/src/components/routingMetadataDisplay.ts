import { TRACE_ALGORITHM, TRACE_LOOPER, X_VSR_SELECTED_DECISION } from '../generated/vsrHeaders'

// Display entries name a matched signal family signal:<type>, and a detail of
// x-vsr-trace trace:<member>, beside the public headers they come with.
export const MATCHED_SIGNAL_KEY_PREFIX = 'signal:'

export const matchedSignalKey = (signalType: string): string =>
  `${MATCHED_SIGNAL_KEY_PREFIX}${signalType}`

export const traceKey = (detail: string): string => `trace:${detail}`

const ACRONYMS: Record<string, string> = {
  api: 'API',
  kb: 'Knowledge Base',
  llm: 'LLM',
  mom: 'MoM',
  pii: 'PII',
  rag: 'RAG',
  remom: 'ReMoM',
  tpot: 'TPOT',
  ttft: 'TTFT',
}

const LANGUAGE_NAMES: Record<string, string> = {
  de: 'German',
  en: 'English',
  es: 'Spanish',
  fr: 'French',
  ja: 'Japanese',
  zh: 'Chinese',
}

const IDENTIFIER_LABELS: Record<string, string> = {
  multi_factor: 'Multi-Factor',
  systemone: 'System One',
}

const humanizeToken = (token: string): string => {
  const normalized = token.toLowerCase()
  return ACRONYMS[normalized] ?? `${normalized.charAt(0).toUpperCase()}${normalized.slice(1)}`
}

const humanizeIdentifier = (value: string, dropRouteSuffix: boolean): string => {
  const normalized = value.replace(/^unified_/, '')
  if (IDENTIFIER_LABELS[normalized]) return IDENTIFIER_LABELS[normalized]
  const [base, qualifier] = normalized.split(':', 2)
  const tokens = base.split(/[_-]+/).filter(Boolean)
  if (
    dropRouteSuffix &&
    tokens.length > 1 &&
    tokens[tokens.length - 1]?.toLowerCase() === 'route'
  ) {
    tokens.pop()
  }
  const label = tokens.map(humanizeToken).join(' ')
  return qualifier ? `${label}: ${humanizeToken(qualifier)}` : label
}

export const formatRoutingMetadataValue = (key: string, value: string): string => {
  const trimmed = value.trim()
  if (!trimmed) return trimmed
  const normalizedKey = key.toLowerCase()

  if (normalizedKey === matchedSignalKey('language')) {
    return LANGUAGE_NAMES[trimmed.toLowerCase()] ?? humanizeIdentifier(trimmed, false)
  }
  if (normalizedKey === X_VSR_SELECTED_DECISION) {
    return humanizeIdentifier(trimmed, true)
  }
  if (
    normalizedKey === traceKey(TRACE_ALGORITHM) ||
    normalizedKey === traceKey(`${TRACE_LOOPER}.algorithm`) ||
    normalizedKey.startsWith(MATCHED_SIGNAL_KEY_PREFIX)
  ) {
    return humanizeIdentifier(trimmed, false)
  }

  return trimmed
}
