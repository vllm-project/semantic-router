import {
  X_VSR_LOOPER_ALGORITHM,
  X_VSR_MATCHED_LANGUAGE,
  X_VSR_SELECTED_ALGORITHM,
  X_VSR_SELECTED_DECISION,
} from '../generated/vsrHeaders'

// Matched-signal headers are named x-vsr-matched-<signal family>.
export const MATCHED_SIGNAL_HEADER_PREFIX = 'x-vsr-matched-'

export const matchedSignalHeader = (signalFamily: string): string =>
  `${MATCHED_SIGNAL_HEADER_PREFIX}${signalFamily}`

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

  if (normalizedKey === X_VSR_MATCHED_LANGUAGE) {
    return LANGUAGE_NAMES[trimmed.toLowerCase()] ?? humanizeIdentifier(trimmed, false)
  }
  if (normalizedKey === X_VSR_SELECTED_DECISION) {
    return humanizeIdentifier(trimmed, true)
  }
  if (
    normalizedKey === X_VSR_SELECTED_ALGORITHM ||
    normalizedKey === X_VSR_LOOPER_ALGORITHM ||
    normalizedKey.startsWith(MATCHED_SIGNAL_HEADER_PREFIX)
  ) {
    return humanizeIdentifier(trimmed, false)
  }

  return trimmed
}
