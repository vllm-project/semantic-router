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
  latency_aware: 'Latency-Aware',
  multi_factor: 'Multi-Factor',
  rl_driven: 'RL-Driven',
  router_dc: 'Router DC',
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
  if (dropRouteSuffix && tokens[tokens.length - 1]?.toLowerCase() === 'route') {
    tokens.pop()
  }
  const label = tokens.map(humanizeToken).join(' ')
  return qualifier ? `${label}: ${humanizeToken(qualifier)}` : label
}

export const formatRoutingMetadataValue = (key: string, value: string): string => {
  const trimmed = value.trim()
  if (!trimmed) return trimmed
  const normalizedKey = key.toLowerCase()

  if (normalizedKey === 'x-vsr-matched-language') {
    return LANGUAGE_NAMES[trimmed.toLowerCase()] ?? humanizeIdentifier(trimmed, false)
  }
  if (normalizedKey === 'x-vsr-selected-decision') {
    return humanizeIdentifier(trimmed, true)
  }
  if (normalizedKey === 'x-vsr-selected-algorithm' || normalizedKey.startsWith('x-vsr-matched-')) {
    return humanizeIdentifier(trimmed, false)
  }

  return trimmed
}
