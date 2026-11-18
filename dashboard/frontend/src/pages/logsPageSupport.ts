export interface LogEntry {
  line: string
  service?: string
}

export type LogLevel = 'error' | 'warn' | 'info' | 'debug' | 'other'

export const LOG_LEVELS: readonly LogLevel[] = ['error', 'warn', 'info', 'debug', 'other']

export function describeLogsAvailability(supported: boolean | null): string {
  if (supported === null) return 'Detecting'
  return supported ? 'Available' : 'Unavailable'
}

const LEVEL_ALIASES: Record<string, LogLevel> = {
  error: 'error',
  err: 'error',
  fatal: 'error',
  panic: 'error',
  critical: 'error',
  warn: 'warn',
  warning: 'warn',
  info: 'info',
  debug: 'debug',
  trace: 'debug',
}

const LEVEL_PATTERNS = [
  /"(?:level|severity|lvl)"\s*:\s*"([a-z]+)"/i,
  /\blevel=([a-z]+)/i,
  /\[(error|warn|warning|info|debug|trace|critical)\]/i,
  /^\S*\s*(ERROR|WARN|WARNING|INFO|DEBUG|FATAL)\b/,
]

/** Reads the level from JSON, logfmt, bracketed (Envoy) or leading-token formats. */
export function getLogLevel(line: string): LogLevel {
  for (const pattern of LEVEL_PATTERNS) {
    const match = line.match(pattern)
    const level = match && LEVEL_ALIASES[match[1].toLowerCase()]
    if (level) return level
  }
  return 'other'
}

/** An empty level list keeps every level. */
export function filterLogs(
  logs: LogEntry[],
  query: string,
  levels: readonly LogLevel[] = [],
): LogEntry[] {
  const normalizedQuery = query.trim().toLocaleLowerCase()
  return logs.filter((entry) => {
    if (levels.length && !levels.includes(getLogLevel(entry.line))) return false
    if (!normalizedQuery) return true
    return `${entry.service ?? ''} ${entry.line}`.toLocaleLowerCase().includes(normalizedQuery)
  })
}

export function matchRanges(text: string, query: string): { start: number; end: number }[] {
  const needle = query.trim().toLocaleLowerCase()
  if (!needle) return []
  const haystack = text.toLocaleLowerCase()
  const ranges: { start: number; end: number }[] = []
  for (let found = haystack.indexOf(needle); found !== -1; ) {
    ranges.push({ start: found, end: found + needle.length })
    found = haystack.indexOf(needle, found + needle.length)
  }
  return ranges
}

/** The combined stream prefixes each line with its component, e.g. `[router] `. */
export function splitServicePrefix(line: string): { service: string | null; rest: string } {
  const match = line.match(/^\[(router|envoy|dashboard)\] /)
  return match
    ? { service: match[1], rest: line.slice(match[0].length) }
    : { service: null, rest: line }
}

/** Structured fields of a JSON log line, or null for plain text. */
export function parseStructuredLine(line: string): Record<string, unknown> | null {
  const { rest } = splitServicePrefix(line)
  const text = rest.trim()
  if (!text.startsWith('{') || !text.endsWith('}')) return null
  try {
    const value: unknown = JSON.parse(text)
    return value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : null
  } catch {
    return null
  }
}

export function logsDownloadName(component: string, now: Date): string {
  return `vllm-sr-${component}-${now.toISOString().slice(0, 19).replace(/[-:T]/g, '')}.log`
}
