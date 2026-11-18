export interface LogEntry {
  line: string
  service?: string
  /** Set for lines read from Kubernetes pods. */
  pod?: string
  timestamp?: string
}

/** GET /api/logs. */
export interface LogsResponse {
  /** local-spool, kubernetes or unsupported. */
  deployment_type: string
  service: string
  supported: boolean
  /** The components this source reads. */
  components?: string[]
  namespace?: string
  pods?: string[]
  logs: LogEntry[]
  count: number
  error?: string
  message?: string
}

export type LogLevel = 'error' | 'warn' | 'info' | 'debug' | 'other'

export const LOG_LEVELS: readonly LogLevel[] = ['error', 'warn', 'info', 'debug', 'other']

export type LogSource = 'spool' | 'kubernetes' | 'none'

export function logSourceOf(deploymentType: string | undefined): LogSource | null {
  if (deploymentType === 'local-spool') return 'spool'
  if (deploymentType === 'kubernetes') return 'kubernetes'
  if (deploymentType === 'unsupported') return 'none'
  return null
}

export const LOG_SOURCE_LABEL: Record<LogSource, string> = {
  spool: 'Log spool',
  kubernetes: 'Kubernetes',
  none: 'None',
}

const COMPONENT_LABEL: Record<string, string> = {
  router: 'Router',
  envoy: 'Envoy',
  dashboard: 'Dashboard',
}

export function componentLabel(component: string): string {
  return COMPONENT_LABEL[component] ?? component
}

/** All, then each component the source reads. */
export function componentOptions(
  components: readonly string[],
): { value: string; label: string }[] {
  return [
    { value: 'all', label: 'All' },
    ...components.map((value) => ({ value, label: componentLabel(value) })),
  ]
}

/**
 * A Deployment pod's name without its ReplicaSet hash, router-x2x4z for
 * router-7d9f8c6b5-x2x4z, so replicas stay apart in a narrow column.
 */
export function shortPodName(pod: string): string {
  const match = pod.match(/^(.+)-[a-z0-9]{5,10}-([a-z0-9]{5})$/)
  return match ? `${match[1]}-${match[2]}` : pod
}

/** What a line came from: its pod, else its component. */
export function lineSource(entry: LogEntry): string | undefined {
  return entry.pod ?? entry.service
}

export function entryKey(entry: LogEntry): string {
  return `${entry.timestamp ?? ''}|${lineSource(entry) ?? ''}|${entry.line}`
}

/** How many lines of `next` follow the last line of `previous`; all of them when it is gone. */
export function countNewLines(previous: readonly LogEntry[], next: readonly LogEntry[]): number {
  if (!previous.length || !next.length) return 0
  const last = entryKey(previous[previous.length - 1])
  for (let index = next.length - 1; index >= 0; index -= 1) {
    if (entryKey(next[index]) === last) return next.length - 1 - index
  }
  return next.length
}

const pad = (value: number, width = 2) => String(value).padStart(width, '0')

/** The local wall-clock time of a timestamp, as 14:05:09. */
export function formatLogTime(timestamp: string | undefined): string | undefined {
  if (!timestamp) return undefined
  const time = new Date(timestamp)
  if (Number.isNaN(time.getTime())) return undefined
  return `${pad(time.getHours())}:${pad(time.getMinutes())}:${pad(time.getSeconds())}`
}

/** A line as copied or downloaded: its timestamp and, in a mixed view, its source. */
export function logLineText(entry: LogEntry, withSource: boolean): string {
  const source = withSource ? lineSource(entry) : undefined
  return [entry.timestamp, source ? `[${source}]` : undefined, entry.line].filter(Boolean).join(' ')
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

/** An empty level list keeps every level. Search matches the line, its component and its pod. */
export function filterLogs(
  logs: LogEntry[],
  query: string,
  levels: readonly LogLevel[] = [],
): LogEntry[] {
  const normalizedQuery = query.trim().toLocaleLowerCase()
  return logs.filter((entry) => {
    if (levels.length && !levels.includes(getLogLevel(entry.line))) return false
    if (!normalizedQuery) return true
    return `${entry.service ?? ''} ${entry.pod ?? ''} ${entry.line}`
      .toLocaleLowerCase()
      .includes(normalizedQuery)
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

/** Structured fields of a JSON log line, or null for plain text. */
export function parseStructuredLine(line: string): Record<string, unknown> | null {
  const text = line.trim()
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

/** Why a log request failed, from its HTTP status and any JSON error body. */
export function logsRequestFailure(status: number, body: unknown): string {
  if (status === 401 || status === 403) return 'Your account cannot read runtime logs.'
  const message =
    body && typeof body === 'object' && 'message' in body && typeof body.message === 'string'
      ? body.message.trim()
      : ''
  return message || `The Dashboard answered HTTP ${status}.`
}

export interface LogsNotice {
  kind: 'unavailable' | 'error' | 'empty'
  title: string
  description: string
}

/**
 * What the log view says when it has no line to show, or null when it has
 * lines: no source, a source the Dashboard may not read, a failed read, or a
 * source with nothing recent.
 */
export function describeEmptyLogs({
  response,
  requestError,
  following,
}: {
  response: LogsResponse | null
  /** The last request failed before an answer arrived. */
  requestError: string | null
  following: boolean
}): LogsNotice | null {
  const retry = following ? ' Follow retries every 5 seconds.' : ''
  if (requestError) {
    return { kind: 'error', title: 'Logs could not be read', description: requestError + retry }
  }
  if (!response || response.logs.length > 0) return null
  const source = logSourceOf(response.deployment_type)
  if (!response.supported) {
    return {
      kind: 'unavailable',
      title: source === 'kubernetes' ? 'The Dashboard cannot read pod logs' : 'No log source',
      description: response.message || 'This deployment offers no runtime logs.',
    }
  }
  if (response.error) {
    return { kind: 'error', title: 'Logs could not be read', description: response.error + retry }
  }
  return {
    kind: 'empty',
    title: 'No recent lines',
    description:
      response.message ||
      (following
        ? 'New lines appear here as they are written.'
        : 'Turn on Follow to see new lines as they are written.'),
  }
}
