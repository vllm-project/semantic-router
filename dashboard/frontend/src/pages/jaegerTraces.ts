const JAEGER_API = '/embedded/jaeger/api'

/** Jaeger reads `lookback` as a Go duration, which has no day unit. */
export const TRACE_LOOKBACKS = [
  { value: '15m', label: 'Last 15 minutes' },
  { value: '1h', label: 'Last hour' },
  { value: '6h', label: 'Last 6 hours' },
  { value: '24h', label: 'Last 24 hours' },
  { value: '168h', label: 'Last 7 days' },
] as const
export type TraceLookback = (typeof TRACE_LOOKBACKS)[number]['value']
export const DEFAULT_TRACE_LOOKBACK: TraceLookback = '1h'

export const TRACE_LIMITS = [20, 50, 100] as const
export const DEFAULT_TRACE_LIMIT = 20

export interface TraceSearch {
  service: string
  /** Empty for every operation. */
  operation: string
  lookback: TraceLookback
  limit: number
  /** A Go duration such as `250ms`, or empty for no minimum. */
  minDuration: string
}

export function readTraceLookback(value: string | null): TraceLookback {
  return TRACE_LOOKBACKS.find((option) => option.value === value)?.value ?? DEFAULT_TRACE_LOOKBACK
}

export function readTraceLimit(value: string | null): number {
  const limit = Number(value)
  return TRACE_LIMITS.find((option) => option === limit) ?? DEFAULT_TRACE_LIMIT
}

const GO_DURATION = /^(?:\d+(?:\.\d+)?(?:ns|us|µs|μs|ms|s|m|h))+$/

/** The duration Jaeger accepts for `minDuration`: `''` for none, null when it cannot parse it. */
export function normalizeTraceDuration(input: string): string | null {
  const value = input.replace(/\s+/g, '')
  if (!value) return ''
  return GO_DURATION.test(value) ? value : null
}

export const jaegerServicesUrl = () => `${JAEGER_API}/services`

export const jaegerOperationsUrl = (service: string) =>
  `${JAEGER_API}/services/${encodeURIComponent(service)}/operations`

export function jaegerTracesUrl(search: TraceSearch): string {
  const params = new URLSearchParams({ service: search.service })
  if (search.operation) params.set('operation', search.operation)
  params.set('lookback', search.lookback)
  params.set('limit', String(search.limit))
  if (search.minDuration) params.set('minDuration', search.minDuration)
  return `${JAEGER_API}/traces?${params}`
}

export const jaegerTraceUrl = (traceID: string) =>
  `${JAEGER_API}/traces/${encodeURIComponent(traceID)}`

export interface KeyValue {
  key: string
  value: string
}

export interface SpanLog {
  /** Microseconds since the epoch. */
  timestamp: number
  fields: KeyValue[]
}

export interface SpanReference {
  refType: string
  spanID: string
}

export interface TraceSpan {
  spanID: string
  references: SpanReference[]
  operationName: string
  serviceName: string
  /** Microseconds since the epoch. */
  startTime: number
  /** Microseconds. */
  duration: number
  tags: KeyValue[]
  logs: SpanLog[]
  error: boolean
  kind: string | null
}

export interface Trace {
  traceID: string
  spans: TraceSpan[]
}

/** Service or operation names from Jaeger, which answers `data: null` when there are none. */
export function readJaegerNames(body: unknown): string[] {
  const data = (body as { data?: unknown } | null)?.data
  if (data === null || data === undefined) return []
  if (!Array.isArray(data)) throw new Error('Jaeger did not return a list.')
  const names = data.flatMap((entry) => {
    const name =
      typeof entry === 'string'
        ? entry
        : entry && typeof entry === 'object'
          ? (entry as { name?: unknown }).name
          : null
    return typeof name === 'string' && name ? [name] : []
  })
  return [...new Set(names)].sort((left, right) => left.localeCompare(right))
}

function stringify(value: unknown): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (value === null || value === undefined) return ''
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function readKeyValues(raw: unknown): KeyValue[] {
  if (!Array.isArray(raw)) return []
  return raw.flatMap((entry) => {
    const { key, value } = (entry ?? {}) as { key?: unknown; value?: unknown }
    return typeof key === 'string' ? [{ key, value: stringify(value) }] : []
  })
}

function readLogs(raw: unknown): SpanLog[] {
  if (!Array.isArray(raw)) return []
  return raw.flatMap((entry) => {
    const { timestamp, fields } = (entry ?? {}) as { timestamp?: unknown; fields?: unknown }
    return typeof timestamp === 'number' ? [{ timestamp, fields: readKeyValues(fields) }] : []
  })
}

/** OpenTelemetry exporters mark a failed span with `error=true` and `otel.status_code=ERROR`. */
function isError(tags: readonly KeyValue[]): boolean {
  return tags.some(
    ({ key, value }) =>
      (key === 'error' && value === 'true') ||
      (key === 'otel.status_code' && value.toUpperCase() === 'ERROR'),
  )
}

function readSpan(raw: unknown, services: ReadonlyMap<string, string>): TraceSpan | null {
  const span = (raw ?? {}) as {
    spanID?: unknown
    operationName?: unknown
    references?: unknown
    startTime?: unknown
    duration?: unknown
    tags?: unknown
    logs?: unknown
    processID?: unknown
  }
  if (
    typeof span.spanID !== 'string' ||
    typeof span.startTime !== 'number' ||
    typeof span.duration !== 'number'
  ) {
    return null
  }
  const tags = readKeyValues(span.tags)
  const references = Array.isArray(span.references)
    ? span.references.flatMap((entry) => {
        const { refType, spanID } = (entry ?? {}) as { refType?: unknown; spanID?: unknown }
        return typeof spanID === 'string' ? [{ refType: String(refType ?? ''), spanID }] : []
      })
    : []
  return {
    spanID: span.spanID,
    references,
    operationName:
      typeof span.operationName === 'string' && span.operationName
        ? span.operationName
        : '(unnamed)',
    serviceName: (typeof span.processID === 'string' && services.get(span.processID)) || 'unknown',
    startTime: span.startTime,
    duration: Math.max(0, span.duration),
    tags,
    logs: readLogs(span.logs),
    error: isError(tags),
    kind: tags.find((tag) => tag.key === 'span.kind')?.value ?? null,
  }
}

function readTrace(raw: unknown): Trace | null {
  const { traceID, spans, processes } = (raw ?? {}) as {
    traceID?: unknown
    spans?: unknown
    processes?: unknown
  }
  if (typeof traceID !== 'string' || !Array.isArray(spans)) return null
  const services = new Map<string, string>()
  if (processes && typeof processes === 'object') {
    for (const [id, process] of Object.entries(processes)) {
      const name = (process as { serviceName?: unknown } | null)?.serviceName
      if (typeof name === 'string' && name) services.set(id, name)
    }
  }
  const parsed = spans.flatMap((span) => readSpan(span, services) ?? [])
  return parsed.length ? { traceID, spans: parsed } : null
}

/**
 * Traces from a Jaeger search or lookup. Malformed spans and empty traces are
 * left out, and a trace a storage backend returns twice is kept once, with
 * the most spans.
 */
export function readJaegerTraces(body: unknown): Trace[] {
  const data = (body as { data?: unknown } | null)?.data
  if (data === null || data === undefined) return []
  if (!Array.isArray(data)) throw new Error('Jaeger did not return traces.')
  const traces = new Map<string, Trace>()
  for (const raw of data) {
    const trace = readTrace(raw)
    const known = trace && traces.get(trace.traceID)
    if (trace && (!known || known.spans.length < trace.spans.length)) {
      traces.set(trace.traceID, trace)
    }
  }
  return [...traces.values()]
}

/**
 * The span a span hangs under in this trace: its CHILD_OF reference, else its
 * FOLLOWS_FROM one. A reference to a span the trace lacks makes it a root.
 */
export function parentSpanID(span: TraceSpan, spanIDs: ReadonlySet<string>): string | null {
  const inTrace = (refType: string) =>
    span.references.find(
      (reference) =>
        reference.refType === refType &&
        reference.spanID !== span.spanID &&
        spanIDs.has(reference.spanID),
    )
  return (inTrace('CHILD_OF') ?? inTrace('FOLLOWS_FROM'))?.spanID ?? null
}

export interface TraceSummary {
  traceID: string
  /** The earliest span without a parent in the trace. */
  root: TraceSpan | null
  /** Each service with its span count, the root's first, then the busiest. */
  services: { name: string; spans: number }[]
  spanCount: number
  errorCount: number
  /** Microseconds since the epoch. */
  start: number
  /** Microseconds from the first span's start to the last span's end. */
  duration: number
}

export function summarizeTrace(trace: Trace): TraceSummary {
  const spanIDs = new Set(trace.spans.map((span) => span.spanID))
  let start = Infinity
  let end = -Infinity
  let root: TraceSpan | null = null
  const counts = new Map<string, number>()
  for (const span of trace.spans) {
    start = Math.min(start, span.startTime)
    end = Math.max(end, span.startTime + span.duration)
    counts.set(span.serviceName, (counts.get(span.serviceName) ?? 0) + 1)
    if (parentSpanID(span, spanIDs) === null && (!root || span.startTime < root.startTime)) {
      root = span
    }
  }
  const rootService = root?.serviceName
  const services = [...counts]
    .map(([name, spans]) => ({ name, spans }))
    .sort(
      (left, right) =>
        Number(right.name === rootService) - Number(left.name === rootService) ||
        right.spans - left.spans ||
        left.name.localeCompare(right.name),
    )
  return {
    traceID: trace.traceID,
    root,
    services,
    spanCount: trace.spans.length,
    errorCount: trace.spans.filter((span) => span.error).length,
    start,
    duration: Math.max(0, end - start),
  }
}

/** A span or trace duration in microseconds, as people read it. */
export function formatMicros(micros: number): string {
  if (!Number.isFinite(micros) || micros < 0) return '—'
  if (micros < 1_000) return `${Math.round(micros)} µs`
  if (micros < 1_000_000) {
    const ms = micros / 1_000
    return `${ms.toFixed(ms < 10 ? 2 : ms < 100 ? 1 : 0)} ms`
  }
  const seconds = micros / 1_000_000
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

/** How long before `now` a trace started, both in milliseconds. */
export function formatAgo(time: number, now: number): string {
  const seconds = Math.max(0, Math.round((now - time) / 1_000))
  if (seconds < 45) return 'Just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 48) return `${hours} h ago`
  return `${Math.round(hours / 24)} d ago`
}

const SERVICE_COLORS = [
  'var(--chart-1)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-2)',
  'var(--chart-6)',
  'var(--chart-7)',
  'var(--chart-8)',
]

/**
 * A service's color by its place in `services`, so one service keeps one
 * color across the trace list and the waterfall. Rose is left out: it marks errors.
 */
export function serviceColor(services: readonly string[], service: string): string {
  const index = services.indexOf(service)
  return index < 0
    ? SERVICE_COLORS[SERVICE_COLORS.length - 1]
    : SERVICE_COLORS[index % SERVICE_COLORS.length]
}
