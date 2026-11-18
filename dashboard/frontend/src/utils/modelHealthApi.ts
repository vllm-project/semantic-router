export type ModelHealthStatus = 'healthy' | 'degraded' | 'down' | 'unchecked'

export interface ModelHealthEndpoint {
  name: string
  status: ModelHealthStatus
  reason?: string
  latencyMs?: number
  httpStatus?: number
}

export interface ModelHealth {
  model: string
  status: ModelHealthStatus
  reason?: string
  checkedAt?: string
  /** The latest answer time of the model's primary answering backend. */
  latencyMs?: number
  /** The 95th percentile of `recentMs`, the answer times of recent checks. */
  p95Ms?: number
  recentMs: number[]
  endpoints: ModelHealthEndpoint[]
}

export interface ModelHealthSnapshot {
  checkedAt: string
  intervalSeconds: number
  models: ModelHealth[]
}

export class ModelHealthApiError extends Error {
  readonly status: number
  readonly code: string

  constructor(message: string, status: number, code = 'health_failed') {
    super(message)
    this.name = 'ModelHealthApiError'
    this.status = status
    this.code = code
  }
}

const STATUSES = new Set<ModelHealthStatus>(['healthy', 'degraded', 'down', 'unchecked'])

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

const text = (value: unknown) =>
  typeof value === 'string' && value.trim() ? value.trim() : undefined

const milliseconds = (value: unknown) =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined

const status = (value: unknown): ModelHealthStatus | undefined =>
  STATUSES.has(value as ModelHealthStatus) ? (value as ModelHealthStatus) : undefined

function parseEndpoint(value: unknown): ModelHealthEndpoint | null {
  if (!isRecord(value)) return null
  const name = text(value.name)
  const state = status(value.status)
  if (!name || !state) return null
  return {
    name,
    status: state,
    reason: text(value.reason),
    latencyMs: milliseconds(value.latencyMs),
    httpStatus: milliseconds(value.httpStatus),
  }
}

function parseModelHealth(value: unknown): ModelHealth | null {
  if (!isRecord(value)) return null
  const model = text(value.model)
  const state = status(value.status)
  if (!model || !state) return null
  return {
    model,
    status: state,
    reason: text(value.reason),
    checkedAt: text(value.checkedAt),
    latencyMs: milliseconds(value.latencyMs),
    p95Ms: milliseconds(value.p95Ms),
    recentMs: Array.isArray(value.recentMs)
      ? value.recentMs.filter((sample): sample is number => milliseconds(sample) !== undefined)
      : [],
    endpoints: Array.isArray(value.endpoints)
      ? value.endpoints
          .map(parseEndpoint)
          .filter((endpoint): endpoint is ModelHealthEndpoint => endpoint !== null)
      : [],
  }
}

export function parseModelHealthSnapshot(value: unknown): ModelHealthSnapshot {
  if (!isRecord(value) || !Array.isArray(value.models)) {
    throw new ModelHealthApiError(
      'Model health returned an invalid contract.',
      502,
      'invalid_health_contract',
    )
  }
  const interval = milliseconds(value.intervalSeconds)
  return {
    checkedAt: text(value.checkedAt) ?? new Date().toISOString(),
    intervalSeconds: interval && interval >= 5 ? interval : 30,
    models: value.models
      .map(parseModelHealth)
      .filter((entry): entry is ModelHealth => entry !== null),
  }
}

/**
 * The health of the configured backend models, or of `models` only. The
 * Dashboard shares recent checks between viewers; `refresh` asks for a new
 * check unless one ran moments ago.
 */
export async function fetchModelHealth({
  models = [],
  refresh = false,
  signal,
}: { models?: string[]; refresh?: boolean; signal?: AbortSignal } = {}) {
  const query = new URLSearchParams()
  for (const model of models) query.append('model', model)
  if (refresh) query.set('refresh', '1')
  const search = query.toString()
  const response = await fetch(`/api/models/health${search ? `?${search}` : ''}`, { signal })
  if (!response.ok) {
    let message = `Model health is unavailable (HTTP ${response.status}).`
    let code = 'health_failed'
    try {
      const body: unknown = await response.json()
      if (isRecord(body)) {
        message = text(body.message) ?? message
        code = text(body.error) ?? code
      }
    } catch {
      // A proxy or an older Dashboard answers without the JSON error body.
    }
    throw new ModelHealthApiError(message, response.status, code)
  }
  return parseModelHealthSnapshot(await response.json())
}
