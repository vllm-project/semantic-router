export type AuditLog = {
  id: number
  userId?: string
  action: string
  resource: string
  method: string
  path: string
  ip: string
  userAgent: string
  statusCode: number
  createdAt: number
  extraJson?: string
}

export type AuditSortOrder = 'asc' | 'desc'

export type AuditLogQuery = {
  query: string
  user: string
  action: string
  resource: string
  status: string
  from: string
  to: string
  sort: string
  order: AuditSortOrder
  page: number
  limit: number
}

export type AuditLogFacet = { value: string; count: number }

/** The most frequent actors and actions, which the server adds when asked with facets=1. */
export type AuditLogFacets = { users: AuditLogFacet[]; actions: AuditLogFacet[] }

export type AuditLogPage = {
  logs: AuditLog[]
  total: number
  page: number
  limit: number
  facets?: AuditLogFacets
}

type AuditLogPagePayload = Partial<AuditLogPage>

export const buildAuditLogQuery = (filters: AuditLogQuery) => {
  const query = new URLSearchParams()
  const optionalValues = [
    ['q', filters.query],
    ['user', filters.user],
    ['action', filters.action],
    ['resource', filters.resource],
    ['from', filters.from],
    ['to', filters.to],
  ] as const

  optionalValues.forEach(([key, value]) => {
    const normalized = value.trim()
    if (normalized) {
      query.set(key, normalized)
    }
  })
  if (filters.status && filters.status !== 'all') {
    query.set('status', filters.status)
  }
  query.set('sort', filters.sort)
  query.set('order', filters.order)
  query.set('page', String(filters.page))
  query.set('limit', String(filters.limit))
  return query.toString()
}

export const normalizeAuditLogPage = (
  payload: AuditLogPagePayload | AuditLog[],
  fallbackPage: number,
  fallbackLimit: number,
): AuditLogPage => {
  if (Array.isArray(payload)) {
    return {
      logs: payload,
      total: payload.length,
      page: fallbackPage,
      limit: fallbackLimit,
    }
  }

  const logs = Array.isArray(payload.logs) ? payload.logs : []
  const facetList = (value: unknown): AuditLogFacet[] =>
    Array.isArray(value)
      ? value.filter(
          (item): item is AuditLogFacet =>
            typeof item?.value === 'string' && typeof item?.count === 'number',
        )
      : []
  return {
    logs,
    total: typeof payload.total === 'number' ? payload.total : logs.length,
    page: typeof payload.page === 'number' ? payload.page : fallbackPage,
    limit: typeof payload.limit === 'number' ? payload.limit : fallbackLimit,
    ...(payload.facets
      ? {
          facets: {
            users: facetList(payload.facets.users),
            actions: facetList(payload.facets.actions),
          },
        }
      : {}),
  }
}

export const AUDIT_RESULT_OPTIONS = [
  { value: 'success', label: 'Succeeded' },
  { value: 'client_error', label: 'Client error' },
  { value: 'server_error', label: 'Server error' },
] as const

export type AuditTimePreset = 'any' | '24h' | '7d' | '30d' | 'custom'

export type AuditTimeRange = { preset: AuditTimePreset; from: string; to: string }

export const AUDIT_TIME_PRESETS: ReadonlyArray<{ value: AuditTimePreset; label: string }> = [
  { value: 'any', label: 'Any time' },
  { value: '24h', label: 'Last 24 hours' },
  { value: '7d', label: 'Last 7 days' },
  { value: '30d', label: 'Last 30 days' },
  { value: 'custom', label: 'Custom range' },
]

const PRESET_SECONDS: Partial<Record<AuditTimePreset, number>> = {
  '24h': 86_400,
  '7d': 7 * 86_400,
  '30d': 30 * 86_400,
}

/** The from/to query values for a time range; relative ranges end now. */
export function auditTimeBounds(
  range: AuditTimeRange,
  nowSeconds = Math.floor(Date.now() / 1000),
): { from: string; to: string } {
  const span = PRESET_SECONDS[range.preset]
  if (span) return { from: String(nowSeconds - span), to: '' }
  if (range.preset === 'custom') return { from: range.from, to: range.to }
  return { from: '', to: '' }
}

/** The label of a time range for its filter chip, or null for any time. */
export function auditTimeLabel(range: AuditTimeRange): string | null {
  if (range.preset === 'custom') {
    if (range.from && range.to) return `${range.from} – ${range.to}`
    if (range.from) return `Since ${range.from}`
    if (range.to) return `Until ${range.to}`
    return null
  }
  if (range.preset === 'any') return null
  return AUDIT_TIME_PRESETS.find((preset) => preset.value === range.preset)?.label ?? null
}

function csvCell(value: unknown): string {
  let text = value === undefined || value === null ? '' : String(value)
  // Spreadsheets run cells that start with these as formulas.
  if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

/** Audit entries as CSV, newest first as given, with the actor's name when known. */
export function auditLogsCsv(
  logs: AuditLog[],
  actorName: (userId: string | undefined) => string,
): string {
  const header = [
    'time',
    'user_id',
    'user',
    'action',
    'resource',
    'method',
    'path',
    'status_code',
    'ip',
    'user_agent',
  ]
  const rows = logs.map((log) => [
    log.createdAt ? new Date(log.createdAt * 1000).toISOString() : '',
    log.userId ?? '',
    actorName(log.userId),
    log.action,
    log.resource,
    log.method,
    log.path,
    log.statusCode || '',
    log.ip,
    log.userAgent,
  ])
  return `${[header, ...rows].map((row) => row.map(csvCell).join(',')).join('\n')}\n`
}

export const isAbortError = (error: unknown) =>
  error instanceof DOMException
    ? error.name === 'AbortError'
    : error instanceof Error && error.name === 'AbortError'

export const createLatestAuditRequest = () => {
  let sequence = 0
  let activeController: AbortController | null = null

  return {
    start() {
      activeController?.abort()
      const controller = new AbortController()
      const requestSequence = ++sequence
      activeController = controller

      return {
        signal: controller.signal,
        isCurrent: () => requestSequence === sequence,
        finish: () => {
          if (requestSequence === sequence) {
            activeController = null
          }
        },
      }
    },
    abort() {
      sequence += 1
      activeController?.abort()
      activeController = null
    },
  }
}
