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

export type AuditLogPage = {
  logs: AuditLog[]
  total: number
  page: number
  limit: number
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
  return {
    logs,
    total: typeof payload.total === 'number' ? payload.total : logs.length,
    page: typeof payload.page === 'number' ? payload.page : fallbackPage,
    limit: typeof payload.limit === 'number' ? payload.limit : fallbackLimit,
  }
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
