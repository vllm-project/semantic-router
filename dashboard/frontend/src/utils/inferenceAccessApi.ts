export type AccessStatus = 'active' | 'disabled'

export interface AccessUser {
  id: string
  email: string
  name: string
  status: AccessStatus
  createdAt?: string
  updatedAt?: string
}

export interface AccessTeam {
  id: string
  name: string
  description: string
  status: AccessStatus
  userIds: string[]
  accessGroupIds: string[]
  budget?: { rpm: number; tpm: number; dailyTokens: number }
  createdAt?: string
  updatedAt?: string
}

export interface AccessAPIKey {
  id: string
  name: string
  prefix: string
  userId?: string
  teamId?: string
  effectiveTeamId?: string
  budgetId?: string
  status: AccessStatus
  expiresAt?: string
  lastUsedAt?: string
  accessGroupIds: string[]
  modelPatterns?: string[]
  budget?: { rpm: number; tpm: number; dailyTokens: number }
  createdAt?: string
}

export interface CreatedAccessAPIKey extends AccessAPIKey {
  secret: string
}

export interface AccessBinding {
  subjectType: 'user' | 'team' | 'key'
  subjectId: string
}

export interface AccessGroup {
  id: string
  name: string
  description: string
  modelPatterns: string[]
  bindings: AccessBinding[]
  createdAt?: string
  updatedAt?: string
}

export interface AccessBudget {
  id: string
  name: string
  scopeType: 'global' | 'user' | 'team' | 'key'
  scopeId: string
  rpm: number
  tpm: number
  dailyTokens: number
  enabled: boolean
  createdAt?: string
  updatedAt?: string
}

export interface AccessUsageEvent {
  id: string
  requestId: string
  keyId: string
  userId?: string
  teamId?: string
  model: string
  statusCode: number
  promptTokens: number
  completionTokens: number
  totalTokens: number
  latencyMs: number
  ttftMs?: number
  errorCode?: string
  metadata?: Record<string, unknown>
  createdAt: string
}

export interface AccessAuditEvent {
  id: string
  actorEmail?: string
  action: string
  resourceType: string
  resourceId?: string
  details?: Record<string, unknown>
  createdAt: string
}

export interface AccessOverview {
  users: number
  teams: number
  activeKeys: number
  expiringKeys: number
  accessGroups: number
  enabledBudgets: number
  requestsToday: number
  successfulToday: number
  tokensToday: number
  p95LatencyMs: number
}

export interface UsagePoint {
  bucket: string
  requests: number
  successful: number
  failed: number
  promptTokens: number
  completionTokens: number
  totalTokens: number
  averageLatencyMs: number
}

export interface UsageSlice {
  id: string
  requests: number
  successful: number
  failed: number
  promptTokens: number
  completionTokens: number
  totalTokens: number
  averageLatencyMs: number
  p95LatencyMs: number
}

export interface UsageSummary {
  granularity: 'minute' | 'hour' | 'day'
  requests: number
  successful: number
  failed: number
  promptTokens: number
  completionTokens: number
  totalTokens: number
  activeKeys: number
  averageLatencyMs: number
  p95LatencyMs: number
  averageTtftMs: number
  p95TtftMs: number
  series: UsagePoint[]
  byModel: UsageSlice[]
  byUser: UsageSlice[]
  byTeam: UsageSlice[]
  byKey: UsageSlice[]
}

export interface AccessPage<T> {
  items: T[]
  total: number
  limit: number
  offset: number
  hasMore: boolean
}

export interface AccessListParams {
  q?: string
  limit?: number
  offset?: number
}

export interface UsageFilter extends AccessListParams {
  userId?: string
  teamId?: string
  keyId?: string
  model?: string
  from?: string
  to?: string
  granularity?: 'auto' | 'minute' | 'hour' | 'day'
  timezoneOffset?: number
}

const base = '/api/v1/access-control'

const query = (values: object) => {
  const params = new URLSearchParams()
  Object.entries(values as Record<string, string | number | undefined>).forEach(([key, value]) => {
    if (value !== undefined && value !== '') params.set(key, String(value))
  })
  return params.toString() ? `?${params.toString()}` : ''
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    ...init,
    headers: init?.body ? { 'Content-Type': 'application/json', ...init.headers } : init?.headers,
  })
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      error?: { message?: string }
    } | null
    throw new Error(payload?.error?.message || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

async function selfRequest<T>(path: string, init?: RequestInit): Promise<T> {
  return request<T>(`/self${path}`, init)
}

export const inferenceAccessApi = {
  overview: () => request<AccessOverview>('/overview'),
  users: (params: AccessListParams = {}) =>
    request<AccessPage<AccessUser>>(`/users${query(params)}`),
  user: (id: string) => request<AccessUser>(`/users/${encodeURIComponent(id)}`),
  saveUser: (item: Partial<AccessUser>) =>
    request<AccessUser>(item.id ? `/users/${item.id}` : '/users', {
      method: item.id ? 'PUT' : 'POST',
      body: JSON.stringify(item),
    }),
  deleteUser: (id: string) => request(`/users/${id}`, { method: 'DELETE' }),
  teams: (params: AccessListParams = {}) =>
    request<AccessPage<AccessTeam>>(`/teams${query(params)}`),
  team: (id: string) => request<AccessTeam>(`/teams/${encodeURIComponent(id)}`),
  saveTeam: (item: Partial<AccessTeam>) =>
    request<AccessTeam>(item.id ? `/teams/${item.id}` : '/teams', {
      method: item.id ? 'PUT' : 'POST',
      body: JSON.stringify(item),
    }),
  deleteTeam: (id: string) => request(`/teams/${id}`, { method: 'DELETE' }),
  keys: (params: AccessListParams = {}) =>
    request<AccessPage<AccessAPIKey>>(`/api-keys${query(params)}`),
  key: (id: string) => request<AccessAPIKey>(`/api-keys/${encodeURIComponent(id)}`),
  keySecret: (id: string) =>
    request<{ secret: string }>(`/api-keys/${encodeURIComponent(id)}/secret`),
  createKey: (item: Partial<AccessAPIKey>) =>
    request<CreatedAccessAPIKey>('/api-keys', { method: 'POST', body: JSON.stringify(item) }),
  saveKey: (item: Partial<AccessAPIKey> & { id: string }) =>
    request<AccessAPIKey>(`/api-keys/${encodeURIComponent(item.id)}`, {
      method: 'PUT',
      body: JSON.stringify(item),
    }),
  rotateKey: (id: string) =>
    request<CreatedAccessAPIKey>(`/api-keys/${encodeURIComponent(id)}/rotate`, { method: 'POST' }),
  setKeyStatus: (id: string, status: AccessStatus) =>
    request<AccessAPIKey>(`/api-keys/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    }),
  groups: (params: AccessListParams = {}) =>
    request<AccessPage<AccessGroup>>(`/access-groups${query(params)}`),
  group: (id: string) => request<AccessGroup>(`/access-groups/${encodeURIComponent(id)}`),
  saveGroup: (item: Partial<AccessGroup>) =>
    request<AccessGroup>(item.id ? `/access-groups/${item.id}` : '/access-groups', {
      method: item.id ? 'PUT' : 'POST',
      body: JSON.stringify(item),
    }),
  deleteGroup: (id: string) => request(`/access-groups/${id}`, { method: 'DELETE' }),
  budgets: (params: AccessListParams = {}) =>
    request<AccessPage<AccessBudget>>(`/budgets${query(params)}`),
  budget: (id: string) => request<AccessBudget>(`/budgets/${encodeURIComponent(id)}`),
  saveBudget: (item: Partial<AccessBudget>) =>
    request<AccessBudget>(item.id ? `/budgets/${item.id}` : '/budgets', {
      method: item.id ? 'PUT' : 'POST',
      body: JSON.stringify(item),
    }),
  deleteBudget: (id: string) => request(`/budgets/${id}`, { method: 'DELETE' }),
  usage: (filter: UsageFilter = {}) => request<UsageSummary>(`/usage${query(filter)}`),
  requestLogs: (filter: UsageFilter = {}) =>
    request<AccessPage<AccessUsageEvent>>(`/request-logs${query(filter)}`),
  requestLog: (id: string) => request<AccessUsageEvent>(`/request-logs/${encodeURIComponent(id)}`),
  auditLogs: (filter: AccessListParams = {}) =>
    request<AccessPage<AccessAuditEvent>>(`/audit-logs${query(filter)}`),
  selfOverview: () => selfRequest<AccessOverview>('/overview'),
  selfTeams: () => selfRequest<{ items: AccessTeam[] }>('/teams'),
  selfKeys: () => selfRequest<AccessPage<AccessAPIKey>>('/api-keys'),
  selfKey: (id: string) => selfRequest<AccessAPIKey>(`/api-keys/${encodeURIComponent(id)}`),
  selfKeySecret: (id: string) =>
    selfRequest<{ secret: string }>(`/api-keys/${encodeURIComponent(id)}/secret`),
  createSelfKey: (name: string) =>
    selfRequest<CreatedAccessAPIKey>('/api-keys', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),
  rotateSelfKey: (id: string) =>
    selfRequest<CreatedAccessAPIKey>(`/api-keys/${encodeURIComponent(id)}/rotate`, {
      method: 'POST',
    }),
  setSelfKeyStatus: (id: string, status: AccessStatus) =>
    selfRequest<AccessAPIKey>(`/api-keys/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    }),
  selfUsage: (filter: UsageFilter = {}) => selfRequest<UsageSummary>(`/usage${query(filter)}`),
  selfRequestLogs: (filter: UsageFilter = {}) =>
    selfRequest<AccessPage<AccessUsageEvent>>(`/request-logs${query(filter)}`),
  selfRequestLog: (id: string) =>
    selfRequest<AccessUsageEvent>(`/request-logs/${encodeURIComponent(id)}`),
}
