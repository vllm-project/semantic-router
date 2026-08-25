import type {
  ManagementCostSummary,
  ManagementRequestLog,
  RateLimitRule,
} from './routerManagementTypes'

export type AccessStatus = 'active' | 'disabled'

export interface TeamMembership {
  teamId: string
  userId: string
  role: 'admin' | 'member'
  revision?: number
}

export interface AccessUser {
  id: string
  email: string
  name: string
  status: AccessStatus
  revision?: number
  accessGroupIds: string[]
  budgetId?: string
  memberships: TeamMembership[]
  createdAt?: string
  updatedAt?: string
}

export interface AccessTeam {
  id: string
  name: string
  description: string
  status: AccessStatus
  revision?: number
  members: TeamMembership[]
  accessGroupIds: string[]
  budgetId: string
  createdAt?: string
  updatedAt?: string
}

export interface SelfTeamCatalog {
  items: AccessTeam[]
  members: Array<Pick<AccessUser, 'id' | 'email' | 'name' | 'status'>>
  accessGroups: AccessGroup[]
  budgets: AccessBudget[]
}

export interface AccessAPIKey {
  id: string
  name: string
  prefix: string
  credentialId?: string
  contextTeamId?: string
  ownerType: 'user' | 'team'
  ownerId: string
  budgetId?: string
  status: AccessStatus
  expiresAt?: string
  lastUsedAt?: string
  accessGroupIds: string[]
  effectiveAccess?: AccessResourceRef[]
  accessPolicySources?: Array<'key' | 'user' | 'team'>
  effectiveBudgetId?: string
  budgetPolicySource?: 'key' | 'user' | 'team'
  quota?: APIKeyQuotaSnapshot
  revision?: number
  createdAt?: string
}

export interface AccessResourceRef {
  resourceType: 'entrypoint' | 'model'
  resourceId: string
}

export interface AccessResourceOption extends AccessResourceRef {
  name: string
  status: 'draft' | 'active' | 'disabled'
}

export interface QuotaMeter {
  policyId: string
  ruleId: string
  bindingId: string
  metric: string
  algorithm: string
  accounting: string
  enforcement: string
  currency?: string
  limit: string
  used: string
  remaining: string | null
  overage?: string
  resetsAt?: string
  window?: string
  completeness: string
  capacityState: string
}

export interface APIKeyQuotaSnapshot {
  budgetId: string
  budgetName: string
  source: 'key' | 'user' | 'team'
  meters: QuotaMeter[]
  asOf: string
}

export interface CreatedAccessAPIKey extends AccessAPIKey {
  secret: string
  deliveryExpiresAt?: string
}

export interface AccessGroup {
  id: string
  name: string
  description: string
  resources: AccessResourceRef[]
  assignmentCount: number
  revision?: number
  status?: AccessStatus
  createdAt?: string
  updatedAt?: string
}

export interface AccessBudget {
  id: string
  name: string
  description: string
  rules: RateLimitRule[]
  enabled: boolean
  assignmentCount: number
  revision?: number
  createdAt?: string
  updatedAt?: string
}

export interface InlineRateLimitPolicyDraft {
  name: string
  description: string
  rules: RateLimitRule[]
}

export interface AccessUsageEvent {
  id: string
  requestId: string
  admissionId?: string
  keyId: string
  userId?: string
  teamId?: string
  namespaceId?: string
  model: string
  entrypointId?: string
  recipeId?: string
  streaming?: boolean
  toolCall?: boolean
  costs?: ManagementRequestLog['costs']
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
  users: string | null
  teams: string | null
  activeKeys: string | null
  expiringKeys: string | null
  accessGroups: string | null
  enabledBudgets: string | null
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
  p95LatencyMs: number
  averageTtftMs: number
  p95TtftMs: number
  costs: ManagementCostSummary[]
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
  costs: ManagementCostSummary[]
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
  costs: ManagementCostSummary[]
  series: UsagePoint[]
  byModel: UsageSlice[]
  byUser: UsageSlice[]
  byTeam: UsageSlice[]
  byKey: UsageSlice[]
}

export interface AccessPage<T> {
  items: T[]
  limit: number
  hasMore: boolean
  nextCursor?: string
  total: number
}

export interface AccessListParams {
  q?: string
  limit?: number
  cursor?: string
  status?: string
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
