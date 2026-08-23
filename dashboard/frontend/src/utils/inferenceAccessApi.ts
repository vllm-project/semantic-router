import {
  assertManagementMe,
  getManagementNamespace,
  managementOperationRequest,
  type ManagementRequestOptions,
} from './managementApiContract'
import {
  MANAGEMENT_API_HEADERS,
  type ManagementApiOperationId,
} from '../generated/managementApiContract'
import type {
  AccessPolicyGrant,
  IssuedAPIKeySecret,
  ManagementAPIKey,
  ManagementAccessStatistics,
  ManagementAccessPolicy,
  ManagementCredential,
  ManagementCostSummary,
  ManagementEffectiveQuota,
  ManagementEffectivePolicy,
  ManagementMembership,
  ManagementPage,
  ManagementPolicyBinding,
  ManagementRequestLog,
  ManagementRequestLogDetail,
  ManagementRateLimitPolicy,
  ManagementTeam,
  ManagementUsageBreakdown,
  ManagementUsageSeries,
  ManagementUsageSummary,
  ManagementUser,
  MutationReceipt,
  RateLimitRule,
  ResourceDetail,
  SelfInferenceKey,
  SubjectType,
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
  // A bounded display count, not an offset-pagination contract.
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

const idempotencyHeaders = () => ({
  [MANAGEMENT_API_HEADERS.idempotencyKey]: crypto.randomUUID(),
})
const etag = (kind: string, revision: number) => `"${kind}:${revision}"`

const query = (values: Record<string, string | number | undefined>) => {
  const params = new URLSearchParams()
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== '') params.set(key, String(value))
  })
  return params
}

async function request<T>(
  operationId: ManagementApiOperationId,
  options: ManagementRequestOptions & {
    pathParameters?: Record<string, string | number>
  } = {},
): Promise<T> {
  const invoke = managementOperationRequest as (
    id: ManagementApiOperationId,
    requestOptions?: ManagementRequestOptions & {
      pathParameters?: Record<string, string | number>
    },
  ) => Promise<unknown>
  return (await invoke(operationId, options)) as T
}

const resource = <T>(payload: ResourceDetail<T>) => payload.data

function viewPage<T, U>(
  page: ManagementPage<T>,
  map: (item: T) => U,
  clientFilter?: string,
): AccessPage<U> {
  let items = page.data.map(map)
  if (clientFilter) {
    const needle = clientFilter.toLocaleLowerCase()
    items = items.filter((item) => JSON.stringify(item).toLocaleLowerCase().includes(needle))
  }
  return {
    items,
    limit: page.page.pageSize,
    hasMore: page.page.hasMore,
    nextCursor: page.page.nextCursor,
    total: items.length + (page.page.hasMore ? 1 : 0),
  }
}

function listQuery(params: AccessListParams) {
  return query({
    cursor: params.cursor,
    pageSize: params.limit,
    status: params.status,
    search: params.q?.trim() || undefined,
  })
}

async function allPages<T>(
  operationId: ManagementApiOperationId,
  pathParameters?: Record<string, string | number>,
  params: AccessListParams = {},
): Promise<T[]> {
  const items: T[] = []
  let cursor = params.cursor
  do {
    const page = await request<ManagementPage<T>>(operationId, {
      pathParameters,
      query: listQuery({ ...params, cursor, limit: Math.min(params.limit ?? 200, 200) }),
    })
    items.push(...page.data)
    cursor = page.page.hasMore ? page.page.nextCursor : undefined
  } while (cursor)
  return items
}

function mapMembership(item: ManagementMembership): TeamMembership {
  return { teamId: item.teamId, userId: item.userId, role: item.role, revision: item.revision }
}

function mapUser(
  item: ManagementUser,
  memberships: ManagementMembership[] = [],
  accessGroupIds: string[] = [],
  budgetId?: string,
): AccessUser {
  return {
    id: item.userId,
    email: item.email,
    name: item.displayName,
    status: item.status === 'active' ? 'active' : 'disabled',
    revision: item.revision,
    accessGroupIds,
    budgetId,
    memberships: memberships.map(mapMembership),
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
  }
}

function mapTeam(
  item: ManagementTeam,
  members: ManagementMembership[] = [],
  accessGroupIds: string[] = [],
  budgetId = '',
): AccessTeam {
  return {
    id: item.teamId,
    name: item.name,
    description: item.description,
    status: item.status === 'active' ? 'active' : 'disabled',
    revision: item.revision,
    members: members.map(mapMembership),
    accessGroupIds,
    budgetId,
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
  }
}

function mapKey(item: ManagementAPIKey, credential?: ManagementCredential): AccessAPIKey {
  return {
    id: item.keyId,
    name: item.name,
    prefix: credential?.kid ?? '',
    credentialId: credential?.credentialId,
    ownerType: item.owner.type,
    ownerId: item.owner.id,
    contextTeamId: item.contextTeamId,
    status: item.status === 'active' ? 'active' : 'disabled',
    expiresAt: item.expiresAt,
    lastUsedAt: item.lastUsedAt,
    accessGroupIds: [],
    revision: item.revision,
    createdAt: item.createdAt,
  }
}

function mapGroup(item: ManagementAccessPolicy): AccessGroup {
  return {
    id: item.policyId,
    name: item.name,
    description: item.description,
    resources: item.grants
      .filter((grant) => grant.effect === 'allow' && grant.permission === 'invoke')
      .map((grant) => ({
        resourceType: grant.resourceType,
        resourceId: grant.resourceId,
      })),
    assignmentCount: 0,
    revision: item.revision,
    status: item.status === 'active' ? 'active' : 'disabled',
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
  }
}

function mapBudget(item: ManagementRateLimitPolicy): AccessBudget {
  return {
    id: item.policyId,
    name: item.name,
    description: item.description,
    rules: item.rules,
    enabled: item.status === 'active',
    assignmentCount: 0,
    revision: item.revision,
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
  }
}

function accessGrants(resources: AccessResourceRef[]): AccessPolicyGrant[] {
  return resources.flatMap(({ resourceType, resourceId }) => [
    { resourceType, resourceId, permission: 'discover', effect: 'allow' },
    { resourceType, resourceId, permission: 'invoke', effect: 'allow' },
  ])
}

function rateLimitRuleInput(rule: RateLimitRule): Omit<RateLimitRule, 'ordinal'> {
  const input = { ...rule }
  delete input.ordinal
  return input
}

async function subjectBindings(type: SubjectType, id: string) {
  const filters = query({ subjectType: type, subjectId: id, pageSize: 200, status: 'active' })
  const [access, rate] = await Promise.all([
    request<ManagementPage<ManagementPolicyBinding>>('getAccessPolicyBindings', {
      query: filters,
    }),
    request<ManagementPage<ManagementPolicyBinding>>('getRateLimitBindings', {
      query: filters,
    }),
  ])
  return {
    accessGroupIds: access.data.map((binding) => binding.policyId),
    budgetId: rate.data.find((binding) => binding.mode === 'allocation')?.policyId,
    accessBindings: access.data,
    rateBindings: rate.data,
  }
}

async function syncSubjectBindings(
  type: SubjectType,
  id: string,
  accessGroupIds: string[] | undefined,
  budgetId: string | undefined,
): Promise<void> {
  if (accessGroupIds === undefined && budgetId === undefined) return
  const current = await subjectBindings(type, id)
  const subject = { type, id }

  if (accessGroupIds !== undefined) {
    const desired = new Set(accessGroupIds)
    for (const binding of current.accessBindings) {
      if (!desired.has(binding.policyId)) {
        await request('deleteAccessPolicyBindingsByBindingId', {
          pathParameters: { bindingId: binding.bindingId },
          headers: {
            [MANAGEMENT_API_HEADERS.ifMatch]: etag('access-policy-binding', binding.revision),
          },
        })
      }
    }
    const existing = new Set(current.accessBindings.map((binding) => binding.policyId))
    for (const policyId of desired) {
      if (!existing.has(policyId)) {
        await request('postAccessPolicyBindings', {
          headers: idempotencyHeaders(),
          body: { policyId, subject },
        })
      }
    }
  }

  if (budgetId !== undefined) {
    const activeAllocation = current.rateBindings.find((binding) => binding.mode === 'allocation')
    if (activeAllocation?.policyId !== budgetId) {
      if (activeAllocation) {
        await request('deleteRateLimitBindingsByBindingId', {
          pathParameters: { bindingId: activeAllocation.bindingId },
          headers: {
            [MANAGEMENT_API_HEADERS.ifMatch]: etag('rate-limit-binding', activeAllocation.revision),
          },
        })
      }
      if (budgetId) {
        await request('postRateLimitBindings', {
          headers: idempotencyHeaders(),
          body: { policyId: budgetId, subject, mode: 'allocation' },
        })
      }
    }
  }
}

async function syncTeamMemberships(teamId: string, desired: TeamMembership[] | undefined) {
  if (desired === undefined) return
  const current = await allPages<ManagementMembership>('getTeamsByTeamIdMembers', { teamId })
  const desiredByUser = new Map(desired.map((membership) => [membership.userId, membership]))

  for (const membership of current) {
    const next = desiredByUser.get(membership.userId)
    if (!next) {
      await request('deleteTeamsByTeamIdMembersByUserId', {
        pathParameters: { teamId, userId: membership.userId },
        headers: {
          [MANAGEMENT_API_HEADERS.ifMatch]: etag('membership', membership.revision),
        },
      })
    } else if (next.role !== membership.role) {
      await request('patchTeamsByTeamIdMembersByUserId', {
        pathParameters: { teamId, userId: membership.userId },
        body: { role: next.role },
        headers: {
          [MANAGEMENT_API_HEADERS.ifMatch]: etag('membership', membership.revision),
        },
      })
    }
  }

  const existingUsers = new Set(current.map((membership) => membership.userId))
  for (const membership of desired) {
    if (!existingUsers.has(membership.userId)) {
      await request('putTeamsByTeamIdMembersByUserId', {
        pathParameters: { teamId, userId: membership.userId },
        body: { role: membership.role },
        headers: idempotencyHeaders(),
      })
    }
  }
}

function quotaSnapshot(quota: ManagementEffectiveQuota): APIKeyQuotaSnapshot | undefined {
  if (!quota.meters.length) return undefined
  const first = quota.meters[0]
  const meters: QuotaMeter[] = quota.meters.map((meter) => ({
    policyId: meter.policyId,
    ruleId: meter.ruleId,
    bindingId: meter.bindingId,
    metric: meter.metric,
    algorithm: meter.algorithm,
    accounting: meter.accounting,
    enforcement: meter.enforcement,
    currency: meter.currency,
    limit: meter.limit,
    used: meter.used,
    remaining: meter.remaining,
    overage: meter.overage,
    resetsAt: meter.resetAt,
    window: meter.window,
    completeness: meter.completeness,
    capacityState: meter.capacityState,
  }))
  const source = ['api_key', 'user', 'team'].includes(first.source.subjectType)
    ? ((first.source.subjectType === 'api_key' ? 'key' : first.source.subjectType) as
        | 'key'
        | 'user'
        | 'team')
    : 'key'
  return {
    budgetId: first.policyId,
    budgetName: first.policyId,
    source,
    meters,
    asOf: quota.asOf,
  }
}

async function keyDetail(id: string): Promise<AccessAPIKey> {
  const detail = await request<ResourceDetail<ManagementAPIKey>>('getApiKeysByKeyId', {
    pathParameters: { keyId: id },
  })
  const [credentials, bindings, policy] = await Promise.all([
    request<ManagementPage<ManagementCredential>>('getApiKeysByKeyIdCredentials', {
      pathParameters: { keyId: id },
      query: new URLSearchParams({ pageSize: '200' }),
    }),
    subjectBindings('api_key', id),
    request<ResourceDetail<ManagementEffectivePolicy> | ManagementEffectivePolicy>(
      'getApiKeysByKeyIdEffectivePolicy',
      { pathParameters: { keyId: id } },
    ),
  ])
  const key = mapKey(
    detail.data,
    credentials.data.find((credential) => credential.status === 'active') ?? credentials.data[0],
  )
  key.accessGroupIds = bindings.accessGroupIds
  key.budgetId = bindings.budgetId
  const policyData = 'data' in policy ? policy.data : policy
  const effectiveGrants = policyData.access.grants.filter(
    (grant) =>
      grant.effect === 'allow' &&
      grant.permissions.includes('invoke') &&
      (grant.resourceType === 'entrypoint' || grant.resourceType === 'model'),
  )
  key.effectiveAccess = effectiveGrants.map((grant) => ({
    resourceType: grant.resourceType,
    resourceId: grant.resourceId,
  }))
  key.accessPolicySources = [
    ...new Set(
      effectiveGrants
        .map((grant) => grant.source.subjectType)
        .filter(
          (source): source is 'api_key' | 'user' | 'team' =>
            source === 'api_key' || source === 'user' || source === 'team',
        )
        .map((source) => (source === 'api_key' ? ('key' as const) : source)),
    ),
  ]
  key.quota = quotaSnapshot(policyData.quota)
  key.effectiveBudgetId = key.quota?.budgetId
  key.budgetPolicySource = key.quota?.source
  return key
}

function issuedKey(item: IssuedAPIKeySecret): CreatedAccessAPIKey {
  return {
    ...mapKey(item.data, item.credential),
    secret: item.secret,
    deliveryExpiresAt: item.deliveryExpiresAt,
    accessGroupIds: item.accessPolicyBindings?.map((binding) => binding.policyId) ?? [],
    budgetId: item.rateLimitOverride?.policyId,
  }
}

async function mutateAndRead<T>(
  operationId: ManagementApiOperationId,
  pathParameters: Record<string, string | number> | undefined,
  body: unknown,
  detail: (id: string) => Promise<T>,
  headers: Record<string, string>,
): Promise<T> {
  const receipt = await request<MutationReceipt>(operationId, { pathParameters, body, headers })
  if (!receipt.resource?.id) throw new Error('Router Management mutation returned no resource.')
  return detail(receipt.resource.id)
}

const usageQuery = (filter: UsageFilter) =>
  query({
    userId: filter.userId,
    teamId: filter.teamId,
    apiKeyId: filter.keyId,
    logicalModelId: filter.model,
    start: filter.from,
    end: filter.to,
    grain: filter.granularity ?? 'auto',
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    cursor: filter.cursor,
    pageSize: filter.limit,
  })

const usageBreakdownQuery = (filter: UsageFilter, dimension: string) => {
  const params = usageQuery(filter)
  params.set('dimension', dimension)
  return params
}

const emptyUsage = (): UsageSummary => ({
  granularity: 'hour',
  requests: 0,
  successful: 0,
  failed: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  activeKeys: 0,
  averageLatencyMs: 0,
  p95LatencyMs: 0,
  averageTtftMs: 0,
  p95TtftMs: 0,
  costs: [],
  series: [],
  byModel: [],
  byUser: [],
  byTeam: [],
  byKey: [],
})

async function loadUsage(
  filter: UsageFilter,
  summaryOperationId:
    | 'getUsage'
    | 'getApiKeysByKeyIdUsage'
    | 'getUsersByUserIdUsage'
    | 'getTeamsByTeamIdUsage',
  pathParameters: Record<string, string> | undefined,
  summaryFilter: UsageFilter = filter,
): Promise<UsageSummary> {
  const [summary, series, byModel, byUser, byTeam, byKey] = await Promise.all([
    request<ManagementUsageSummary>(summaryOperationId, {
      pathParameters,
      query: usageQuery(summaryFilter),
    }),
    request<ManagementUsageSeries>('getUsageSeries', { query: usageQuery(filter) }),
    request<ManagementUsageBreakdown>('getUsageBreakdowns', {
      query: usageBreakdownQuery(filter, 'logical_model'),
    }),
    request<ManagementUsageBreakdown>('getUsageBreakdowns', {
      query: usageBreakdownQuery(filter, 'user'),
    }),
    request<ManagementUsageBreakdown>('getUsageBreakdowns', {
      query: usageBreakdownQuery(filter, 'team'),
    }),
    request<ManagementUsageBreakdown>('getUsageBreakdowns', {
      query: usageBreakdownQuery(filter, 'api_key'),
    }),
  ])
  const totals = summary.totals
  const requests = Number(totals.requests)
  const successful = Number(totals.successfulRequests)
  const slice = (item: ManagementUsageBreakdown['rows'][number]): UsageSlice => ({
    id: item.value,
    requests: Number(item.totals.requests),
    successful: Number(item.totals.successfulRequests),
    failed: Math.max(0, Number(item.totals.requests) - Number(item.totals.successfulRequests)),
    promptTokens: Number(item.totals.inputTokens),
    completionTokens: Number(item.totals.outputTokens),
    totalTokens: Number(item.totals.totalTokens),
    averageLatencyMs: item.totals.latency.averageMilliseconds,
    p95LatencyMs: item.totals.latency.p95Milliseconds,
    costs: item.totals.costs,
  })
  return {
    ...emptyUsage(),
    granularity: summary.grain,
    requests,
    successful,
    failed: Math.max(0, requests - successful),
    promptTokens: Number(totals.inputTokens),
    completionTokens: Number(totals.outputTokens),
    totalTokens: Number(totals.totalTokens),
    averageLatencyMs: totals.latency.averageMilliseconds,
    p95LatencyMs: totals.latency.p95Milliseconds,
    averageTtftMs: totals.ttft.averageMilliseconds,
    p95TtftMs: totals.ttft.p95Milliseconds,
    costs: totals.costs,
    series: series.points.map((point) => ({
      bucket: point.bucketStart,
      requests: Number(point.totals.requests),
      successful: Number(point.totals.successfulRequests),
      failed: Math.max(0, Number(point.totals.requests) - Number(point.totals.successfulRequests)),
      promptTokens: Number(point.totals.inputTokens),
      completionTokens: Number(point.totals.outputTokens),
      totalTokens: Number(point.totals.totalTokens),
      averageLatencyMs: point.totals.latency.averageMilliseconds,
      p95LatencyMs: point.totals.latency.p95Milliseconds,
      averageTtftMs: point.totals.ttft.averageMilliseconds,
      p95TtftMs: point.totals.ttft.p95Milliseconds,
      costs: point.totals.costs,
    })),
    byModel: byModel.rows.map(slice),
    byUser: byUser.rows.map(slice),
    byTeam: byTeam.rows.map(slice),
    byKey: byKey.rows.map(slice),
  }
}

async function usage(filter: UsageFilter = {}): Promise<UsageSummary> {
  return loadUsage(filter, 'getUsage', undefined)
}

async function keyUsage(keyId: string, filter: UsageFilter = {}): Promise<UsageSummary> {
  const exactFilter = { ...filter, keyId }
  const summaryFilter = { ...filter }
  delete summaryFilter.keyId
  return loadUsage(exactFilter, 'getApiKeysByKeyIdUsage', { keyId }, summaryFilter)
}

async function userUsage(userId: string, filter: UsageFilter = {}): Promise<UsageSummary> {
  const exactFilter = { ...filter, userId }
  const summaryFilter = { ...filter }
  delete summaryFilter.userId
  return loadUsage(exactFilter, 'getUsersByUserIdUsage', { userId }, summaryFilter)
}

async function teamUsage(teamId: string, filter: UsageFilter = {}): Promise<UsageSummary> {
  const exactFilter = { ...filter, teamId }
  const summaryFilter = { ...filter }
  delete summaryFilter.teamId
  return loadUsage(exactFilter, 'getTeamsByTeamIdUsage', { teamId }, summaryFilter)
}

function mapRequestLog(item: ManagementRequestLog): AccessUsageEvent {
  const inputTokens = Number(item.inputTokens)
  const outputTokens = Number(item.outputTokens)
  const namespaceID = getManagementNamespace()
  return {
    id: namespaceID ? `${namespaceID}:${item.admissionId}` : item.admissionId,
    requestId: item.metadata?.externalRequestId || item.eventId,
    admissionId: item.admissionId,
    keyId: item.apiKeyId ?? '',
    userId: item.userId,
    teamId: item.teamId,
    namespaceId: namespaceID || undefined,
    model: item.entrypointId || item.recipeId || '',
    entrypointId: item.entrypointId,
    recipeId: item.recipeId,
    streaming: item.stream,
    toolCall: item.toolCall,
    costs: item.costs,
    statusCode: item.statusCode,
    promptTokens: inputTokens,
    completionTokens: outputTokens,
    totalTokens: inputTokens + outputTokens,
    latencyMs: item.latencyMilliseconds,
    ttftMs: item.ttftMilliseconds,
    errorCode: item.errorCode,
    metadata: item.metadata,
    createdAt: item.occurredAt,
  }
}

async function overview(): Promise<AccessOverview> {
  const [statistics, currentUsage] = await Promise.all([
    request<ManagementAccessStatistics>('getStatistics'),
    usage(),
  ])
  return {
    users: statistics.users ?? null,
    teams: statistics.teams ?? null,
    activeKeys: statistics.activeApiKeys ?? null,
    expiringKeys: statistics.expiringApiKeys ?? null,
    accessGroups: statistics.accessPolicies ?? null,
    enabledBudgets: statistics.activeRatePolicies ?? null,
    requestsToday: currentUsage.requests,
    successfulToday: currentUsage.successful,
    tokensToday: currentUsage.totalTokens,
    p95LatencyMs: currentUsage.p95LatencyMs,
  }
}

export const inferenceAccessApi = {
  overview,
  users: async (params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementUser>>('getUsers', {
      query: listQuery(params),
    })
    return viewPage(page, (item) => mapUser(item))
  },
  user: async (id: string) => {
    const [detail, memberships, bindings] = await Promise.all([
      request<ResourceDetail<ManagementUser>>('getUsersByUserId', {
        pathParameters: { userId: id },
      }),
      allPages<ManagementMembership>('getUsersByUserIdMemberships', { userId: id }),
      subjectBindings('user', id),
    ])
    return mapUser(detail.data, memberships, bindings.accessGroupIds, bindings.budgetId)
  },
  userSummary: async (id: string) =>
    mapUser(
      resource(
        await request<ResourceDetail<ManagementUser>>('getUsersByUserId', {
          pathParameters: { userId: id },
        }),
      ),
    ),
  saveUser: async (item: Partial<AccessUser> & { id: string }) => {
    await mutateAndRead(
      'patchUsersByUserId',
      { userId: item.id },
      { email: item.email, displayName: item.name, status: item.status },
      inferenceAccessApi.user,
      { [MANAGEMENT_API_HEADERS.ifMatch]: etag('user', item.revision ?? 0) },
    )
    await syncSubjectBindings('user', item.id, item.accessGroupIds, item.budgetId)
    return inferenceAccessApi.user(item.id)
  },
  deleteUser: async (id: string) => {
    const current = await inferenceAccessApi.user(id)
    await request('deleteUsersByUserId', {
      pathParameters: { userId: id },
      headers: { [MANAGEMENT_API_HEADERS.ifMatch]: etag('user', current.revision ?? 0) },
    })
  },
  teams: async (params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementTeam>>('getTeams', {
      query: listQuery(params),
    })
    return viewPage(page, (item) => mapTeam(item))
  },
  team: async (id: string) => {
    const [detail, members, bindings] = await Promise.all([
      request<ResourceDetail<ManagementTeam>>('getTeamsByTeamId', {
        pathParameters: { teamId: id },
      }),
      allPages<ManagementMembership>('getTeamsByTeamIdMembers', { teamId: id }),
      subjectBindings('team', id),
    ])
    return mapTeam(detail.data, members, bindings.accessGroupIds, bindings.budgetId)
  },
  teamSummary: async (id: string) =>
    mapTeam(
      resource(
        await request<ResourceDetail<ManagementTeam>>('getTeamsByTeamId', {
          pathParameters: { teamId: id },
        }),
      ),
    ),
  saveTeam: async (item: Partial<AccessTeam>) => {
    let team: AccessTeam
    if (!item.id) {
      team = await mutateAndRead(
        'postTeams',
        undefined,
        {
          name: item.name,
          description: item.description,
          accessPolicyIds: item.accessGroupIds,
          rateLimitPolicyId: item.budgetId,
        },
        inferenceAccessApi.team,
        idempotencyHeaders(),
      )
    } else {
      team = await mutateAndRead(
        'patchTeamsByTeamId',
        { teamId: item.id },
        { name: item.name, description: item.description, status: item.status },
        inferenceAccessApi.team,
        { [MANAGEMENT_API_HEADERS.ifMatch]: etag('team', item.revision ?? 0) },
      )
      await syncSubjectBindings('team', team.id, item.accessGroupIds, item.budgetId)
    }
    await syncTeamMemberships(team.id, item.members)
    return inferenceAccessApi.team(team.id)
  },
  deleteTeam: async (id: string) => {
    const current = await inferenceAccessApi.team(id)
    await request('deleteTeamsByTeamId', {
      pathParameters: { teamId: id },
      headers: { [MANAGEMENT_API_HEADERS.ifMatch]: etag('team', current.revision ?? 0) },
    })
  },
  keys: async (params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementAPIKey>>('getApiKeys', {
      query: listQuery(params),
    })
    return viewPage(page, (item) => mapKey(item))
  },
  key: keyDetail,
  keySummary: async (id: string) =>
    mapKey(
      resource(
        await request<ResourceDetail<ManagementAPIKey>>('getApiKeysByKeyId', {
          pathParameters: { keyId: id },
        }),
      ),
    ),
  keySecret: async (id: string) => {
    const credentials = await request<ManagementPage<ManagementCredential>>(
      'getApiKeysByKeyIdCredentials',
      {
        pathParameters: { keyId: id },
        query: new URLSearchParams({ pageSize: '200' }),
      },
    )
    const credential = credentials.data.find((item) => item.status === 'active' && item.revealable)
    if (!credential) throw new Error('This key has no revealable active credential.')
    return request<{ secret: string }>('postApiKeysByKeyIdCredentialsByCredentialIdReveal', {
      pathParameters: { keyId: id, credentialId: credential.credentialId },
    })
  },
  createKey: async (item: Partial<AccessAPIKey>, inlineRateLimit?: InlineRateLimitPolicyDraft) => {
    const response = await request<IssuedAPIKeySecret>('postApiKeys', {
      headers: idempotencyHeaders(),
      body: {
        name: item.name,
        owner: { type: item.ownerType, id: item.ownerId },
        contextTeamId: item.contextTeamId,
        expiresAt: item.expiresAt,
        revealable: true,
        accessPolicyIds: item.accessGroupIds,
        rateLimitOverride: inlineRateLimit
          ? {
              inlinePolicy: {
                name: inlineRateLimit.name,
                description: inlineRateLimit.description,
                rules: inlineRateLimit.rules.map(rateLimitRuleInput),
              },
            }
          : item.budgetId
            ? { policyId: item.budgetId }
            : undefined,
      },
    })
    return issuedKey(response)
  },
  saveKey: async (item: Partial<AccessAPIKey> & { id: string }) => {
    await request<ResourceDetail<ManagementAPIKey>>('patchApiKeysByKeyId', {
      pathParameters: { keyId: item.id },
      body: { name: item.name },
      headers: { [MANAGEMENT_API_HEADERS.ifMatch]: etag('key', item.revision ?? 0) },
    })
    await syncSubjectBindings('api_key', item.id, item.accessGroupIds, item.budgetId)
    return inferenceAccessApi.key(item.id)
  },
  rotateKey: async (id: string) => {
    const current = await inferenceAccessApi.key(id)
    const response = await request<IssuedAPIKeySecret>('postApiKeysByKeyIdCredentialsRotate', {
      pathParameters: { keyId: id },
      body: { overlapSeconds: 0, revealable: true },
      headers: {
        ...idempotencyHeaders(),
        [MANAGEMENT_API_HEADERS.ifMatch]: etag('key', current.revision ?? 0),
      },
    })
    return issuedKey(response)
  },
  setKeyStatus: async (id: string, status: AccessStatus) => {
    const current = await inferenceAccessApi.key(id)
    const operationId =
      status === 'active' ? 'postApiKeysByKeyIdEnable' : 'postApiKeysByKeyIdDisable'
    await request(operationId, {
      pathParameters: { keyId: id },
      body: {},
      headers: {
        ...idempotencyHeaders(),
        [MANAGEMENT_API_HEADERS.ifMatch]: etag('key', current.revision ?? 0),
      },
    })
    return inferenceAccessApi.key(id)
  },
  deleteKey: async (id: string) => {
    const current = await inferenceAccessApi.key(id)
    await request('deleteApiKeysByKeyId', {
      pathParameters: { keyId: id },
      headers: { [MANAGEMENT_API_HEADERS.ifMatch]: etag('key', current.revision ?? 0) },
    })
    return { deleted: true }
  },
  groups: async (params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementAccessPolicy>>('getAccessPolicies', {
      query: listQuery(params),
    })
    return viewPage(page, mapGroup)
  },
  group: async (id: string) =>
    mapGroup(
      resource(
        await request<ResourceDetail<ManagementAccessPolicy>>('getAccessPoliciesByPolicyId', {
          pathParameters: { policyId: id },
        }),
      ),
    ),
  saveGroup: async (item: Partial<AccessGroup>) => {
    const body = {
      name: item.name,
      description: item.description,
      status: item.status ?? 'active',
      grants: accessGrants(item.resources ?? []),
    }
    if (!item.id) {
      return mutateAndRead(
        'postAccessPolicies',
        undefined,
        body,
        inferenceAccessApi.group,
        idempotencyHeaders(),
      )
    }
    return mutateAndRead(
      'patchAccessPoliciesByPolicyId',
      { policyId: item.id },
      body,
      inferenceAccessApi.group,
      { [MANAGEMENT_API_HEADERS.ifMatch]: etag('access-policy', item.revision ?? 0) },
    )
  },
  deleteGroup: async (id: string) => {
    const current = await inferenceAccessApi.group(id)
    await request('deleteAccessPoliciesByPolicyId', {
      pathParameters: { policyId: id },
      headers: {
        [MANAGEMENT_API_HEADERS.ifMatch]: etag('access-policy', current.revision ?? 0),
      },
    })
  },
  budgets: async (params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementRateLimitPolicy>>('getRateLimitPolicies', {
      query: listQuery(params),
    })
    return viewPage(page, mapBudget)
  },
  budget: async (id: string) =>
    mapBudget(
      resource(
        await request<ResourceDetail<ManagementRateLimitPolicy>>('getRateLimitPoliciesByPolicyId', {
          pathParameters: { policyId: id },
        }),
      ),
    ),
  saveBudget: async (item: Partial<AccessBudget>) => {
    const body = {
      name: item.name,
      description: item.description,
      status: item.enabled === false ? 'disabled' : 'active',
      rules: (item.rules ?? []).map(rateLimitRuleInput),
    }
    if (!item.id) {
      return mutateAndRead(
        'postRateLimitPolicies',
        undefined,
        body,
        inferenceAccessApi.budget,
        idempotencyHeaders(),
      )
    }
    return mutateAndRead(
      'patchRateLimitPoliciesByPolicyId',
      { policyId: item.id },
      body,
      inferenceAccessApi.budget,
      { [MANAGEMENT_API_HEADERS.ifMatch]: etag('rate-limit-policy', item.revision ?? 0) },
    )
  },
  deleteBudget: async (id: string) => {
    const current = await inferenceAccessApi.budget(id)
    await request('deleteRateLimitPoliciesByPolicyId', {
      pathParameters: { policyId: id },
      headers: {
        [MANAGEMENT_API_HEADERS.ifMatch]: etag('rate-limit-policy', current.revision ?? 0),
      },
    })
  },
  usage,
  keyUsage,
  userUsage,
  teamUsage,
  requestLogs: async (filter: UsageFilter = {}) => {
    const page = await request<ManagementPage<ManagementRequestLog>>('getRequestLogs', {
      query: usageQuery(filter),
    })
    return viewPage(page, mapRequestLog, filter.q)
  },
  requestLog: async (id: string) => {
    if (!id.includes(':'))
      throw new Error('Request log identity must include namespace and admission id.')
    const [namespaceID, admissionID] = id.split(':', 2)
    const detail = await request<ResourceDetail<ManagementRequestLogDetail>>(
      'getNamespacesByNamespaceIdRequestLogsByAdmissionId',
      { pathParameters: { namespaceId: namespaceID, admissionId: admissionID } },
    )
    return mapRequestLog(detail.data.request)
  },
  auditLogs: async (filter: AccessListParams = {}) => {
    const page = await request<ManagementPage<AccessAuditEvent>>('getAuditEvents', {
      query: listQuery(filter),
    })
    return viewPage(page, (item) => item, filter.q)
  },
  selfTeams: async (): Promise<SelfTeamCatalog> => {
    const identity = assertManagementMe(
      await managementOperationRequest('getMe', { namespace: null }),
    )
    const selectedNamespace = getManagementNamespace()
    const scope =
      identity.namespaces.find((item) => item.namespace.namespaceId === selectedNamespace) ??
      identity.namespaces[0]
    if (!scope?.user) {
      return { items: [], members: [], accessGroups: [], budgets: [] }
    }
    const user = scope.user
    const member = {
      id: user.userId,
      email: user.email,
      name: user.displayName,
      status: user.status === 'active' ? ('active' as const) : ('disabled' as const),
    }
    return {
      items: scope.teams.map((team) => ({
        id: team.teamId,
        name: team.name,
        description: '',
        status: team.status === 'active' ? 'active' : 'disabled',
        members: [{ teamId: team.teamId, userId: user.userId, role: team.role }],
        accessGroupIds: [],
        budgetId: '',
      })),
      members: [member],
      accessGroups: [],
      budgets: [],
    }
  },
  selfTeam: async (id: string) => {
    const catalog = await inferenceAccessApi.selfTeams()
    const team = catalog.items.find((item) => item.id === id)
    if (!team) throw new Error('Team is not visible to this user.')
    return team
  },
  saveSelfTeam: (item: Partial<AccessTeam> & { id: string }) => inferenceAccessApi.saveTeam(item),
  selfKeys: async () => {
    const page = await request<ManagementPage<SelfInferenceKey>>('getSelfInferenceKeys', {
      query: new URLSearchParams({ pageSize: '200' }),
    })
    return viewPage(page, (item) => ({
      id: item.keyId,
      name: item.name,
      prefix: '',
      contextTeamId: item.contextTeamId,
      ownerType: item.owner.type,
      ownerId: item.owner.id,
      status: 'active' as const,
      expiresAt: item.expiresAt,
      accessGroupIds: [],
    }))
  },
  selfKey: keyDetail,
  selfKeySecret: async (id: string) => inferenceAccessApi.keySecret(id),
  createSelfKey: (
    name: string,
    ownerType: 'user' | 'team',
    ownerId: string,
    contextTeamId?: string,
  ) =>
    inferenceAccessApi.createKey({
      name,
      ownerType,
      ownerId,
      contextTeamId,
      accessGroupIds: [],
      revision: 0,
    }),
  rotateSelfKey: (id: string) => inferenceAccessApi.rotateKey(id),
  setSelfKeyStatus: (id: string, status: AccessStatus) =>
    inferenceAccessApi.setKeyStatus(id, status),
  deleteSelfKey: (id: string) => inferenceAccessApi.deleteKey(id),
  selfUsage: usage,
  selfKeyUsage: keyUsage,
  selfRequestLogs: (filter: UsageFilter = {}) => inferenceAccessApi.requestLogs(filter),
  selfRequestLog: (id: string) => inferenceAccessApi.requestLog(id),
}
