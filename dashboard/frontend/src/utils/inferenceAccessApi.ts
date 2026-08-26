import {
  assertManagementMe,
  getManagementNamespace,
  managementOperationRequest,
} from './managementApiContract'
import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'
import {
  allPages,
  etag,
  idempotencyHeaders,
  listQuery,
  mutateAndRead,
  query,
  request,
  resource,
  viewPage,
} from './inferenceAccessTransport'
import {
  keyUsage,
  overview,
  requestLog,
  requestLogs,
  teamUsage,
  usage,
  userUsage,
} from './inferenceAccessUsageApi'
import type {
  AccessPolicyGrant,
  IssuedAPIKeySecret,
  ManagementAPIKey,
  ManagementAccessPolicy,
  ManagementAccessPolicyBinding,
  ManagementCredential,
  ManagementEffectiveQuota,
  ManagementEffectivePolicy,
  ManagementMembership,
  ManagementPage,
  ManagementRateLimitBinding,
  ManagementRateLimitPolicy,
  ManagementTeam,
  ManagementUser,
  RateLimitRule,
  RateLimitRuleWrite,
  ResourceDetail,
  SelfInferenceKey,
  SubjectType,
} from './routerManagementTypes'
import type {
  APIKeyQuotaSnapshot,
  AccessAssignment,
  AccessAPIKey,
  AccessAuditEvent,
  AccessBudget,
  AccessGroup,
  AccessListParams,
  AccessResourceRef,
  AccessStatus,
  AccessTeam,
  AccessUser,
  CreatedAccessAPIKey,
  InlineRateLimitPolicyDraft,
  QuotaMeter,
  SelfTeamCatalog,
  TeamMembership,
  UsageFilter,
} from './inferenceAccessTypes'

export type * from './inferenceAccessTypes'

function mapMembership(item: ManagementMembership): TeamMembership {
  if (item.role !== 'admin' && item.role !== 'member') {
    throw new Error('Router returned an unsupported team membership role.')
  }
  const relation = item as ManagementMembership & {
    teamName?: string
    teamStatus?: AccessStatus
    displayName?: string
    email?: string
    userStatus?: AccessStatus
  }
  return {
    teamId: item.teamId,
    userId: item.userId,
    role: item.role,
    revision: item.revision,
    teamName: relation.teamName,
    teamStatus: relation.teamStatus,
    userName: relation.displayName,
    userEmail: relation.email,
    userStatus: relation.userStatus,
  }
}

function mapAccessAssignment(
  item: ManagementAccessPolicyBinding | ManagementRateLimitBinding,
): AccessAssignment {
  return {
    id: item.bindingId,
    policyId: item.policyId,
    subjectType: item.subject.type,
    subjectId: item.subject.id,
    status: item.status === 'active' ? 'active' : 'disabled',
    mode: 'mode' in item ? item.mode : undefined,
    revision: item.revision,
    createdAt: item.createdAt,
  }
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

function mapGroup(item: ManagementAccessPolicy, assignmentCount = 0): AccessGroup {
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
    assignmentCount,
    revision: item.revision,
    status: item.status === 'active' ? 'active' : 'disabled',
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
  }
}

function mapBudget(item: ManagementRateLimitPolicy, assignmentCount = 0): AccessBudget {
  return {
    id: item.policyId,
    name: item.name,
    description: item.description,
    rules: item.rules,
    enabled: item.status === 'active',
    assignmentCount,
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

function rateLimitRuleInput(rule: RateLimitRule): RateLimitRuleWrite {
  const input: Partial<RateLimitRule> = { ...rule }
  delete input.ordinal
  delete input.ruleId
  return input as RateLimitRuleWrite
}

async function allSubjectBindings<T>(
  operationId: 'getAccessPolicyBindings' | 'getRateLimitBindings',
  type: SubjectType,
  id: string,
): Promise<T[]> {
  const items: T[] = []
  let cursor: string | undefined
  do {
    const page = await request<ManagementPage<T>>(operationId, {
      query: query({
        subjectType: type,
        subjectId: id,
        status: 'active',
        pageSize: 200,
        cursor,
      }),
    })
    items.push(...page.data)
    cursor = page.page.hasMore ? page.page.nextCursor : undefined
  } while (cursor)
  return items
}

async function subjectBindings(type: SubjectType, id: string) {
  const [access, rate] = await Promise.all([
    allSubjectBindings<ManagementAccessPolicyBinding>('getAccessPolicyBindings', type, id),
    allSubjectBindings<ManagementRateLimitBinding>('getRateLimitBindings', type, id),
  ])
  return {
    accessGroupIds: access.map((binding) => binding.policyId),
    budgetId: rate.find((binding) => binding.mode === 'allocation')?.policyId,
    accessBindings: access,
    rateBindings: rate,
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

const relationshipParams = (params: AccessListParams = {}) => ({
  ...params,
  limit: Math.min(params.limit ?? 12, 200),
  includeTotal: params.includeTotal ?? !params.cursor,
})

async function subjectAssignmentPage<
  T extends ManagementAccessPolicyBinding | ManagementRateLimitBinding,
>(
  operationId: 'getAccessPolicyBindings' | 'getRateLimitBindings',
  filter: { subjectType?: SubjectType; subjectId?: string; policyId?: string },
  params: AccessListParams = {},
) {
  const normalized = relationshipParams(params)
  const page = await request<ManagementPage<T>>(operationId, {
    query: query({
      ...filter,
      cursor: normalized.cursor,
      pageSize: normalized.limit,
      status: normalized.status,
      includeTotal: normalized.includeTotal ? 'true' : undefined,
    }),
  })
  return viewPage(page, mapAccessAssignment)
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
      request<ManagementPage<ManagementMembership>>('getUsersByUserIdMemberships', {
        pathParameters: { userId: id },
        query: listQuery({ limit: 12, includeTotal: true }),
      })
        .then((page) => page.data)
        .catch(() => []),
      subjectBindings('user', id).catch(() => ({ accessGroupIds: [], budgetId: undefined })),
    ])
    return mapUser(detail.data, memberships, bindings.accessGroupIds, bindings.budgetId)
  },
  userMemberships: async (id: string, params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementMembership>>(
      'getUsersByUserIdMemberships',
      {
        pathParameters: { userId: id },
        query: listQuery(relationshipParams(params)),
      },
    )
    return viewPage(page, mapMembership)
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
      request<ManagementPage<ManagementMembership>>('getTeamsByTeamIdMembers', {
        pathParameters: { teamId: id },
        query: listQuery({ limit: 12, includeTotal: true }),
      })
        .then((page) => page.data)
        .catch(() => []),
      subjectBindings('team', id).catch(() => ({ accessGroupIds: [], budgetId: undefined })),
    ])
    return mapTeam(detail.data, members, bindings.accessGroupIds, bindings.budgetId)
  },
  teamMembers: async (id: string, params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementMembership>>('getTeamsByTeamIdMembers', {
      pathParameters: { teamId: id },
      query: listQuery(relationshipParams(params)),
    })
    return viewPage(page, mapMembership)
  },
  teamForEdit: async (id: string) => {
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
  ownedKeys: async (ownerType: 'user' | 'team', ownerId: string, params: AccessListParams = {}) => {
    const normalized = relationshipParams(params)
    const page = await request<ManagementPage<ManagementAPIKey>>('getApiKeys', {
      query: query({
        ownerType,
        ownerId,
        cursor: normalized.cursor,
        pageSize: normalized.limit,
        status: normalized.status,
        includeTotal: normalized.includeTotal ? 'true' : undefined,
      }),
    })
    return viewPage(page, mapKey)
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
  group: async (id: string) => {
    const [detail, assignments] = await Promise.all([
      request<ResourceDetail<ManagementAccessPolicy>>('getAccessPoliciesByPolicyId', {
        pathParameters: { policyId: id },
      }),
      subjectAssignmentPage<ManagementAccessPolicyBinding>(
        'getAccessPolicyBindings',
        { policyId: id },
        { limit: 1 },
      ),
    ])
    return mapGroup(detail.data, assignments.total)
  },
  groupSummary: async (id: string) =>
    mapGroup(
      resource(
        await request<ResourceDetail<ManagementAccessPolicy>>('getAccessPoliciesByPolicyId', {
          pathParameters: { policyId: id },
        }),
      ),
    ),
  accessAssignments: (
    filter: { subjectType?: SubjectType; subjectId?: string; policyId?: string },
    params: AccessListParams = {},
  ) =>
    subjectAssignmentPage<ManagementAccessPolicyBinding>('getAccessPolicyBindings', filter, params),
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
  budget: async (id: string) => {
    const [detail, assignments] = await Promise.all([
      request<ResourceDetail<ManagementRateLimitPolicy>>('getRateLimitPoliciesByPolicyId', {
        pathParameters: { policyId: id },
      }),
      subjectAssignmentPage<ManagementRateLimitBinding>(
        'getRateLimitBindings',
        { policyId: id },
        { limit: 1 },
      ),
    ])
    return mapBudget(detail.data, assignments.total)
  },
  budgetSummary: async (id: string) =>
    mapBudget(
      resource(
        await request<ResourceDetail<ManagementRateLimitPolicy>>('getRateLimitPoliciesByPolicyId', {
          pathParameters: { policyId: id },
        }),
      ),
    ),
  budgetAssignments: (
    filter: { subjectType?: SubjectType; subjectId?: string; policyId?: string },
    params: AccessListParams = {},
  ) => subjectAssignmentPage<ManagementRateLimitBinding>('getRateLimitBindings', filter, params),
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
  requestLogs,
  requestLog,
  auditLogs: async (filter: AccessListParams = {}) => {
    const page = await request<ManagementPage<AccessAuditEvent>>('getAuditEvents', {
      // Audit exposes typed exact filters, not the generic collection search
      // contract. Keep free-text filtering in the product view until the
      // Router publishes a bounded audit-search selector.
      query: query({ cursor: filter.cursor, pageSize: filter.limit }),
    })
    return viewPage(page, (item) => item, filter.q?.trim())
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
  selfKeys: async (params: AccessListParams = {}) => {
    const page = await request<ManagementPage<SelfInferenceKey>>('getSelfInferenceKeys', {
      query: listQuery({ ...params, limit: Math.min(params.limit ?? 25, 200) }),
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
  selfKey: async (id: string) => {
    const detail = await request<ResourceDetail<SelfInferenceKey>>('getSelfInferenceKeysByKeyId', {
      pathParameters: { keyId: id },
    })
    return {
      id: detail.data.keyId,
      name: detail.data.name,
      prefix: '',
      contextTeamId: detail.data.contextTeamId,
      ownerType: detail.data.owner.type,
      ownerId: detail.data.owner.id,
      status: 'active' as const,
      expiresAt: detail.data.expiresAt,
      accessGroupIds: [],
    }
  },
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
