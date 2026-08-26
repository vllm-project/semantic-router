import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'
import {
  etag,
  idempotencyHeaders,
  listQuery,
  query,
  request,
  resource,
  viewPage,
} from './inferenceAccessTransport'
import {
  rateLimitRuleInput,
  relationshipParams,
  subjectBindings,
  syncSubjectBindings,
} from './inferenceAccessPolicySupport'
import type {
  IssuedAPIKeySecret,
  ManagementAPIKey,
  ManagementCredential,
  ManagementEffectivePolicy,
  ManagementEffectiveQuota,
  ManagementPage,
  ResourceDetail,
} from './routerManagementTypes'
import type {
  APIKeyQuotaSnapshot,
  AccessAPIKey,
  AccessListParams,
  AccessStatus,
  CreatedAccessAPIKey,
  InlineRateLimitPolicyDraft,
  QuotaMeter,
} from './inferenceAccessTypes'

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

async function key(id: string): Promise<AccessAPIKey> {
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
  const mappedKey = mapKey(
    detail.data,
    credentials.data.find((credential) => credential.status === 'active') ?? credentials.data[0],
  )
  mappedKey.accessGroupIds = bindings.accessGroupIds
  mappedKey.budgetId = bindings.budgetId
  const policyData = 'data' in policy ? policy.data : policy
  const effectiveGrants = policyData.access.grants.filter(
    (grant) =>
      grant.effect === 'allow' &&
      grant.permissions.includes('invoke') &&
      (grant.resourceType === 'entrypoint' || grant.resourceType === 'model'),
  )
  mappedKey.effectiveAccess = effectiveGrants.map((grant) => ({
    resourceType: grant.resourceType,
    resourceId: grant.resourceId,
  }))
  mappedKey.accessPolicySources = [
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
  mappedKey.quota = quotaSnapshot(policyData.quota)
  mappedKey.effectiveBudgetId = mappedKey.quota?.budgetId
  mappedKey.budgetPolicySource = mappedKey.quota?.source
  return mappedKey
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

async function keySecret(id: string) {
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
}

async function createKey(
  item: Partial<AccessAPIKey>,
  inlineRateLimit?: InlineRateLimitPolicyDraft,
) {
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
}

export const credentialAccessApi = {
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
  key,
  keySummary: async (id: string) =>
    mapKey(
      resource(
        await request<ResourceDetail<ManagementAPIKey>>('getApiKeysByKeyId', {
          pathParameters: { keyId: id },
        }),
      ),
    ),
  keySecret,
  createKey,
  saveKey: async (item: Partial<AccessAPIKey> & { id: string }) => {
    await request<ResourceDetail<ManagementAPIKey>>('patchApiKeysByKeyId', {
      pathParameters: { keyId: item.id },
      body: { name: item.name },
      headers: { [MANAGEMENT_API_HEADERS.ifMatch]: etag('key', item.revision ?? 0) },
    })
    await syncSubjectBindings('api_key', item.id, item.accessGroupIds, item.budgetId)
    return key(item.id)
  },
  rotateKey: async (id: string) => {
    const current = await key(id)
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
    const current = await key(id)
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
    return key(id)
  },
  deleteKey: async (id: string) => {
    const current = await key(id)
    await request('deleteApiKeysByKeyId', {
      pathParameters: { keyId: id },
      headers: { [MANAGEMENT_API_HEADERS.ifMatch]: etag('key', current.revision ?? 0) },
    })
    return { deleted: true }
  },
}
