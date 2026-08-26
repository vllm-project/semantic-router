import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'
import {
  etag,
  idempotencyHeaders,
  listQuery,
  mutateAndRead,
  request,
  resource,
  viewPage,
} from './inferenceAccessTransport'
import { rateLimitRuleInput, subjectAssignmentPage } from './inferenceAccessPolicySupport'
import type {
  AccessPolicyGrant,
  ManagementAccessPolicy,
  ManagementAccessPolicyBinding,
  ManagementPage,
  ManagementRateLimitBinding,
  ManagementRateLimitPolicy,
  ResourceDetail,
  SubjectType,
} from './routerManagementTypes'
import type {
  AccessBudget,
  AccessGroup,
  AccessListParams,
  AccessResourceRef,
} from './inferenceAccessTypes'

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

async function group(id: string) {
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
}

async function budget(id: string) {
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
}

export const policyAccessApi = {
  groups: async (params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementAccessPolicy>>('getAccessPolicies', {
      query: listQuery(params),
    })
    return viewPage(page, mapGroup)
  },
  group,
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
      return mutateAndRead('postAccessPolicies', undefined, body, group, idempotencyHeaders())
    }
    return mutateAndRead('patchAccessPoliciesByPolicyId', { policyId: item.id }, body, group, {
      [MANAGEMENT_API_HEADERS.ifMatch]: etag('access-policy', item.revision ?? 0),
    })
  },
  deleteGroup: async (id: string) => {
    const current = await group(id)
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
  budget,
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
      return mutateAndRead('postRateLimitPolicies', undefined, body, budget, idempotencyHeaders())
    }
    return mutateAndRead('patchRateLimitPoliciesByPolicyId', { policyId: item.id }, body, budget, {
      [MANAGEMENT_API_HEADERS.ifMatch]: etag('rate-limit-policy', item.revision ?? 0),
    })
  },
  deleteBudget: async (id: string) => {
    const current = await budget(id)
    await request('deleteRateLimitPoliciesByPolicyId', {
      pathParameters: { policyId: id },
      headers: {
        [MANAGEMENT_API_HEADERS.ifMatch]: etag('rate-limit-policy', current.revision ?? 0),
      },
    })
  },
}
