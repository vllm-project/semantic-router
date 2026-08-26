import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'
import { etag, idempotencyHeaders, query, request, viewPage } from './inferenceAccessTransport'
import type {
  ManagementAccessPolicyBinding,
  ManagementPage,
  ManagementRateLimitBinding,
  RateLimitRule,
  RateLimitRuleWrite,
  SubjectType,
} from './routerManagementTypes'
import type { AccessAssignment, AccessListParams } from './inferenceAccessTypes'

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

export function rateLimitRuleInput(rule: RateLimitRule): RateLimitRuleWrite {
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

export async function subjectBindings(type: SubjectType, id: string) {
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

export async function syncSubjectBindings(
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

export const relationshipParams = (params: AccessListParams = {}) => ({
  ...params,
  limit: Math.min(params.limit ?? 12, 200),
  includeTotal: params.includeTotal ?? !params.cursor,
})

export async function subjectAssignmentPage<
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
