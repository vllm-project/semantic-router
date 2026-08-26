import type { AccessAssignment, AccessResourceRef } from '../utils/inferenceAccessApi'

export type EntityPolicyKind = 'access' | 'budget'

export interface EntityPolicyNameResolver {
  resolve: (kind: EntityPolicyKind, policyId: string) => Promise<string>
}

interface PendingPolicyName {
  kind: EntityPolicyKind
  policyId: string
  resolve: (name: string) => void
}

export const ENTITY_POLICY_NAME_CONCURRENCY = 4

export function createEntityPolicyNameResolver(
  loadName: (kind: EntityPolicyKind, policyId: string) => Promise<string>,
  maximumConcurrency = ENTITY_POLICY_NAME_CONCURRENCY,
): EntityPolicyNameResolver {
  const cache = new Map<string, Promise<string>>()
  const pending: PendingPolicyName[] = []
  const concurrency = Math.max(1, Math.floor(maximumConcurrency))
  let active = 0

  const drain = () => {
    while (active < concurrency && pending.length) {
      const next = pending.shift()
      if (!next) return
      active += 1
      void (async () => {
        try {
          const name = (await loadName(next.kind, next.policyId)).trim()
          next.resolve(name || next.policyId)
        } catch {
          next.resolve(next.policyId)
        } finally {
          active -= 1
          drain()
        }
      })()
    }
  }

  return {
    resolve(kind, policyId) {
      const cacheKey = `${kind}\u0000${policyId}`
      const cached = cache.get(cacheKey)
      if (cached) return cached
      const request = new Promise<string>((resolve) => {
        pending.push({ kind, policyId, resolve })
        drain()
      })
      cache.set(cacheKey, request)
      return request
    },
  }
}

export async function resolveEntityPolicyNames(
  kind: EntityPolicyKind,
  policyIds: string[],
  resolver: EntityPolicyNameResolver,
): Promise<ReadonlyMap<string, string>> {
  const uniquePolicyIds = [...new Set(policyIds)]
  const entries = await Promise.all(
    uniquePolicyIds.map(
      async (policyId) => [policyId, await resolver.resolve(kind, policyId)] as const,
    ),
  )
  return new Map(entries)
}

export function formatEntityPolicyNames(
  assignments: AccessAssignment[],
  names: ReadonlyMap<string, string>,
) {
  return assignments
    .map((assignment) => names.get(assignment.policyId) ?? assignment.policyId)
    .join(', ')
}

export function AccessGroupResourceTags({
  resources,
  resourceName,
}: {
  resources: AccessResourceRef[]
  resourceName: (resourceType: AccessResourceRef['resourceType'], resourceId: string) => string
}) {
  return (
    <>
      {resources.map((resource) => (
        <code key={`${resource.resourceType}:${resource.resourceId}`}>
          {resourceName(resource.resourceType, resource.resourceId)}
        </code>
      ))}
    </>
  )
}
