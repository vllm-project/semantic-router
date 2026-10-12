import type { CanonicalModelDeployment } from './configPageSupport'
import type { RouterConfig } from './dashboardPageTypes'
import { decisionRuntimeDeclarations } from './decisionRuntimeDeployment'

export type ReplicaPlacement = NonNullable<CanonicalModelDeployment['replicas']>[number]

export function deploymentPlacements(deployment: CanonicalModelDeployment): ReplicaPlacement[] {
  if (deployment.replicas?.length) return deployment.replicas.map((replica) => ({ ...replica }))
  return deployment.endpoint
    ? [{ endpoint: deployment.endpoint, served_name: deployment.served_name }]
    : [{ device: deployment.device || 'auto' }]
}

// Replica edits preserve the logical model, bindings and public listener grants.
export function withDecisionReplicas(
  config: RouterConfig,
  name: string,
  replicas: ReplicaPlacement[],
): RouterConfig {
  const deployment = decisionRuntimeDeclarations(config)[name]
  if (!deployment || deployment.provider !== 'model_runtime')
    throw new Error('This runtime deployment is no longer configured. Refresh before scaling.')
  if (!replicas.length) throw new Error('Keep at least one replica.')
  if (replicas.length > 64) throw new Error('A deployment supports at most 64 replicas.')
  const placements = replicas.map((replica) => {
    if (replica.endpoint !== undefined) {
      const endpoint = replica.endpoint.trim()
      if (!endpoint) throw new Error('Each attached replica needs an endpoint.')
      let url: URL
      try {
        url = new URL(endpoint)
      } catch {
        throw new Error('Enter a valid runtime endpoint.')
      }
      if (
        !['http:', 'https:', 'unix:'].includes(url.protocol) ||
        url.username ||
        url.password ||
        url.search ||
        url.hash
      )
        throw new Error(
          'Use an HTTP, HTTPS or Unix runtime endpoint without credentials or a query.',
        )
      return {
        endpoint,
        ...(replica.served_name?.trim() ? { served_name: replica.served_name.trim() } : {}),
      }
    }
    const device = replica.device?.trim() || 'auto'
    if (!/^[a-z][a-z0-9_]*(?::[0-9]+)?$/.test(device))
      throw new Error('Enter a device such as auto, cpu, rocm:0 or cuda:0.')
    return { device }
  })
  const next = structuredClone(config)
  const target = decisionRuntimeDeclarations(next)[name]
  delete target.device
  delete target.endpoint
  delete target.served_name
  target.replicas = placements
  return next
}
