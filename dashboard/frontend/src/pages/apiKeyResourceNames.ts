import type { AccessResourceRef } from '../utils/inferenceAccessApi'
import type { KeyScopedRoutingCatalog } from '../utils/keyScopedRoutingCatalog'

export type APIKeyResourceResolution =
  | { status: 'ready'; callableName: string }
  | { status: 'unavailable' }

export type APIKeyResourceResolutions = Record<string, APIKeyResourceResolution>

export function apiKeyResourceResolutionKey(resource: AccessResourceRef) {
  return `${resource.resourceType}:${resource.resourceId}`
}

export function apiKeyResourceResolutions(
  resources: AccessResourceRef[],
  catalog: KeyScopedRoutingCatalog | null,
): APIKeyResourceResolutions {
  const models = new Map<string, string>(
    catalog?.models.map((model) => [model.id, model.name.trim()]) ?? [],
  )
  const entrypoints = new Map<string, string>(
    catalog?.entrypoints.map((entrypoint) => [
      entrypoint.id,
      entrypoint.aliases.find((alias) => alias.trim())?.trim() ?? '',
    ]) ?? [],
  )
  return resources.reduce<APIKeyResourceResolutions>((resolutions, resource) => {
    const key = apiKeyResourceResolutionKey(resource)
    const callableName =
      resource.resourceType === 'model'
        ? models.get(resource.resourceId)
        : entrypoints.get(resource.resourceId)
    resolutions[key] = callableName ? { status: 'ready', callableName } : { status: 'unavailable' }
    return resolutions
  }, {})
}

export function apiKeyVisibleResourceName(
  resource: AccessResourceRef,
  resolutions: APIKeyResourceResolutions,
) {
  const resolution = resolutions[apiKeyResourceResolutionKey(resource)]
  if (resolution?.status === 'ready') return resolution.callableName
  if (!resolution) return 'Loading…'
  return resource.resourceType === 'entrypoint'
    ? 'Unavailable Mixture-of-Model'
    : 'Unavailable model'
}

export function apiKeyVisibleResourceNames(
  resources: AccessResourceRef[],
  resolutions: APIKeyResourceResolutions,
) {
  return Array.from(
    new Set(resources.map((resource) => apiKeyVisibleResourceName(resource, resolutions))),
  )
}

export function apiKeyQuickstartModel(
  resources: AccessResourceRef[],
  resolutions: APIKeyResourceResolutions,
) {
  const callableName = (resource: AccessResourceRef) => {
    const resolution = resolutions[apiKeyResourceResolutionKey(resource)]
    return resolution?.status === 'ready' ? resolution.callableName : ''
  }
  const entrypoint = resources.find(
    (resource) => resource.resourceType === 'entrypoint' && callableName(resource),
  )
  return (
    (entrypoint && callableName(entrypoint)) || resources.map(callableName).find(Boolean) || null
  )
}
