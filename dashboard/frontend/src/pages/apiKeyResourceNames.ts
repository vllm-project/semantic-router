import type { AccessResourceRef } from '../utils/inferenceAccessApi'
import type { KeyScopedRoutingCatalog } from '../utils/keyScopedRoutingCatalog'

export type APIKeyResourceResolution = { status: 'ready'; name: string } | { status: 'unavailable' }

export type APIKeyResourceResolutions = Record<string, APIKeyResourceResolution>

export const API_KEY_MODEL_PLACEHOLDER = 'YOUR_MODEL'

export function apiKeyResourceResolutionKey(resource: AccessResourceRef) {
  return `${resource.resourceType}:${resource.resourceId}`
}

export function apiKeyResourceResolutions(
  resources: AccessResourceRef[],
  catalog: KeyScopedRoutingCatalog | null,
): APIKeyResourceResolutions {
  const modelNames = new Map(catalog?.models.map((model) => [model.id, model.name]) ?? [])
  const entrypointNames = new Map(
    catalog?.entrypoints.map((entrypoint) => [entrypoint.id, entrypoint.name]) ?? [],
  )
  return resources.reduce<APIKeyResourceResolutions>((resolutions, resource) => {
    const key = apiKeyResourceResolutionKey(resource)
    const name =
      resource.resourceType === 'model'
        ? modelNames.get(resource.resourceId)
        : entrypointNames.get(resource.resourceId)
    resolutions[key] = name?.trim()
      ? { status: 'ready', name: name.trim() }
      : { status: 'unavailable' }
    return resolutions
  }, {})
}

export function apiKeyVisibleResourceName(
  resource: AccessResourceRef,
  resolutions: APIKeyResourceResolutions,
) {
  const resolution = resolutions[apiKeyResourceResolutionKey(resource)]
  if (resolution?.status === 'ready') return resolution.name
  if (!resolution) return 'Loading…'
  return resource.resourceType === 'entrypoint'
    ? 'Unavailable Mixture-of-Model'
    : 'Unavailable model'
}

export function apiKeyQuickstartModel(
  resources: AccessResourceRef[],
  resolutions: APIKeyResourceResolutions,
) {
  const readyName = (resource: AccessResourceRef) => {
    const resolution = resolutions[apiKeyResourceResolutionKey(resource)]
    return resolution?.status === 'ready' ? resolution.name : ''
  }
  const entrypoint = resources.find(
    (resource) => resource.resourceType === 'entrypoint' && readyName(resource),
  )
  return (
    (entrypoint && readyName(entrypoint)) ||
    resources.map(readyName).find(Boolean) ||
    API_KEY_MODEL_PLACEHOLDER
  )
}
