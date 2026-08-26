import type { AccessResourceRef } from '../utils/inferenceAccessApi'
import type { KeyScopedRoutingCatalog } from '../utils/keyScopedRoutingCatalog'

export type APIKeyResourceResolution =
  | { status: 'ready'; name: string; requestModel?: string }
  | { status: 'unavailable' }

export type APIKeyResourceResolutions = Record<string, APIKeyResourceResolution>

type CatalogResourceName = { name: string; requestModel?: string }

export function apiKeyResourceResolutionKey(resource: AccessResourceRef) {
  return `${resource.resourceType}:${resource.resourceId}`
}

export function apiKeyResourceResolutions(
  resources: AccessResourceRef[],
  catalog: KeyScopedRoutingCatalog | null,
): APIKeyResourceResolutions {
  const models = new Map<string, CatalogResourceName>(
    catalog?.models.map(
      (model): [string, CatalogResourceName] => [
        model.id,
        { name: model.name, requestModel: model.name },
      ],
    ) ?? [],
  )
  const entrypoints = new Map<string, CatalogResourceName>(
    catalog?.entrypoints.map((entrypoint): [string, CatalogResourceName] => {
      const requestModel = entrypoint.aliases.find((alias) => alias.trim())?.trim()
      return [
        entrypoint.id,
        { name: entrypoint.name, ...(requestModel ? { requestModel } : {}) },
      ]
    }) ?? [],
  )
  return resources.reduce<APIKeyResourceResolutions>((resolutions, resource) => {
    const key = apiKeyResourceResolutionKey(resource)
    const resolved =
      resource.resourceType === 'model'
        ? models.get(resource.resourceId)
        : entrypoints.get(resource.resourceId)
    resolutions[key] = resolved?.name.trim()
      ? {
          status: 'ready',
          name: resolved.name.trim(),
          ...(resolved.requestModel?.trim()
            ? { requestModel: resolved.requestModel.trim() }
            : {}),
        }
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
  const requestModel = (resource: AccessResourceRef) => {
    const resolution = resolutions[apiKeyResourceResolutionKey(resource)]
    return resolution?.status === 'ready' ? (resolution.requestModel ?? '') : ''
  }
  const entrypoint = resources.find(
    (resource) => resource.resourceType === 'entrypoint' && requestModel(resource),
  )
  return (
    (entrypoint && requestModel(entrypoint)) ||
    resources.map(requestModel).find(Boolean) ||
    null
  )
}
