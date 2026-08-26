export const CANONICAL_AUTO_MODEL = 'vllm-sr/auto'

interface RouterModelRecord {
  id?: unknown
  owned_by?: unknown
  description?: unknown
  routing?: unknown
}

interface RouterModelRoutingRecord {
  resolution?: unknown
  selectable?: unknown
  default_route?: unknown
  recipe?: unknown
}

interface RouterModelsResponse {
  data?: unknown
}

export interface RouterModelOption {
  id: string
  description: string
  kind?: 'individual'
  recipe?: string
}

export interface RouterModelListOptions {
  includeIndividualModels?: boolean
}

type RouterModelResolution = 'virtual' | 'passthrough'

interface RouterModelRoutingMetadata {
  resolution: RouterModelResolution
  selectable: boolean
  defaultRoute: boolean
  recipe?: string
}

const RETIRED_ROUTER_MODEL_IDS = new Set(['mom'])

function normalizeModelRecords(payload: unknown): RouterModelRecord[] {
  if (!payload || typeof payload !== 'object') {
    return []
  }

  const { data } = payload as RouterModelsResponse
  if (!Array.isArray(data)) {
    return []
  }

  return data.filter((entry): entry is RouterModelRecord =>
    Boolean(entry && typeof entry === 'object'),
  )
}

function modelId(entry: RouterModelRecord): string {
  return typeof entry.id === 'string' ? entry.id.trim() : ''
}

function isRetiredRouterModel(entry: RouterModelRecord): boolean {
  return RETIRED_ROUTER_MODEL_IDS.has(modelId(entry).toLocaleLowerCase())
}

function modelRoutingMetadata(entry: RouterModelRecord): RouterModelRoutingMetadata | null {
  if (entry.routing !== undefined) {
    if (!entry.routing || typeof entry.routing !== 'object' || Array.isArray(entry.routing)) {
      return null
    }
    const {
      resolution,
      selectable,
      default_route: defaultRoute,
      recipe,
    } = entry.routing as RouterModelRoutingRecord
    if (
      (resolution !== 'virtual' && resolution !== 'passthrough') ||
      typeof selectable !== 'boolean' ||
      (defaultRoute !== undefined && typeof defaultRoute !== 'boolean') ||
      (recipe !== undefined && (typeof recipe !== 'string' || !recipe.trim()))
    ) {
      return null
    }

    const isDefaultRoute = defaultRoute ?? false
    if (isDefaultRoute && (resolution !== 'virtual' || !selectable)) return null
    return {
      resolution,
      selectable,
      defaultRoute: isDefaultRoute,
      recipe: typeof recipe === 'string' ? recipe.trim() : undefined,
    }
  }

  return null
}

function isAutomaticRouterModel(entry: RouterModelRecord): boolean {
  const id = modelId(entry)
  const routing = modelRoutingMetadata(entry)
  return (
    Boolean(id) &&
    !isRetiredRouterModel(entry) &&
    Boolean(routing?.selectable && routing.defaultRoute)
  )
}

function isVisibleRouterModel(entry: RouterModelRecord, includeIndividualModels: boolean): boolean {
  const id = modelId(entry)
  const routing = modelRoutingMetadata(entry)
  if (!id || isRetiredRouterModel(entry) || !routing) return false
  if (routing.resolution === 'passthrough') return includeIndividualModels
  return routing.selectable
}

export function selectRouterAutoModel(payload: unknown): string | null {
  const records = normalizeModelRecords(payload)
  const canonical = records.find(
    (entry) => modelId(entry) === CANONICAL_AUTO_MODEL && isAutomaticRouterModel(entry),
  )
  if (canonical) {
    return CANONICAL_AUTO_MODEL
  }

  const automatic = records.find(isAutomaticRouterModel)
  return automatic ? modelId(automatic) : null
}

export function listRouterModels(
  payload: unknown,
  { includeIndividualModels = false }: RouterModelListOptions = {},
): RouterModelOption[] {
  const seen = new Set<string>()
  const models = normalizeModelRecords(payload)
    .filter((entry) => isVisibleRouterModel(entry, includeIndividualModels))
    .map((entry) => ({
      id: modelId(entry),
      description: typeof entry.description === 'string' ? entry.description.trim() : '',
      defaultRoute: modelRoutingMetadata(entry)?.defaultRoute ?? false,
      kind:
        modelRoutingMetadata(entry)?.resolution === 'passthrough'
          ? ('individual' as const)
          : undefined,
      recipe: modelRoutingMetadata(entry)?.recipe,
    }))
    .filter((model) => {
      if (seen.has(model.id)) return false
      seen.add(model.id)
      return true
    })
  const toOption = (model: (typeof models)[number]): RouterModelOption => ({
    id: model.id,
    description: model.description,
    ...(model.kind ? { kind: model.kind } : {}),
    ...(model.recipe ? { recipe: model.recipe } : {}),
  })
  const explicitVirtualModels = models.filter((model) => !model.defaultRoute && !model.kind)
  const individualModels = models.filter((model) => model.kind === 'individual')
  if (explicitVirtualModels.length > 0) {
    return [...explicitVirtualModels, ...individualModels].map(toOption)
  }
  const automaticID = selectRouterAutoModel(payload)
  const automaticModel = models.find((model) => model.id === automaticID)
  return [...(automaticModel ? [automaticModel] : []), ...individualModels].map(toOption)
}

export function getRouterModelsEndpoint(chatCompletionsEndpoint: string): string {
  const marker = '/v1/chat/completions'
  const markerIndex = chatCompletionsEndpoint.indexOf(marker)

  if (markerIndex === -1) {
    return '/v1/models'
  }

  return `${chatCompletionsEndpoint.slice(0, markerIndex)}${marker.replace('/chat/completions', '/models')}`
}
