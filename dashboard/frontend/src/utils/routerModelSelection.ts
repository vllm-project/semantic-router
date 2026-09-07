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
  recipe?: string
}

type RouterModelResolution = 'virtual' | 'passthrough'

interface RouterModelRoutingMetadata {
  resolution: RouterModelResolution
  selectable: boolean
  defaultRoute: boolean
  recipe?: string
}

const LEGACY_AUTO_MODEL_IDS = new Set(['auto', CANONICAL_AUTO_MODEL])
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

  // Older routers do not emit routing metadata. Preserve only their standard
  // auto aliases; custom aliases require the explicit contract.
  const owner = typeof entry.owned_by === 'string' ? entry.owned_by.trim().toLowerCase() : ''
  if (owner !== 'vllm-semantic-router') return null
  const normalizedId = modelId(entry).toLowerCase()
  if (!LEGACY_AUTO_MODEL_IDS.has(normalizedId)) return null
  return { resolution: 'virtual', selectable: true, defaultRoute: true }
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

function isSelectableRouterModel(entry: RouterModelRecord): boolean {
  const id = modelId(entry)
  return (
    Boolean(id) && !isRetiredRouterModel(entry) && modelRoutingMetadata(entry)?.selectable === true
  )
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

export function listRouterModels(payload: unknown): RouterModelOption[] {
  const seen = new Set<string>()
  const models = normalizeModelRecords(payload)
    .filter(isSelectableRouterModel)
    .map((entry) => ({
      id: modelId(entry),
      description: typeof entry.description === 'string' ? entry.description.trim() : '',
      defaultRoute: modelRoutingMetadata(entry)?.defaultRoute ?? false,
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
    ...(model.recipe ? { recipe: model.recipe } : {}),
  })
  const explicitModels = models.filter((model) => !model.defaultRoute)
  if (explicitModels.length > 0) return explicitModels.map(toOption)

  const canonical = models.find((model) => model.id === CANONICAL_AUTO_MODEL)
  return canonical ? [toOption(canonical)] : models.slice(0, 1).map(toOption)
}

export function getRouterModelsEndpoint(chatCompletionsEndpoint: string): string {
  const marker = '/v1/chat/completions'
  const markerIndex = chatCompletionsEndpoint.indexOf(marker)

  if (markerIndex === -1) {
    return '/api/router/v1/models'
  }

  return `${chatCompletionsEndpoint.slice(0, markerIndex)}${marker.replace('/chat/completions', '/models')}`
}
