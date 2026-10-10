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
  api?: unknown
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
      api,
    } = entry.routing as RouterModelRoutingRecord
    if (
      (api !== undefined && api !== 'chat') ||
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
  return Boolean(id) && Boolean(routing?.selectable && routing.defaultRoute)
}

function isSelectableRouterModel(entry: RouterModelRecord): boolean {
  const id = modelId(entry)
  return Boolean(id) && modelRoutingMetadata(entry)?.selectable === true
}

export function selectRouterAutoModel(payload: unknown): string | null {
  const records = normalizeModelRecords(payload)
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
  return models.map(toOption)
}

export function getRouterModelsEndpoint(chatCompletionsEndpoint: string): string {
  const marker = '/v1/chat/completions'
  const markerIndex = chatCompletionsEndpoint.indexOf(marker)

  if (markerIndex === -1) {
    return '/api/router/v1/models'
  }

  return `${chatCompletionsEndpoint.slice(0, markerIndex)}${marker.replace('/chat/completions', '/models')}`
}
