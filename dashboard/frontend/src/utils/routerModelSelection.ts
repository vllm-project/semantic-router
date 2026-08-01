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
}

interface RouterModelsResponse {
  data?: unknown
}

export interface RouterModelOption {
  id: string
  description: string
}

type RouterModelResolution = 'virtual' | 'passthrough'

interface RouterModelRoutingMetadata {
  resolution: RouterModelResolution
  selectable: boolean
  defaultRoute: boolean
}

const LEGACY_AUTO_MODEL_IDS = new Set(['auto', CANONICAL_AUTO_MODEL])

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
    } = entry.routing as RouterModelRoutingRecord
    if (
      (resolution !== 'virtual' && resolution !== 'passthrough') ||
      typeof selectable !== 'boolean' ||
      (defaultRoute !== undefined && typeof defaultRoute !== 'boolean')
    ) {
      return null
    }

    const isDefaultRoute = defaultRoute ?? false
    if (isDefaultRoute && (resolution !== 'virtual' || !selectable)) return null
    return { resolution, selectable, defaultRoute: isDefaultRoute }
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
  return Boolean(id) && Boolean(routing?.selectable && routing.defaultRoute)
}

function isSelectableRouterModel(entry: RouterModelRecord): boolean {
  const id = modelId(entry)
  return Boolean(id) && modelRoutingMetadata(entry)?.selectable === true
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
    }))
    .filter((model) => {
      if (seen.has(model.id)) return false
      seen.add(model.id)
      return true
    })
  const toOption = (model: (typeof models)[number]): RouterModelOption => ({
    id: model.id,
    description: model.description,
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
