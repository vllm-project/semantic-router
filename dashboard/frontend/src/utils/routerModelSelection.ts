export const CANONICAL_AUTO_MODEL = 'vllm-sr/auto'

interface RouterModelRecord {
  id?: unknown
  owned_by?: unknown
  description?: unknown
  routing_type?: unknown
}

interface RouterModelsResponse {
  data?: unknown
}

export interface RouterModelOption {
  id: string
  description: string
}

type RouterModelRoutingType = 'auto_alias' | 'entrypoint' | 'looper' | 'backend'

const LEGACY_AUTO_MODEL_IDS = new Set(['auto', 'mom', CANONICAL_AUTO_MODEL])
const RETIRED_MOM_MODEL_IDS = new Set(['mom', 'vllm-sr/mom'])

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

function modelRoutingType(entry: RouterModelRecord): RouterModelRoutingType | null {
  switch (entry.routing_type) {
    case 'auto_alias':
    case 'entrypoint':
    case 'looper':
    case 'backend':
      return entry.routing_type
    default:
      if (entry.routing_type !== undefined) return null
  }

  // Older routers do not emit routing_type. Preserve only their standard auto
  // aliases; custom aliases and entrypoints require the explicit contract.
  const owner = typeof entry.owned_by === 'string' ? entry.owned_by.trim().toLowerCase() : ''
  if (owner !== 'vllm-semantic-router') return null
  const normalizedId = modelId(entry).toLowerCase()
  return LEGACY_AUTO_MODEL_IDS.has(normalizedId) ? 'auto_alias' : null
}

function isRetiredMomAlias(id: string): boolean {
  return RETIRED_MOM_MODEL_IDS.has(id.toLowerCase())
}

function isAutomaticRouterModel(entry: RouterModelRecord): boolean {
  const id = modelId(entry)
  return Boolean(id) && !isRetiredMomAlias(id) && modelRoutingType(entry) === 'auto_alias'
}

function isSelectableRouterModel(entry: RouterModelRecord): boolean {
  const id = modelId(entry)
  const routingType = modelRoutingType(entry)
  return (
    Boolean(id) &&
    !isRetiredMomAlias(id) &&
    (routingType === 'auto_alias' || routingType === 'entrypoint' || routingType === 'looper')
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
      routingType: modelRoutingType(entry),
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
  const explicitModels = models.filter((model) => model.routingType !== 'auto_alias')
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
