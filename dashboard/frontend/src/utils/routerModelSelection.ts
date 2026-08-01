export const CANONICAL_AUTO_MODEL = 'vllm-sr/auto'

interface RouterModelRecord {
  id?: unknown
  owned_by?: unknown
  description?: unknown
}

interface RouterModelsResponse {
  data?: unknown
}

export interface RouterModelOption {
  id: string
  description: string
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

function isRouterOwned(entry: RouterModelRecord): boolean {
  if (typeof entry.owned_by !== 'string') return false
  const owner = entry.owned_by.trim().toLowerCase()
  return owner === 'vllm-semantic-router' || owner === 'amd'
}

function isAutomaticRouterModel(entry: RouterModelRecord): boolean {
  const id = modelId(entry)
  const normalizedId = id.toLowerCase()
  if (!id || !isRouterOwned(entry) || normalizedId === 'mom' || normalizedId.endsWith('/mom')) {
    return false
  }

  const description = typeof entry.description === 'string' ? entry.description.toLowerCase() : ''

  return (
    normalizedId === 'auto' ||
    normalizedId.endsWith('/auto') ||
    description.includes('automatic model routing') ||
    description.includes('intelligent router for mixture-of-models')
  )
}

export function selectRouterAutoModel(payload: unknown): string | null {
  const records = normalizeModelRecords(payload)
  const canonical = records.find(
    (entry) => modelId(entry) === CANONICAL_AUTO_MODEL && isRouterOwned(entry),
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
    .filter(isRouterOwned)
    .map((entry) => ({
      id: modelId(entry),
      description: typeof entry.description === 'string' ? entry.description.trim() : '',
      automatic: isAutomaticRouterModel(entry),
    }))
    .filter((model) => {
      const normalizedId = model.id.toLowerCase()
      if (
        !model.id ||
        normalizedId === 'mom' ||
        normalizedId.endsWith('/mom') ||
        seen.has(model.id)
      )
        return false
      seen.add(model.id)
      return true
    })
  const toOption = (model: (typeof models)[number]): RouterModelOption => ({
    id: model.id,
    description: model.description,
  })
  const explicitModels = models.filter((model) => !model.automatic)
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
