export const CANONICAL_AUTO_MODEL = 'vllm-sr/auto'

interface RouterModelRecord {
  id?: unknown
  owned_by?: unknown
  description?: unknown
}

interface RouterModelsResponse {
  data?: unknown
}

function normalizeModelRecords(payload: unknown): RouterModelRecord[] {
  if (!payload || typeof payload !== 'object') {
    return []
  }

  const { data } = payload as RouterModelsResponse
  if (!Array.isArray(data)) {
    return []
  }

  return data.filter((entry): entry is RouterModelRecord => Boolean(entry && typeof entry === 'object'))
}

function modelId(entry: RouterModelRecord): string {
  return typeof entry.id === 'string' ? entry.id.trim() : ''
}

function isAutomaticRouterModel(entry: RouterModelRecord): boolean {
  const id = modelId(entry)
  if (!id) {
    return false
  }

  const description = typeof entry.description === 'string' ? entry.description.toLowerCase() : ''

  return id === 'auto'
    || id === 'MoM'
    || id.toLowerCase().includes('/auto')
    || description.includes('automatic model routing')
    || description.includes('intelligent router for mixture-of-models')
}

export function selectRouterAutoModel(payload: unknown): string | null {
  const records = normalizeModelRecords(payload)
  const canonical = records.find((entry) => modelId(entry) === CANONICAL_AUTO_MODEL)
  if (canonical) {
    return CANONICAL_AUTO_MODEL
  }

  const automatic = records.find(isAutomaticRouterModel)
  return automatic ? modelId(automatic) : null
}

export function getRouterModelsEndpoint(chatCompletionsEndpoint: string): string {
  const marker = '/v1/chat/completions'
  const markerIndex = chatCompletionsEndpoint.indexOf(marker)

  if (markerIndex === -1) {
    return '/api/router/v1/models'
  }

  return `${chatCompletionsEndpoint.slice(0, markerIndex)}${marker.replace('/chat/completions', '/models')}`
}
