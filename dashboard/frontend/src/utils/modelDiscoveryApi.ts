export interface DiscoveredProviderModel {
  id: string
  ownedBy?: string
}

export interface ModelDiscoveryInput {
  baseUrl: string
  apiKey?: string
  authHeader?: string
  authPrefix?: string
  extraHeaders?: Record<string, string>
}

export class ModelDiscoveryApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ModelDiscoveryApiError'
    this.status = status
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

export async function discoverProviderModels(
  input: ModelDiscoveryInput,
  signal?: AbortSignal,
): Promise<DiscoveredProviderModel[]> {
  const response = await fetch('/api/models/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
    signal,
  })
  const payload: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const message =
      isRecord(payload) && typeof payload.message === 'string'
        ? payload.message
        : `Models could not be loaded (HTTP ${response.status}).`
    throw new ModelDiscoveryApiError(message, response.status)
  }
  if (!isRecord(payload) || !Array.isArray(payload.models)) {
    throw new ModelDiscoveryApiError('This connection returned an invalid model list.', 502)
  }
  return payload.models
    .filter(
      (model): model is Record<string, unknown> =>
        isRecord(model) && typeof model.id === 'string' && Boolean(model.id.trim()),
    )
    .map((model) => ({
      id: String(model.id).trim(),
      ownedBy: typeof model.ownedBy === 'string' ? model.ownedBy.trim() : undefined,
    }))
}
