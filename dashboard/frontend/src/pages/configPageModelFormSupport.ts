import type {
  BackendRefEntry,
  ConfigData,
  LoRAAdapter,
  ModelPricing,
} from './configPageSupport'

export function normalizeModelLoras(value: unknown): LoRAAdapter[] {
  if (!Array.isArray(value)) return []

  return value
    .filter((entry): entry is Record<string, unknown> => Boolean(entry) && typeof entry === 'object' && !Array.isArray(entry))
    .map((entry) => ({
      name: typeof entry.name === 'string' ? entry.name.trim() : '',
      description: typeof entry.description === 'string' && entry.description.trim()
        ? entry.description.trim()
        : undefined,
    }))
    .filter((entry) => entry.name)
}

export function normalizeModelBackendRefs(value: unknown): BackendRefEntry[] {
  if (!Array.isArray(value)) return []

  return value
    .filter((entry): entry is Record<string, unknown> => Boolean(entry) && typeof entry === 'object' && !Array.isArray(entry))
    .map((entry) => {
      const normalized: BackendRefEntry = {}
      if (typeof entry.name === 'string' && entry.name.trim()) normalized.name = entry.name.trim()
      if (typeof entry.backend_id === 'string' && entry.backend_id.trim()) normalized.backend_id = entry.backend_id.trim()
      if (typeof entry.engine_kind === 'string' && entry.engine_kind.trim()) normalized.engine_kind = entry.engine_kind.trim()
      if (typeof entry.endpoint === 'string' && entry.endpoint.trim()) normalized.endpoint = entry.endpoint.trim()
      if (entry.protocol === 'https') normalized.protocol = 'https'
      else if (entry.protocol === 'http') normalized.protocol = 'http'
      if (typeof entry.weight === 'number' && Number.isFinite(entry.weight)) normalized.weight = entry.weight
      if (typeof entry.type === 'string' && entry.type.trim()) normalized.type = entry.type.trim()
      if (typeof entry.base_url === 'string' && entry.base_url.trim()) normalized.base_url = entry.base_url.trim()
      if (typeof entry.provider === 'string' && entry.provider.trim()) normalized.provider = entry.provider.trim()
      if (typeof entry.auth_header === 'string' && entry.auth_header.trim()) normalized.auth_header = entry.auth_header.trim()
      if (typeof entry.auth_prefix === 'string' && entry.auth_prefix.trim()) normalized.auth_prefix = entry.auth_prefix.trim()
      if (entry.extra_headers && typeof entry.extra_headers === 'object' && !Array.isArray(entry.extra_headers)) {
        normalized.extra_headers = Object.fromEntries(
          Object.entries(entry.extra_headers as Record<string, unknown>)
            .filter(([, nestedValue]) => typeof nestedValue === 'string')
            .map(([key, nestedValue]) => [key, nestedValue as string]),
        )
      }
      if (typeof entry.api_version === 'string' && entry.api_version.trim()) normalized.api_version = entry.api_version.trim()
      if (typeof entry.chat_path === 'string' && entry.chat_path.trim()) normalized.chat_path = entry.chat_path.trim()
      if (typeof entry.api_key === 'string' && entry.api_key.trim()) normalized.api_key = entry.api_key.trim()
      if (typeof entry.api_key_env === 'string' && entry.api_key_env.trim()) normalized.api_key_env = entry.api_key_env.trim()
      return normalized
    })
}

export function normalizeModelStringMap(value: unknown): Record<string, string> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined

  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([key, item]) => key.trim() && typeof item === 'string' && item.trim())
    .map(([key, item]) => [key.trim(), (item as string).trim()])
  return entries.length > 0 ? Object.fromEntries(entries) : undefined
}

export function normalizeModelPricing(value: unknown): ModelPricing | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined

  const pricing = value as Record<string, unknown>
  const normalized: ModelPricing = {}
  if (typeof pricing.currency === 'string' && pricing.currency.trim()) normalized.currency = pricing.currency.trim()
  if (typeof pricing.prompt_per_1m === 'number' && Number.isFinite(pricing.prompt_per_1m)) normalized.prompt_per_1m = pricing.prompt_per_1m
  if (typeof pricing.cached_input_per_1m === 'number' && Number.isFinite(pricing.cached_input_per_1m)) {
    normalized.cached_input_per_1m = pricing.cached_input_per_1m
  }
  if (typeof pricing.cache_write_per_1m === 'number' && Number.isFinite(pricing.cache_write_per_1m)) {
    normalized.cache_write_per_1m = pricing.cache_write_per_1m
  }
  if (typeof pricing.completion_per_1m === 'number' && Number.isFinite(pricing.completion_per_1m)) normalized.completion_per_1m = pricing.completion_per_1m
  return Object.keys(normalized).length > 0 ? normalized : undefined
}

export function buildProviderModelPayload(
  name: string,
  data: Record<string, unknown>,
  existingModel?: NonNullable<NonNullable<ConfigData['providers']>['models']>[number],
) {
  return {
    name,
    reasoning_family:
      typeof data.reasoning_family === 'string' && data.reasoning_family.trim()
        ? data.reasoning_family.trim()
        : undefined,
    provider_model_id:
      typeof data.provider_model_id === 'string' && data.provider_model_id.trim()
        ? data.provider_model_id.trim()
        : existingModel?.provider_model_id || name,
    api_format:
      typeof data.api_format === 'string' && data.api_format.trim()
        ? data.api_format.trim()
        : undefined,
    external_model_ids: normalizeModelStringMap(data.external_model_ids),
    backend_refs: normalizeModelBackendRefs(data.backend_refs),
    pricing: normalizeModelPricing(data.pricing),
  }
}
