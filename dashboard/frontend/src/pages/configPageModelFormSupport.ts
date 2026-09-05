import type {
  BackendRefEntry,
  ConfigData,
  LoRAAdapter,
  ModelEvaluationConfig,
  ModelPricing,
  ModelReasoningConfig,
  ProviderReliability,
} from './configPageSupport'

export function normalizeModelLoras(value: unknown): LoRAAdapter[] {
  if (!Array.isArray(value)) return []

  return value
    .filter(
      (entry): entry is Record<string, unknown> =>
        Boolean(entry) && typeof entry === 'object' && !Array.isArray(entry),
    )
    .map((entry) => ({
      name: typeof entry.name === 'string' ? entry.name.trim() : '',
      description:
        typeof entry.description === 'string' && entry.description.trim()
          ? entry.description.trim()
          : undefined,
    }))
    .filter((entry) => entry.name)
}

export function normalizeModelBackendRefs(value: unknown): BackendRefEntry[] {
  if (!Array.isArray(value)) return []

  return value
    .filter(
      (entry): entry is Record<string, unknown> =>
        Boolean(entry) && typeof entry === 'object' && !Array.isArray(entry),
    )
    .map((entry) => {
      const normalized: BackendRefEntry = {}
      if (typeof entry.name === 'string' && entry.name.trim()) normalized.name = entry.name.trim()
      if (typeof entry.endpoint === 'string' && entry.endpoint.trim())
        normalized.endpoint = entry.endpoint.trim()
      if (entry.protocol === 'https') normalized.protocol = 'https'
      else if (entry.protocol === 'http') normalized.protocol = 'http'
      if (typeof entry.weight === 'number' && Number.isFinite(entry.weight))
        normalized.weight = entry.weight
      if (typeof entry.base_url === 'string' && entry.base_url.trim())
        normalized.base_url = entry.base_url.trim()
      if (typeof entry.provider === 'string' && entry.provider.trim())
        normalized.provider = entry.provider.trim()
      if (typeof entry.auth_header === 'string' && entry.auth_header.trim())
        normalized.auth_header = entry.auth_header.trim()
      if (typeof entry.auth_prefix === 'string' && entry.auth_prefix.trim())
        normalized.auth_prefix = entry.auth_prefix.trim()
      if (
        entry.extra_headers &&
        typeof entry.extra_headers === 'object' &&
        !Array.isArray(entry.extra_headers)
      ) {
        normalized.extra_headers = Object.fromEntries(
          Object.entries(entry.extra_headers as Record<string, unknown>)
            .filter(([, nestedValue]) => typeof nestedValue === 'string')
            .map(([key, nestedValue]) => [key, nestedValue as string]),
        )
      }
      if (typeof entry.api_version === 'string' && entry.api_version.trim())
        normalized.api_version = entry.api_version.trim()
      if (typeof entry.chat_path === 'string' && entry.chat_path.trim())
        normalized.chat_path = entry.chat_path.trim()
      if (typeof entry.api_key === 'string' && entry.api_key.trim())
        normalized.api_key = entry.api_key.trim()
      if (typeof entry.api_key_env === 'string' && entry.api_key_env.trim())
        normalized.api_key_env = entry.api_key_env.trim()
      return normalized
    })
}

export function normalizeModelEvaluations(value: unknown): ModelEvaluationConfig[] {
  if (!Array.isArray(value)) return []

  return value
    .filter(
      (entry): entry is Record<string, unknown> =>
        Boolean(entry) && typeof entry === 'object' && !Array.isArray(entry),
    )
    .map((entry) => {
      const metrics =
        entry.metrics && typeof entry.metrics === 'object' && !Array.isArray(entry.metrics)
          ? Object.fromEntries(
              Object.entries(entry.metrics as Record<string, unknown>)
                .filter(([, metric]) =>
                  (typeof metric === 'number' && Number.isFinite(metric)) ||
                  (typeof metric === 'string' &&
                    metric.trim() !== '' &&
                    Number.isFinite(Number(metric))),
                )
                .map(([key, metric]) => [key.trim(), Number(metric)] as const)
                .filter(([key]) => key.length > 0),
            )
          : {}
      const metadata =
        entry.metadata && typeof entry.metadata === 'object' && !Array.isArray(entry.metadata)
          ? Object.fromEntries(
              Object.entries(entry.metadata as Record<string, unknown>)
                .filter(([key, item]) =>
                  Boolean(key.trim()) &&
                  (typeof item === 'string' ||
                    typeof item === 'number' ||
                    typeof item === 'boolean' ||
                    item === null),
                )
                .map(
                  ([key, item]) =>
                    [key.trim(), item as string | number | boolean | null] as const,
                ),
            )
          : undefined
      return {
        benchmark: typeof entry.benchmark === 'string' ? entry.benchmark.trim() : '',
        metrics,
        source:
          typeof entry.source === 'string' && entry.source.trim()
            ? entry.source.trim()
            : undefined,
        measured_at:
          typeof entry.measured_at === 'string' && entry.measured_at.trim()
            ? entry.measured_at.trim()
            : undefined,
        metadata: metadata && Object.keys(metadata).length > 0 ? metadata : undefined,
      }
    })
    .filter((entry) => entry.benchmark && Object.keys(entry.metrics).length > 0)
}

function normalizeReasoning(data: Record<string, unknown>): ModelReasoningConfig | undefined {
  const family = typeof data.reasoning_family === 'string' ? data.reasoning_family.trim() : ''
  const type = typeof data.reasoning_type === 'string' ? data.reasoning_type.trim() : ''
  const parameter =
    typeof data.reasoning_parameter === 'string' ? data.reasoning_parameter.trim() : ''
  if (family) return { family }
  if (!type && !parameter) return undefined
  const levels =
    typeof data.reasoning_levels === 'string'
      ? data.reasoning_levels
          .split(',')
          .map((level) => level.trim())
          .filter(Boolean)
      : []
  const defaultLevel =
    typeof data.reasoning_default === 'string' ? data.reasoning_default.trim() : ''
  return {
    type,
    parameter,
    levels: levels.length > 0 ? levels : undefined,
    default: defaultLevel || undefined,
  }
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
  if (typeof pricing.currency === 'string' && pricing.currency.trim())
    normalized.currency = pricing.currency.trim()
  if (typeof pricing.prompt_per_1m === 'number' && Number.isFinite(pricing.prompt_per_1m))
    normalized.prompt_per_1m = pricing.prompt_per_1m
  if (
    typeof pricing.cached_input_per_1m === 'number' &&
    Number.isFinite(pricing.cached_input_per_1m)
  ) {
    normalized.cached_input_per_1m = pricing.cached_input_per_1m
  }
  if (
    typeof pricing.cache_write_per_1m === 'number' &&
    Number.isFinite(pricing.cache_write_per_1m)
  ) {
    normalized.cache_write_per_1m = pricing.cache_write_per_1m
  }
  if (typeof pricing.completion_per_1m === 'number' && Number.isFinite(pricing.completion_per_1m))
    normalized.completion_per_1m = pricing.completion_per_1m
  return Object.keys(normalized).length > 0 ? normalized : undefined
}

export function normalizeModelReliability(value: unknown): ProviderReliability | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined

  const source = value as Record<string, unknown>
  const normalized: ProviderReliability = {}
  const stringFields = [
    'lb_policy',
    'retry_on',
    'base_ejection_time',
    'health_check_path',
    'health_check_interval',
    'health_check_timeout',
  ] as const
  const numberFields = ['retry_count', 'consecutive_5xx', 'max_ejection_percent'] as const

  for (const field of stringFields) {
    if (typeof source[field] === 'string' && source[field].trim()) {
      normalized[field] = source[field].trim()
    }
  }
  for (const field of numberFields) {
    if (typeof source[field] === 'number' && Number.isFinite(source[field])) {
      normalized[field] = source[field]
    }
  }
  return Object.keys(normalized).length > 0 ? normalized : undefined
}

export function buildProviderModelPayload(
  name: string,
  data: Record<string, unknown>,
  existingModel?: NonNullable<NonNullable<ConfigData['providers']>['models']>[number],
) {
  const catalog =
    typeof data.catalog === 'string'
      ? data.catalog.trim() || undefined
      : existingModel?.catalog
  return {
    name,
    catalog,
    reasoning: catalog ? undefined : normalizeReasoning(data),
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
    reliability: normalizeModelReliability(data.reliability),
  }
}
