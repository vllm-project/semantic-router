import type { EditFormData, FieldConfig } from '../components/EditModal'
import {
  DEFAULT_SECTIONS,
  OPTIONAL_ROUTER_KEYS,
  PYTHON_ROUTER_KEYS,
  SECTION_META,
} from './configPageRouterDefaultsCatalog'
import { fieldsForKey } from './configPageRouterDefaultsFields'
import { GLOBAL_SECTION_PATHS, ROUTER_SECTION_LAYERS } from './configPageRouterDefaultsMetadata'
import {
  normalizeRouterStructuredFields,
  REMOTE_EMBEDDING_API_KEY_ENV,
} from './configPageRouterStructuredSchema'
import type { CanonicalGlobalConfig, ConfigData, Tool } from './configPageSupport'

export { ROUTER_LAYER_META } from './configPageRouterDefaultsMetadata'

export type RouterSystemKey =
  | 'router_core'
  | 'response_api'
  | 'router_replay'
  | 'authz'
  | 'ratelimit'
  | 'memory'
  | 'semantic_cache'
  | 'vector_store'
  | 'tools'
  | 'prompt_guard'
  | 'classifier'
  | 'hallucination_mitigation'
  | 'feedback_detector'
  | 'external_models'
  | 'system_models'
  | 'embedding_models'
  | 'prompt_compression'
  | 'modality_detector'
  | 'observability'
  | 'looper'
  | 'clear_route_cache'
  | 'model_selection'
  | 'api'

export type RouterConfigSectionData = Partial<Record<RouterSystemKey, unknown>>

export type RouterLayerKey = 'router' | 'services' | 'stores' | 'integrations' | 'model_catalog'

export interface RouterSectionBadge {
  label: string
  tone: 'active' | 'inactive' | 'info'
}

export interface RouterSectionSummaryItem {
  label: string
  value: string
}

export interface RouterSectionCard {
  key: RouterSystemKey
  layer: RouterLayerKey
  path: string[]
  title: string
  eyebrow: string
  description: string
  data: unknown
  sourceLabel: string
  sourceTone: 'active' | 'inactive' | 'info'
  status: RouterSectionBadge
  badges: RouterSectionBadge[]
  summary: RouterSectionSummaryItem[]
  editData: EditFormData
  editFields: FieldConfig[]
  save: (data: EditFormData) => Partial<ConfigData>
}

interface RouterSectionContext {
  config: ConfigData | null
  routerConfig: RouterConfigSectionData
  routerDefaults: CanonicalGlobalConfig | null
  toolsData: Tool[]
  toolsLoading: boolean
  toolsError: string | null
}

function cloneDefaultSection(key: RouterSystemKey): unknown {
  return JSON.parse(JSON.stringify(DEFAULT_SECTIONS[key]))
}

const LEGACY_ROOT_KEYS: Partial<Record<RouterSystemKey, keyof ConfigData>> = {
  response_api: 'response_api',
  router_replay: 'router_replay',
  memory: 'memory',
  semantic_cache: 'semantic_cache',
  tools: 'tools',
  prompt_guard: 'prompt_guard',
  classifier: 'classifier',
  hallucination_mitigation: 'hallucination_mitigation',
  feedback_detector: 'feedback_detector',
  external_models: 'external_models',
  embedding_models: 'embedding_models',
  observability: 'observability',
  looper: 'looper',
  clear_route_cache: 'clear_route_cache',
  model_selection: 'model_selection',
  api: 'api',
}

function asObject(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function asArray(value: unknown): Array<Record<string, unknown>> | undefined {
  return Array.isArray(value) ? (value as Array<Record<string, unknown>>) : undefined
}

function stringOrFallback(value: unknown, fallback = 'Not set'): string {
  if (typeof value === 'string' && value.trim()) {
    return value
  }
  if (typeof value === 'number') {
    return String(value)
  }
  if (typeof value === 'boolean') {
    return value ? 'Enabled' : 'Disabled'
  }
  return fallback
}

function compactPathLikeString(value: unknown, fallback = 'Not set'): string {
  if (typeof value !== 'string' || !value.trim()) {
    return stringOrFallback(value, fallback)
  }

  const trimmed = value.trim()
  if (trimmed.length <= 40 || !trimmed.includes('/')) {
    return trimmed
  }

  const segments = trimmed.split('/').filter(Boolean)
  if (segments.length === 0) {
    return trimmed
  }

  const tail = segments[segments.length - 1]
  if (segments[0] === 'models') {
    return `models/.../${tail}`
  }

  return `.../${tail}`
}

function percentOrFallback(value: unknown, fallback = 'Not set'): string {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : fallback
}

function enabledBadge(value: boolean | undefined): RouterSectionBadge {
  if (value === undefined) {
    return { label: 'Configured', tone: 'info' }
  }
  return value ? { label: 'Enabled', tone: 'active' } : { label: 'Disabled', tone: 'inactive' }
}

function sourceBadge(
  key: RouterSystemKey,
  routerDefaults: CanonicalGlobalConfig | null,
  data: unknown,
): { label: string; tone: 'active' | 'inactive' | 'info' } {
  if (getSectionValue(routerDefaults, key) !== undefined) {
    return { label: 'router effective defaults', tone: 'active' }
  }
  if (data !== undefined) {
    return { label: 'config.yaml override', tone: 'info' }
  }
  return { label: 'Router default available', tone: 'inactive' }
}

function getNestedValue(value: unknown, path: string[]): unknown {
  let current: unknown = value
  for (const segment of path) {
    const objectValue = asObject(current)
    if (!objectValue) {
      return undefined
    }
    current = objectValue[segment]
  }
  return current
}

function buildNestedPatch(path: string[], value: unknown): Record<string, unknown> {
  if (path.length === 0) {
    return {}
  }
  if (path.length === 1) {
    return { [path[0]]: value }
  }
  const [head, ...tail] = path
  return { [head]: buildNestedPatch(tail, value) }
}

function getSectionValue(
  config: ConfigData | CanonicalGlobalConfig | null,
  key: RouterSystemKey,
): unknown {
  if (!config) {
    return undefined
  }

  const canonicalValue = getNestedValue(config, GLOBAL_SECTION_PATHS[key])
  if (canonicalValue !== undefined) {
    return canonicalValue
  }

  const maybeConfig = config as ConfigData
  const globalValue = getNestedValue(maybeConfig.global, GLOBAL_SECTION_PATHS[key])
  if (globalValue !== undefined) {
    return globalValue
  }

  const legacyKey = LEGACY_ROOT_KEYS[key]
  return legacyKey ? maybeConfig[legacyKey] : undefined
}

function summaryForKey(key: RouterSystemKey, data: unknown): RouterSectionSummaryItem[] {
  const section = asObject(data)

  switch (key) {
    case 'router_core':
      return [
        { label: 'Config source', value: stringOrFallback(section?.config_source, 'file') },
        { label: 'Strategy', value: stringOrFallback(section?.strategy) },
        { label: 'Auto model name', value: stringOrFallback(section?.auto_model_name) },
        {
          label: 'Auto model aliases',
          value: Array.isArray(section?.auto_model_names)
            ? section.auto_model_names.join(', ')
            : 'Not set',
        },
      ]
    case 'response_api':
      return [
        { label: 'Store backend', value: stringOrFallback(section?.store_backend) },
        { label: 'TTL', value: section?.ttl_seconds ? `${section.ttl_seconds}s` : 'Not set' },
        { label: 'Max responses', value: stringOrFallback(section?.max_responses) },
      ]
    case 'router_replay':
      return [
        { label: 'Enabled', value: stringOrFallback(section?.enabled, 'Disabled') },
        { label: 'Store backend', value: stringOrFallback(section?.store_backend) },
        { label: 'Retention', value: section?.ttl_seconds ? `${section.ttl_seconds}s` : 'Not set' },
        { label: 'Async writes', value: stringOrFallback(section?.async_writes, 'Disabled') },
      ]
    case 'authz':
      return [
        { label: 'Fail open', value: stringOrFallback(section?.fail_open, 'Disabled') },
        {
          label: 'User header',
          value: stringOrFallback(asObject(section?.identity)?.user_id_header, 'x-authz-user-id'),
        },
        { label: 'Providers', value: `${(asArray(section?.providers) || []).length}` },
      ]
    case 'ratelimit':
      return [
        { label: 'Fail open', value: stringOrFallback(section?.fail_open, 'Disabled') },
        { label: 'Providers', value: `${(asArray(section?.providers) || []).length}` },
        {
          label: 'Rules',
          value: `${(asArray(section?.providers) || []).flatMap((provider) => asArray(provider.rules) || []).length}`,
        },
      ]
    case 'memory':
      return [
        { label: 'Milvus address', value: stringOrFallback(asObject(section?.milvus)?.address) },
        { label: 'Embedding model', value: stringOrFallback(section?.embedding_model) },
        {
          label: 'Similarity threshold',
          value: percentOrFallback(section?.default_similarity_threshold),
        },
      ]
    case 'semantic_cache':
      return [
        { label: 'Backend', value: stringOrFallback(section?.backend_type) },
        { label: 'Similarity threshold', value: percentOrFallback(section?.similarity_threshold) },
        { label: 'Retention', value: section?.ttl_seconds ? `${section.ttl_seconds}s` : 'Not set' },
      ]
    case 'vector_store':
      return [
        { label: 'Backend', value: stringOrFallback(section?.backend_type) },
        { label: 'Embedding model', value: stringOrFallback(section?.embedding_model) },
        { label: 'Enabled', value: stringOrFallback(section?.enabled, 'Disabled') },
      ]
    case 'tools':
      return [
        { label: 'Top K', value: stringOrFallback(section?.top_k) },
        { label: 'Similarity threshold', value: percentOrFallback(section?.similarity_threshold) },
        { label: 'Tool DB path', value: stringOrFallback(section?.tools_db_path) },
      ]
    case 'prompt_guard':
      return [
        {
          label: 'Model Ref',
          value: compactPathLikeString(section?.model_ref ?? section?.model_id),
        },
        { label: 'Threshold', value: percentOrFallback(section?.threshold) },
        { label: 'Runtime', value: section?.use_cpu ? 'CPU' : 'GPU' },
      ]
    case 'classifier': {
      const classifier = section
      return [
        {
          label: 'Domain model',
          value: compactPathLikeString(
            asObject(classifier?.domain)?.model_ref ?? asObject(classifier?.domain)?.model_id,
          ),
        },
        {
          label: 'PII model',
          value: compactPathLikeString(
            asObject(classifier?.pii)?.model_ref ?? asObject(classifier?.pii)?.model_id,
          ),
        },
        {
          label: 'Preference mode',
          value: asObject(classifier?.preference)?.use_contrastive
            ? 'Contrastive'
            : 'External / unset',
        },
      ]
    }
    case 'hallucination_mitigation':
      return [
        {
          label: 'Fact-check model',
          value: compactPathLikeString(
            asObject(section?.fact_check)?.model_ref ?? asObject(section?.fact_check)?.model_id,
          ),
        },
        {
          label: 'Detector model',
          value: compactPathLikeString(
            asObject(section?.detector)?.model_ref ?? asObject(section?.detector)?.model_id,
          ),
        },
        {
          label: 'Explainer model',
          value: compactPathLikeString(
            asObject(section?.explainer)?.model_ref ?? asObject(section?.explainer)?.model_id,
          ),
        },
      ]
    case 'feedback_detector':
      return [
        {
          label: 'Model Ref',
          value: compactPathLikeString(section?.model_ref ?? section?.model_id),
        },
        { label: 'Threshold', value: percentOrFallback(section?.threshold) },
        { label: 'Runtime', value: section?.use_cpu ? 'CPU' : 'GPU' },
      ]
    case 'external_models': {
      const models = asArray(data) || []
      const roles = models
        .map((item) => (typeof item.model_role === 'string' ? item.model_role : null))
        .filter((value): value is string => Boolean(value))
      return [
        { label: 'Configured models', value: `${models.length}` },
        { label: 'Roles', value: roles.length ? roles.join(', ') : 'Not set' },
        {
          label: 'Providers',
          value:
            models
              .map((item) => (typeof item.llm_provider === 'string' ? item.llm_provider : null))
              .filter(Boolean)
              .join(', ') || 'Not set',
        },
      ]
    }
    case 'system_models':
      return [
        { label: 'Prompt Guard', value: compactPathLikeString(section?.prompt_guard) },
        { label: 'Domain', value: compactPathLikeString(section?.domain_classifier) },
        { label: 'PII', value: compactPathLikeString(section?.pii_classifier) },
      ]
    case 'embedding_models': {
      const semantic = asObject(section?.semantic)
      return [
        {
          label: 'Semantic path',
          value: compactPathLikeString(
            semantic?.mmbert_model_path ??
              semantic?.qwen3_model_path ??
              semantic?.gemma_model_path ??
              semantic?.multimodal_model_path,
          ),
        },
        { label: 'BERT path', value: compactPathLikeString(semantic?.bert_model_path) },
        { label: 'Runtime', value: semantic?.use_cpu ? 'CPU' : 'GPU' },
      ]
    }
    case 'prompt_compression':
      return [
        { label: 'Enabled', value: stringOrFallback(section?.enabled, 'Disabled') },
        { label: 'Profile', value: stringOrFallback(section?.profile) },
        { label: 'Max tokens', value: stringOrFallback(section?.max_tokens) },
        {
          label: 'Skip signals',
          value:
            (Array.isArray(section?.skip_signals) ? section?.skip_signals.join(', ') : 'Not set') ||
            'Not set',
        },
      ]
    case 'modality_detector':
      return [
        { label: 'Enabled', value: stringOrFallback(section?.enabled, 'Disabled') },
        { label: 'Method', value: stringOrFallback(section?.method) },
        {
          label: 'Prompt prefixes',
          value: `${Array.isArray(section?.prompt_prefixes) ? section.prompt_prefixes.length : 0}`,
        },
      ]
    case 'observability':
      return [
        { label: 'Metrics', value: asObject(section?.metrics)?.enabled ? 'Enabled' : 'Disabled' },
        {
          label: 'Tracing provider',
          value: stringOrFallback(asObject(section?.tracing)?.provider),
        },
        {
          label: 'Trace endpoint',
          value: stringOrFallback(asObject(asObject(section?.tracing)?.exporter)?.endpoint),
        },
      ]
    case 'looper':
      return [
        { label: 'Endpoint', value: stringOrFallback(section?.endpoint) },
        {
          label: 'Timeout',
          value: section?.timeout_seconds ? `${section.timeout_seconds}s` : 'Not set',
        },
        { label: 'Headers', value: `${Object.keys(asObject(section?.headers) || {}).length}` },
      ]
    case 'clear_route_cache':
      return [
        {
          label: 'Auxiliary mutations',
          value: data ? 'Invalidate route cache' : 'Do not invalidate',
        },
      ]
    case 'model_selection': {
      const ml = asObject(section?.ml)
      return [
        { label: 'Method', value: stringOrFallback(section?.method) },
        { label: 'Models path', value: stringOrFallback(ml?.models_path) },
        { label: 'ML configured', value: ml ? 'Enabled' : 'Disabled' },
      ]
    }
    case 'api': {
      const batchClassification = asObject(section?.batch_classification)
      const metrics = asObject(batchClassification?.metrics)
      const batchRanges = Array.isArray(metrics?.batch_size_ranges)
        ? metrics.batch_size_ranges.length
        : 0
      return [
        { label: 'Metrics', value: metrics?.enabled ? 'Enabled' : 'Disabled' },
        { label: 'Batch size ranges', value: `${batchRanges}` },
        {
          label: 'Sample rate',
          value: typeof metrics?.sample_rate === 'number' ? `${metrics.sample_rate}` : 'Not set',
        },
      ]
    }
  }

  const keys = section ? Object.keys(section).length : 0
  return [{ label: 'Fields', value: `${keys}` }]
}

function badgesForKey(
  key: RouterSystemKey,
  data: unknown,
  ctx: RouterSectionContext,
): RouterSectionBadge[] {
  const section = asObject(data)
  const badges: RouterSectionBadge[] = []

  if (key === 'tools') {
    if (ctx.toolsLoading) {
      badges.push({ label: 'Loading tools DB', tone: 'info' })
    } else if (ctx.toolsError) {
      badges.push({ label: 'Tools DB error', tone: 'inactive' })
    } else if (ctx.toolsData.length > 0) {
      badges.push({ label: `${ctx.toolsData.length} tools loaded`, tone: 'active' })
    }
  }

  if (key === 'embedding_models') {
    const hnsw = asObject(section?.embedding_config)
    if (hnsw?.preload_embeddings !== undefined) {
      badges.push({
        label: hnsw.preload_embeddings ? 'Preload embeddings' : 'Lazy embeddings',
        tone: hnsw.preload_embeddings ? 'active' : 'info',
      })
    }
  }

  if (key === 'system_models') {
    const configuredRefs = Object.values(section || {}).filter(
      (value) => typeof value === 'string' && value.trim(),
    ).length
    badges.push({
      label: `${configuredRefs} bindings`,
      tone: configuredRefs > 0 ? 'active' : 'inactive',
    })
  }

  if (key === 'external_models') {
    const models = asArray(data) || []
    if (models.length === 0) {
      badges.push({ label: 'No external models', tone: 'inactive' })
    }
  }

  return badges
}

function statusForKey(data: unknown): RouterSectionBadge {
  if (data === undefined) {
    return { label: 'Missing', tone: 'inactive' }
  }
  if (typeof data === 'boolean') {
    return enabledBadge(data)
  }
  if (Array.isArray(data)) {
    return data.length > 0
      ? { label: `${data.length} configured`, tone: 'active' }
      : { label: 'Not configured', tone: 'inactive' }
  }
  const section = asObject(data)
  return enabledBadge(typeof section?.enabled === 'boolean' ? section.enabled : undefined)
}

function editDataForKey(key: RouterSystemKey, data: unknown): EditFormData {
  if (key === 'clear_route_cache') {
    return { value: Boolean(data) }
  }
  if (key === 'external_models') {
    return { items: Array.isArray(data) ? data : cloneDefaultSection(key) }
  }
  if (key === 'router_core') {
    const router = asObject(data)
    return {
      ...(router || {}),
      config_source: router?.config_source,
      strategy: router?.strategy,
      auto_model_name: router?.auto_model_name,
      auto_model_names: Array.isArray(router?.auto_model_names) ? router.auto_model_names : [],
      include_config_models_in_list: router?.include_config_models_in_list,
      streamed_body: asObject(router?.streamed_body) || {},
    }
  }
  if (key === 'classifier') {
    const classifier = asObject(data)
    return {
      ...(classifier || {}),
      domain: asObject(classifier?.domain) || {},
      pii: asObject(classifier?.pii) || {},
      mcp: asObject(classifier?.mcp) || {},
      preference: asObject(classifier?.preference) || {},
    }
  }
  if (key === 'hallucination_mitigation') {
    const hallucination = asObject(data)
    return {
      ...(hallucination || {}),
      enabled: hallucination?.enabled,
      on_hallucination_detected: hallucination?.on_hallucination_detected,
      fact_check: asObject(hallucination?.fact_check) || {},
      detector: asObject(hallucination?.detector) || {},
      explainer: asObject(hallucination?.explainer) || {},
    }
  }
  if (key === 'embedding_models') {
    const embeddings = asObject(data)
    const semantic = asObject(embeddings?.semantic)
    const endpoint = asObject(semantic?.endpoint)
    return {
      ...(semantic || {}),
      endpoint: endpoint
        ? {
            ...endpoint,
            api_key_env:
              endpoint.api_key_env === REMOTE_EMBEDDING_API_KEY_ENV
                ? REMOTE_EMBEDDING_API_KEY_ENV
                : undefined,
          }
        : undefined,
      bert: asObject(embeddings?.bert) || {},
      __catalog: embeddings || {},
    }
  }
  if (key === 'model_selection') {
    const selection = asObject(data)
    const ml = asObject(selection?.ml)
    return {
      ...(selection || {}),
      enabled: selection?.enabled,
      default_algorithm: selection?.method,
      models_path: ml?.models_path,
      knn: asObject(ml?.knn) || {},
      kmeans: asObject(ml?.kmeans) || {},
      svm: asObject(ml?.svm) || {},
      router_dc: asObject(selection?.router_dc) || {},
      automix: asObject(selection?.automix) || {},
      hybrid: asObject(selection?.hybrid) || {},
    }
  }
  const objectData = asObject(data)
  return objectData ? { ...objectData } : asObject(cloneDefaultSection(key)) || {}
}

function saveForKey(key: RouterSystemKey, rawData: EditFormData): Partial<ConfigData> {
  const data = normalizeRouterStructuredFields(key, rawData)
  if (key === 'clear_route_cache') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], Boolean(data.value)) as Partial<ConfigData>
  }
  if (key === 'external_models') {
    return buildNestedPatch(
      GLOBAL_SECTION_PATHS[key],
      Array.isArray(data.items) ? data.items : [],
    ) as Partial<ConfigData>
  }
  if (key === 'router_core') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      ...data,
      config_source: data.config_source,
      strategy: data.strategy,
      auto_model_name: data.auto_model_name,
      auto_model_names: Array.isArray(data.auto_model_names) ? data.auto_model_names : [],
      include_config_models_in_list: Boolean(data.include_config_models_in_list),
      streamed_body: asObject(data.streamed_body) || {},
    }) as Partial<ConfigData>
  }
  if (key === 'classifier') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      ...data,
      domain: asObject(data.domain) || {},
      pii: asObject(data.pii) || {},
      mcp: asObject(data.mcp) || {},
      preference: asObject(data.preference) || {},
    }) as Partial<ConfigData>
  }
  if (key === 'hallucination_mitigation') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      ...data,
      enabled: Boolean(data.enabled),
      on_hallucination_detected: data.on_hallucination_detected,
      fact_check: asObject(data.fact_check) || {},
      detector: asObject(data.detector) || {},
      explainer: asObject(data.explainer) || {},
    }) as Partial<ConfigData>
  }
  if (key === 'embedding_models') {
    const { bert, __catalog, endpoint, ...semanticFields } = data
    const catalog = asObject(__catalog)
    const originalSemantic = asObject(catalog?.semantic)
    const originalEndpoint = asObject(originalSemantic?.endpoint)
    const currentEndpoint = asObject(endpoint)
    const endpointPatch = currentEndpoint
      ? buildEmbeddingEndpointPatch(originalEndpoint, currentEndpoint)
      : originalEndpoint
        ? null
        : undefined
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      ...(catalog || {}),
      semantic: {
        ...(originalSemantic || {}),
        ...semanticFields,
        ...(endpointPatch !== undefined ? { endpoint: endpointPatch } : {}),
      },
      bert: asObject(bert) || {},
    }) as Partial<ConfigData>
  }
  if (key === 'model_selection') {
    const { default_algorithm, models_path, knn, kmeans, svm, ml, ...selectionFields } = data
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      ...selectionFields,
      enabled: Boolean(data.enabled),
      method: default_algorithm,
      router_dc: asObject(data.router_dc) || {},
      automix: asObject(data.automix) || {},
      hybrid: asObject(data.hybrid) || {},
      ml: {
        ...(asObject(ml) || {}),
        models_path,
        knn: asObject(knn) || {},
        kmeans: asObject(kmeans) || {},
        svm: asObject(svm) || {},
      },
    }) as Partial<ConfigData>
  }
  return buildNestedPatch(GLOBAL_SECTION_PATHS[key], data) as Partial<ConfigData>
}

const REMOTE_EMBEDDING_OPTIONAL_FIELDS = [
  'api_key_env',
  'timeout_seconds',
  'max_retries',
  'dimensions',
] as const

function buildEmbeddingEndpointPatch(
  original: Record<string, unknown> | undefined,
  current: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  if (!current) return undefined

  const patch: Record<string, unknown> = {
    ...current,
  }
  if (current.api_key_env === REMOTE_EMBEDDING_API_KEY_ENV) {
    patch.api_key_env = REMOTE_EMBEDDING_API_KEY_ENV
  } else {
    delete patch.api_key_env
  }
  for (const key of REMOTE_EMBEDDING_OPTIONAL_FIELDS) {
    if (
      original &&
      Object.prototype.hasOwnProperty.call(original, key) &&
      patch[key] === undefined
    ) {
      patch[key] = null
    }
  }
  return patch
}

export function buildEffectiveRouterConfig(
  routerDefaults: CanonicalGlobalConfig | null,
  config: ConfigData | null,
): RouterConfigSectionData {
  return {
    router_core:
      getSectionValue(routerDefaults, 'router_core') ?? getSectionValue(config, 'router_core'),
    response_api:
      getSectionValue(routerDefaults, 'response_api') ?? getSectionValue(config, 'response_api'),
    router_replay:
      getSectionValue(routerDefaults, 'router_replay') ?? getSectionValue(config, 'router_replay'),
    authz: getSectionValue(routerDefaults, 'authz') ?? getSectionValue(config, 'authz'),
    ratelimit: getSectionValue(routerDefaults, 'ratelimit') ?? getSectionValue(config, 'ratelimit'),
    memory: getSectionValue(routerDefaults, 'memory') ?? getSectionValue(config, 'memory'),
    semantic_cache:
      getSectionValue(routerDefaults, 'semantic_cache') ??
      getSectionValue(config, 'semantic_cache'),
    vector_store:
      getSectionValue(routerDefaults, 'vector_store') ?? getSectionValue(config, 'vector_store'),
    tools: getSectionValue(routerDefaults, 'tools') ?? getSectionValue(config, 'tools'),
    prompt_guard:
      getSectionValue(routerDefaults, 'prompt_guard') ?? getSectionValue(config, 'prompt_guard'),
    classifier:
      getSectionValue(routerDefaults, 'classifier') ?? getSectionValue(config, 'classifier'),
    hallucination_mitigation:
      getSectionValue(routerDefaults, 'hallucination_mitigation') ??
      getSectionValue(config, 'hallucination_mitigation'),
    feedback_detector:
      getSectionValue(routerDefaults, 'feedback_detector') ??
      getSectionValue(config, 'feedback_detector'),
    external_models:
      getSectionValue(routerDefaults, 'external_models') ??
      getSectionValue(config, 'external_models'),
    system_models:
      getSectionValue(routerDefaults, 'system_models') ?? getSectionValue(config, 'system_models'),
    embedding_models:
      getSectionValue(routerDefaults, 'embedding_models') ??
      getSectionValue(config, 'embedding_models'),
    prompt_compression:
      getSectionValue(routerDefaults, 'prompt_compression') ??
      getSectionValue(config, 'prompt_compression'),
    modality_detector:
      getSectionValue(routerDefaults, 'modality_detector') ??
      getSectionValue(config, 'modality_detector'),
    observability:
      getSectionValue(routerDefaults, 'observability') ?? getSectionValue(config, 'observability'),
    looper: getSectionValue(routerDefaults, 'looper') ?? getSectionValue(config, 'looper'),
    clear_route_cache:
      getSectionValue(routerDefaults, 'clear_route_cache') ??
      getSectionValue(config, 'clear_route_cache'),
    model_selection:
      getSectionValue(routerDefaults, 'model_selection') ??
      getSectionValue(config, 'model_selection'),
    api: getSectionValue(routerDefaults, 'api') ?? getSectionValue(config, 'api'),
  }
}

export function buildRouterSectionCards(ctx: RouterSectionContext): RouterSectionCard[] {
  const optionalKeys = OPTIONAL_ROUTER_KEYS.filter((key) => ctx.routerConfig[key] !== undefined)
  const orderedKeys = [...PYTHON_ROUTER_KEYS, ...optionalKeys]

  return orderedKeys.map((key) => {
    const data = ctx.routerConfig[key]
    const meta = SECTION_META[key]
    const source = sourceBadge(key, ctx.routerDefaults, data)

    return {
      key,
      layer: ROUTER_SECTION_LAYERS[key],
      path: GLOBAL_SECTION_PATHS[key],
      title: meta.title,
      eyebrow: meta.eyebrow,
      description: meta.description,
      data,
      sourceLabel: source.label,
      sourceTone: source.tone,
      status: statusForKey(data),
      badges: badgesForKey(key, data, ctx),
      summary: summaryForKey(key, data),
      editData: editDataForKey(key, data),
      editFields: fieldsForKey(key),
      save: (nextData) => saveForKey(key, nextData),
    }
  })
}
