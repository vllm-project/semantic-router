import type { EditFormData, FieldConfig } from '../components/EditModal'
import {
  DEFAULT_SECTIONS,
  OPTIONAL_ROUTER_KEYS,
  PYTHON_ROUTER_KEYS,
  SECTION_META,
} from './configPageRouterDefaultsCatalog'
import type {
  CanonicalGlobalConfig,
  ConfigData,
  Tool,
} from './configPageSupport'

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

export type RouterLayerKey =
  | 'router'
  | 'services'
  | 'stores'
  | 'integrations'
  | 'model_catalog'

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

export const ROUTER_LAYER_META: Record<
  RouterLayerKey,
  { title: string; description: string }
> = {
  router: {
    title: 'Router',
    description: 'Core router-engine controls, startup behavior, and model-selection strategy.',
  },
  services: {
    title: 'Services',
    description: 'APIs, replay, observability, and other router-owned service surfaces.',
  },
  stores: {
    title: 'Stores',
    description: 'Shared storage-backed capabilities such as semantic cache and memory.',
  },
  integrations: {
    title: 'Integrations',
    description: 'Auxiliary runtime integrations used by routing and tool selection.',
  },
  model_catalog: {
    title: 'Model Catalog',
    description: 'Router-owned embedding catalogs, external models, and model-backed modules.',
  },
}

function cloneDefaultSection(key: RouterSystemKey): unknown {
  return JSON.parse(JSON.stringify(DEFAULT_SECTIONS[key]))
}

const GLOBAL_SECTION_PATHS: Record<RouterSystemKey, string[]> = {
  router_core: ['router'],
  response_api: ['services', 'response_api'],
  router_replay: ['services', 'router_replay'],
  authz: ['services', 'authz'],
  ratelimit: ['services', 'ratelimit'],
  memory: ['stores', 'memory'],
  semantic_cache: ['stores', 'semantic_cache'],
  vector_store: ['stores', 'vector_store'],
  tools: ['integrations', 'tools'],
  prompt_guard: ['model_catalog', 'modules', 'prompt_guard'],
  classifier: ['model_catalog', 'modules', 'classifier'],
  hallucination_mitigation: ['model_catalog', 'modules', 'hallucination_mitigation'],
  feedback_detector: ['model_catalog', 'modules', 'feedback_detector'],
  external_models: ['model_catalog', 'external'],
  system_models: ['model_catalog', 'system'],
  embedding_models: ['model_catalog', 'embeddings'],
  prompt_compression: ['model_catalog', 'modules', 'prompt_compression'],
  modality_detector: ['model_catalog', 'modules', 'modality_detector'],
  observability: ['services', 'observability'],
  looper: ['integrations', 'looper'],
  clear_route_cache: ['router', 'clear_route_cache'],
  model_selection: ['router', 'model_selection'],
  api: ['services', 'api'],
}

const ROUTER_SECTION_LAYERS: Record<RouterSystemKey, RouterLayerKey> = {
  router_core: 'router',
  response_api: 'services',
  router_replay: 'services',
  authz: 'services',
  ratelimit: 'services',
  memory: 'stores',
  semantic_cache: 'stores',
  vector_store: 'stores',
  tools: 'integrations',
  prompt_guard: 'model_catalog',
  classifier: 'model_catalog',
  hallucination_mitigation: 'model_catalog',
  feedback_detector: 'model_catalog',
  external_models: 'model_catalog',
  system_models: 'model_catalog',
  embedding_models: 'model_catalog',
  prompt_compression: 'model_catalog',
  modality_detector: 'model_catalog',
  observability: 'services',
  looper: 'integrations',
  clear_route_cache: 'router',
  model_selection: 'router',
  api: 'services',
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
    ? value as Record<string, unknown>
    : undefined
}

function asArray(value: unknown): Array<Record<string, unknown>> | undefined {
  return Array.isArray(value) ? value as Array<Record<string, unknown>> : undefined
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
  return value
    ? { label: 'Enabled', tone: 'active' }
    : { label: 'Disabled', tone: 'inactive' }
}

function sourceBadge(key: RouterSystemKey, routerDefaults: CanonicalGlobalConfig | null, data: unknown): { label: string; tone: 'active' | 'inactive' | 'info' } {
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

function getSectionValue(config: ConfigData | CanonicalGlobalConfig | null, key: RouterSystemKey): unknown {
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
        { label: 'User header', value: stringOrFallback(asObject(section?.identity)?.user_id_header, 'x-authz-user-id') },
        { label: 'Providers', value: `${(asArray(section?.providers) || []).length}` },
      ]
    case 'ratelimit':
      return [
        { label: 'Fail open', value: stringOrFallback(section?.fail_open, 'Disabled') },
        { label: 'Providers', value: `${(asArray(section?.providers) || []).length}` },
        { label: 'Rules', value: `${(asArray(section?.providers) || []).flatMap((provider) => asArray(provider.rules) || []).length}` },
      ]
    case 'memory':
      return [
        { label: 'Milvus address', value: stringOrFallback(asObject(section?.milvus)?.address) },
        { label: 'Embedding model', value: stringOrFallback(section?.embedding_model) },
        { label: 'Similarity threshold', value: percentOrFallback(section?.default_similarity_threshold) },
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
        { label: 'Model Ref', value: compactPathLikeString(section?.model_ref ?? section?.model_id) },
        { label: 'Threshold', value: percentOrFallback(section?.threshold) },
        { label: 'Runtime', value: section?.use_cpu ? 'CPU' : 'GPU' },
      ]
    case 'classifier': {
      const classifier = section
      return [
        { label: 'Domain model', value: compactPathLikeString(asObject(classifier?.domain)?.model_ref ?? asObject(classifier?.domain)?.model_id) },
        { label: 'PII model', value: compactPathLikeString(asObject(classifier?.pii)?.model_ref ?? asObject(classifier?.pii)?.model_id) },
        { label: 'Preference mode', value: asObject(classifier?.preference)?.use_contrastive ? 'Contrastive' : 'External / unset' },
      ]
    }
    case 'hallucination_mitigation':
      return [
        { label: 'Fact-check model', value: compactPathLikeString(asObject(section?.fact_check)?.model_ref ?? asObject(section?.fact_check)?.model_id) },
        { label: 'Detector model', value: compactPathLikeString(asObject(section?.detector)?.model_ref ?? asObject(section?.detector)?.model_id) },
        { label: 'Explainer model', value: compactPathLikeString(asObject(section?.explainer)?.model_ref ?? asObject(section?.explainer)?.model_id) },
      ]
    case 'feedback_detector':
      return [
        { label: 'Model Ref', value: compactPathLikeString(section?.model_ref ?? section?.model_id) },
        { label: 'Threshold', value: percentOrFallback(section?.threshold) },
        { label: 'Runtime', value: section?.use_cpu ? 'CPU' : 'GPU' },
      ]
    case 'external_models': {
      const models = asArray(data) || []
      const roles = models
        .map((item) => typeof item.model_role === 'string' ? item.model_role : null)
        .filter((value): value is string => Boolean(value))
      return [
        { label: 'Configured models', value: `${models.length}` },
        { label: 'Roles', value: roles.length ? roles.join(', ') : 'Not set' },
        { label: 'Providers', value: models.map((item) => typeof item.llm_provider === 'string' ? item.llm_provider : null).filter(Boolean).join(', ') || 'Not set' },
      ]
    }
    case 'system_models':
      return [
        { label: 'Prompt Guard', value: compactPathLikeString(section?.prompt_guard) },
        { label: 'Domain', value: compactPathLikeString(section?.domain_classifier) },
        { label: 'PII', value: compactPathLikeString(section?.pii_classifier) },
      ]
    case 'embedding_models':
      {
        const semantic = asObject(section?.semantic)
        return [
          { label: 'Semantic path', value: compactPathLikeString(semantic?.mmbert_model_path ?? semantic?.qwen3_model_path ?? semantic?.gemma_model_path ?? semantic?.multimodal_model_path) },
          { label: 'BERT path', value: compactPathLikeString(semantic?.bert_model_path) },
          { label: 'Runtime', value: semantic?.use_cpu ? 'CPU' : 'GPU' },
        ]
      }
    case 'prompt_compression':
      return [
        { label: 'Enabled', value: stringOrFallback(section?.enabled, 'Disabled') },
        { label: 'Max tokens', value: stringOrFallback(section?.max_tokens) },
        { label: 'Skip signals', value: (Array.isArray(section?.skip_signals) ? section?.skip_signals.join(', ') : 'Not set') || 'Not set' },
      ]
    case 'modality_detector':
      return [
        { label: 'Enabled', value: stringOrFallback(section?.enabled, 'Disabled') },
        { label: 'Method', value: stringOrFallback(section?.method) },
        { label: 'Prompt prefixes', value: `${Array.isArray(section?.prompt_prefixes) ? section.prompt_prefixes.length : 0}` },
      ]
    case 'observability':
      return [
        { label: 'Metrics', value: asObject(section?.metrics)?.enabled ? 'Enabled' : 'Disabled' },
        { label: 'Tracing provider', value: stringOrFallback(asObject(section?.tracing)?.provider) },
        { label: 'Trace endpoint', value: stringOrFallback(asObject(asObject(section?.tracing)?.exporter)?.endpoint) },
      ]
    case 'looper':
      return [
        { label: 'Endpoint', value: stringOrFallback(section?.endpoint) },
        { label: 'Timeout', value: section?.timeout_seconds ? `${section.timeout_seconds}s` : 'Not set' },
        { label: 'Headers', value: `${Object.keys(asObject(section?.headers) || {}).length}` },
      ]
    case 'clear_route_cache':
      return [
        { label: 'Startup behavior', value: data ? 'Clear route cache' : 'Retain route cache' },
      ]
    case 'model_selection':
      {
        const ml = asObject(section?.ml)
        return [
          { label: 'Method', value: stringOrFallback(section?.method) },
          { label: 'Models path', value: stringOrFallback(ml?.models_path) },
          { label: 'ML configured', value: ml ? 'Enabled' : 'Disabled' },
        ]
      }
    case 'api':
      {
        const batchClassification = asObject(section?.batch_classification)
        const metrics = asObject(batchClassification?.metrics)
        const batchRanges = Array.isArray(metrics?.batch_size_ranges) ? metrics.batch_size_ranges.length : 0
        return [
          { label: 'Metrics', value: metrics?.enabled ? 'Enabled' : 'Disabled' },
          { label: 'Batch size ranges', value: `${batchRanges}` },
          { label: 'Sample rate', value: typeof metrics?.sample_rate === 'number' ? `${metrics.sample_rate}` : 'Not set' },
        ]
      }
  }

  const keys = section ? Object.keys(section).length : 0
  return [{ label: 'Fields', value: `${keys}` }]
}

function badgesForKey(key: RouterSystemKey, data: unknown, ctx: RouterSectionContext): RouterSectionBadge[] {
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
    const configuredRefs = Object.values(section || {}).filter((value) => typeof value === 'string' && value.trim()).length
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

function fieldsForKey(key: RouterSystemKey): FieldConfig[] {
  switch (key) {
    case 'router_core':
      return [
        { name: 'config_source', label: 'Config Source', type: 'select', options: ['file', 'kubernetes'], required: true },
        { name: 'strategy', label: 'Routing Strategy', type: 'text', placeholder: 'static, elo, router_dc, automix...' },
        { name: 'auto_model_name', label: 'Auto Model Name', type: 'text', placeholder: 'MoM' },
        { name: 'include_config_models_in_list', label: 'Include Config Models In List', type: 'boolean' },
        { name: 'streamed_body', label: 'Streamed Body (JSON)', type: 'json', placeholder: '{"enabled":false}' },
      ]
    case 'response_api':
      return [
        { name: 'enabled', label: 'Enable Response API', type: 'boolean' },
        { name: 'store_backend', label: 'Store Backend', type: 'select', options: ['memory', 'milvus', 'redis'], required: true },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '86400' },
        { name: 'max_responses', label: 'Max Responses', type: 'number', placeholder: '1000' },
      ]
    case 'router_replay':
      return [
        { name: 'enabled', label: 'Enable Router Replay', type: 'boolean' },
        { name: 'store_backend', label: 'Store Backend', type: 'select', options: ['memory', 'redis', 'postgres', 'milvus'], required: true },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '2592000' },
        { name: 'async_writes', label: 'Async Writes', type: 'boolean' },
      ]
    case 'authz':
      return [
        { name: 'fail_open', label: 'Fail Open', type: 'boolean' },
        { name: 'identity', label: 'Identity Headers (JSON)', type: 'json', placeholder: '{"user_id_header":"x-authz-user-id","user_groups_header":"x-authz-user-groups"}' },
        { name: 'providers', label: 'Providers (JSON)', type: 'json', placeholder: '[]' },
      ]
    case 'ratelimit':
      return [
        { name: 'fail_open', label: 'Fail Open', type: 'boolean' },
        { name: 'providers', label: 'Providers (JSON)', type: 'json', placeholder: '[]' },
      ]
    case 'memory':
      return [
        { name: 'enabled', label: 'Enable Memory', type: 'boolean' },
        { name: 'auto_store', label: 'Auto Store Facts', type: 'boolean' },
        { name: 'milvus', label: 'Milvus Config (JSON)', type: 'json', placeholder: '{"address":"","collection":"agentic_memory","dimension":384}' },
        { name: 'embedding_model', label: 'Embedding Model', type: 'text', placeholder: 'bert' },
        { name: 'default_retrieval_limit', label: 'Default Retrieval Limit', type: 'number', placeholder: '5' },
        { name: 'default_similarity_threshold', label: 'Similarity Threshold', type: 'percentage', placeholder: '70' },
        { name: 'extraction_batch_size', label: 'Extraction Batch Size', type: 'number', placeholder: '10' },
        { name: 'hybrid_search', label: 'Hybrid Search', type: 'boolean' },
        { name: 'hybrid_mode', label: 'Hybrid Mode', type: 'text', placeholder: 'rerank' },
        { name: 'adaptive_threshold', label: 'Adaptive Threshold', type: 'boolean' },
        { name: 'quality_scoring', label: 'Quality Scoring (JSON)', type: 'json' },
        { name: 'reflection', label: 'Reflection (JSON)', type: 'json' },
      ]
    case 'semantic_cache':
      return [
        { name: 'enabled', label: 'Enable Semantic Cache', type: 'boolean' },
        { name: 'backend_type', label: 'Backend Type', type: 'select', options: ['memory', 'milvus', 'redis'], required: true },
        { name: 'similarity_threshold', label: 'Similarity Threshold', type: 'percentage', placeholder: '80' },
        { name: 'max_entries', label: 'Max Entries', type: 'number', placeholder: '1000' },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '3600' },
        { name: 'eviction_policy', label: 'Eviction Policy', type: 'select', options: ['fifo', 'lru', 'lfu'] },
        { name: 'embedding_model', label: 'Embedding Model Override', type: 'text', placeholder: 'mmbert' },
        { name: 'redis', label: 'Redis Backend (JSON)', type: 'json' },
        { name: 'milvus', label: 'Milvus Backend (JSON)', type: 'json' },
      ]
    case 'vector_store':
      return [
        { name: 'enabled', label: 'Enable Vector Store', type: 'boolean' },
        { name: 'backend_type', label: 'Backend Type', type: 'select', options: ['memory', 'milvus', 'llama_stack'], required: true },
        { name: 'file_storage_dir', label: 'File Storage Dir', type: 'text', placeholder: '/var/lib/vsr/data' },
        { name: 'max_file_size_mb', label: 'Max File Size (MB)', type: 'number', placeholder: '50' },
        { name: 'embedding_model', label: 'Embedding Model', type: 'select', options: ['bert', 'qwen3', 'gemma', 'mmbert', 'multimodal'] },
        { name: 'embedding_dimension', label: 'Embedding Dimension', type: 'number', placeholder: '384' },
        { name: 'ingestion_workers', label: 'Ingestion Workers', type: 'number', placeholder: '2' },
        { name: 'supported_formats', label: 'Supported Formats (JSON)', type: 'json', placeholder: '[".txt",".md",".json",".csv",".html"]' },
        { name: 'memory', label: 'Memory Backend (JSON)', type: 'json' },
        { name: 'milvus', label: 'Milvus Backend (JSON)', type: 'json' },
        { name: 'llama_stack', label: 'Llama Stack Backend (JSON)', type: 'json' },
      ]
    case 'tools':
      return [
        { name: 'enabled', label: 'Enable Tool Auto Selection', type: 'boolean' },
        { name: 'top_k', label: 'Top K', type: 'number', placeholder: '3' },
        { name: 'similarity_threshold', label: 'Similarity Threshold', type: 'percentage', placeholder: '20' },
        { name: 'tools_db_path', label: 'Tools DB Path', type: 'text', placeholder: 'config/tools_db.json' },
        { name: 'fallback_to_empty', label: 'Fallback To Empty', type: 'boolean' },
        { name: 'advanced_filtering', label: 'Advanced Filtering (JSON)', type: 'json' },
      ]
    case 'prompt_guard':
      return [
        { name: 'enabled', label: 'Enable Prompt Guard', type: 'boolean' },
        { name: 'model_ref', label: 'Model Ref', type: 'text', placeholder: 'prompt_guard' },
        { name: 'model_id', label: 'Model ID Override', type: 'text', placeholder: 'models/mmbert32k-jailbreak-detector-merged' },
        { name: 'threshold', label: 'Threshold', type: 'percentage', placeholder: '70' },
        { name: 'use_cpu', label: 'Use CPU', type: 'boolean' },
        { name: 'use_mmbert_32k', label: 'Use mmBERT 32K', type: 'boolean' },
        { name: 'use_modernbert', label: 'Use ModernBERT', type: 'boolean' },
        { name: 'jailbreak_mapping_path', label: 'Mapping Path', type: 'text', placeholder: 'models/.../jailbreak_type_mapping.json' },
      ]
    case 'classifier':
      return [
        { name: 'domain', label: 'Domain Module (JSON)', type: 'json' },
        { name: 'pii', label: 'PII Module (JSON)', type: 'json' },
        { name: 'mcp', label: 'MCP Module (JSON)', type: 'json' },
        { name: 'preference', label: 'Preference Module (JSON)', type: 'json' },
      ]
    case 'hallucination_mitigation':
      return [
        { name: 'enabled', label: 'Enable Hallucination Mitigation', type: 'boolean' },
        { name: 'on_hallucination_detected', label: 'On Detection Action', type: 'text', placeholder: 'block' },
        { name: 'fact_check', label: 'Fact Check Module (JSON)', type: 'json' },
        { name: 'detector', label: 'Detector Module (JSON)', type: 'json' },
        { name: 'explainer', label: 'Explainer Module (JSON)', type: 'json' },
      ]
    case 'feedback_detector':
      return [
        { name: 'enabled', label: 'Enable Feedback Detector', type: 'boolean' },
        { name: 'model_ref', label: 'Model Ref', type: 'text', placeholder: 'feedback_detector' },
        { name: 'model_id', label: 'Model ID Override', type: 'text', placeholder: 'models/mmbert32k-feedback-detector-merged' },
        { name: 'threshold', label: 'Threshold', type: 'percentage', placeholder: '70' },
        { name: 'use_cpu', label: 'Use CPU', type: 'boolean' },
        { name: 'use_mmbert_32k', label: 'Use mmBERT 32K', type: 'boolean' },
        { name: 'use_modernbert', label: 'Use ModernBERT', type: 'boolean' },
      ]
    case 'external_models':
      return [{ name: 'items', label: 'External Models (JSON)', type: 'json', placeholder: '[]' }]
    case 'system_models':
      return [
        { name: 'prompt_guard', label: 'Prompt Guard Binding', type: 'text', placeholder: 'models/mmbert32k-jailbreak-detector-merged' },
        { name: 'domain_classifier', label: 'Domain Classifier Binding', type: 'text', placeholder: 'models/mmbert32k-intent-classifier-merged' },
        { name: 'pii_classifier', label: 'PII Classifier Binding', type: 'text', placeholder: 'models/mmbert32k-pii-detector-merged' },
        { name: 'fact_check_classifier', label: 'Fact Check Binding', type: 'text', placeholder: 'models/mmbert32k-factcheck-classifier-merged' },
        { name: 'hallucination_detector', label: 'Hallucination Detector Binding', type: 'text', placeholder: 'models/mom-halugate-detector' },
        { name: 'hallucination_explainer', label: 'Hallucination Explainer Binding', type: 'text', placeholder: 'models/mom-halugate-explainer' },
        { name: 'feedback_detector', label: 'Feedback Detector Binding', type: 'text', placeholder: 'models/mmbert32k-feedback-detector-merged' },
      ]
    case 'embedding_models':
      return [
        { name: 'qwen3_model_path', label: 'Qwen3 Model Path', type: 'text', placeholder: 'models/mom-embedding-pro' },
        { name: 'gemma_model_path', label: 'Gemma Model Path', type: 'text', placeholder: 'models/mom-embedding-flash' },
        { name: 'mmbert_model_path', label: 'mmBERT Model Path', type: 'text', placeholder: 'models/mom-embedding-ultra' },
        { name: 'multimodal_model_path', label: 'Multimodal Model Path', type: 'text', placeholder: 'models/mom-embedding-multimodal' },
        { name: 'bert_model_path', label: 'BERT Model Path', type: 'text', placeholder: 'models/mom-embedding-bert' },
        { name: 'use_cpu', label: 'Use CPU', type: 'boolean' },
        { name: 'embedding_config', label: 'Embedding Config (JSON)', type: 'json' },
      ]
    case 'prompt_compression':
      return [
        { name: 'enabled', label: 'Enable Prompt Compression', type: 'boolean' },
        { name: 'max_tokens', label: 'Max Tokens', type: 'number', placeholder: '512' },
        { name: 'min_length', label: 'Min Length', type: 'number', placeholder: '64' },
        { name: 'skip_signals', label: 'Skip Signals (JSON)', type: 'json', placeholder: '["jailbreak","pii"]' },
        { name: 'textrank_weight', label: 'TextRank Weight', type: 'number', step: 0.01, placeholder: '1.0' },
        { name: 'position_weight', label: 'Position Weight', type: 'number', step: 0.01, placeholder: '1.0' },
        { name: 'tfidf_weight', label: 'TFIDF Weight', type: 'number', step: 0.01, placeholder: '1.0' },
        { name: 'position_depth', label: 'Position Depth', type: 'number', step: 0.01, placeholder: '1.0' },
      ]
    case 'modality_detector':
      return [
        { name: 'enabled', label: 'Enable Modality Detector', type: 'boolean' },
        { name: 'prompt_prefixes', label: 'Prompt Prefixes (JSON)', type: 'json', placeholder: '["generate an image of ","draw "]' },
        { name: 'method', label: 'Detection Method', type: 'select', options: ['classifier', 'keyword', 'hybrid'] },
        { name: 'classifier', label: 'Classifier (JSON)', type: 'json' },
        { name: 'keywords', label: 'Keywords (JSON)', type: 'json' },
        { name: 'both_keywords', label: 'Both Keywords (JSON)', type: 'json' },
        { name: 'confidence_threshold', label: 'Confidence Threshold', type: 'percentage', placeholder: '80' },
        { name: 'lower_threshold_ratio', label: 'Lower Threshold Ratio', type: 'percentage', placeholder: '60' },
      ]
    case 'observability':
      return [
        { name: 'metrics', label: 'Metrics Config (JSON)', type: 'json', placeholder: '{"enabled": true}' },
        { name: 'tracing', label: 'Tracing Config (JSON)', type: 'json' },
      ]
    case 'looper':
      return [
        { name: 'enabled', label: 'Enable Looper', type: 'boolean' },
        { name: 'endpoint', label: 'Endpoint', type: 'text', placeholder: 'http://localhost:8899/v1/chat/completions' },
        { name: 'timeout_seconds', label: 'Timeout (seconds)', type: 'number', placeholder: '1200' },
        { name: 'headers', label: 'Headers (JSON)', type: 'json', placeholder: '{}' },
      ]
    case 'clear_route_cache':
      return [{ name: 'value', label: 'Clear Route Cache On Reload', type: 'boolean' }]
    case 'model_selection':
      return [
        { name: 'enabled', label: 'Enable Model Selection', type: 'boolean' },
        { name: 'default_algorithm', label: 'Method', type: 'select', options: ['knn', 'kmeans', 'svm', 'elo', 'router_dc', 'automix', 'hybrid'], required: true },
        { name: 'models_path', label: 'ML Models Path', type: 'text', placeholder: 'models/model_selection' },
        { name: 'knn', label: 'KNN Config (JSON)', type: 'json' },
        { name: 'kmeans', label: 'KMeans Config (JSON)', type: 'json' },
        { name: 'svm', label: 'SVM Config (JSON)', type: 'json' },
        { name: 'elo', label: 'ELO Config (JSON)', type: 'json' },
        { name: 'router_dc', label: 'RouterDC Config (JSON)', type: 'json' },
        { name: 'automix', label: 'AutoMix Config (JSON)', type: 'json' },
        { name: 'hybrid', label: 'Hybrid Config (JSON)', type: 'json' },
      ]
    case 'api':
      return [{ name: 'batch_classification', label: 'Batch Classification (JSON)', type: 'json', placeholder: '{"metrics":{"enabled":true}}' }]
  }

  return []
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
      config_source: router?.config_source,
      strategy: router?.strategy,
      auto_model_name: router?.auto_model_name,
      include_config_models_in_list: router?.include_config_models_in_list,
      streamed_body: asObject(router?.streamed_body) || {},
    }
  }
  if (key === 'classifier') {
    const classifier = asObject(data)
    return {
      domain: asObject(classifier?.domain) || {},
      pii: asObject(classifier?.pii) || {},
      mcp: asObject(classifier?.mcp) || {},
      preference: asObject(classifier?.preference) || {},
    }
  }
  if (key === 'hallucination_mitigation') {
    const hallucination = asObject(data)
    return {
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
    return {
      ...(semantic || {}),
      bert: asObject(embeddings?.bert) || {},
    }
  }
  if (key === 'model_selection') {
    const selection = asObject(data)
    const ml = asObject(selection?.ml)
    return {
      enabled: selection?.enabled,
      default_algorithm: selection?.method,
      models_path: ml?.models_path,
      knn: asObject(ml?.knn) || {},
      kmeans: asObject(ml?.kmeans) || {},
      svm: asObject(ml?.svm) || {},
      elo: asObject(selection?.elo) || {},
      router_dc: asObject(selection?.router_dc) || {},
      automix: asObject(selection?.automix) || {},
      hybrid: asObject(selection?.hybrid) || {},
    }
  }
  const objectData = asObject(data)
  return objectData ? { ...objectData } : asObject(cloneDefaultSection(key)) || {}
}

function saveForKey(key: RouterSystemKey, data: EditFormData): Partial<ConfigData> {
  if (key === 'clear_route_cache') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], Boolean(data.value)) as Partial<ConfigData>
  }
  if (key === 'external_models') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], Array.isArray(data.items) ? data.items : []) as Partial<ConfigData>
  }
  if (key === 'router_core') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      config_source: data.config_source,
      strategy: data.strategy,
      auto_model_name: data.auto_model_name,
      include_config_models_in_list: Boolean(data.include_config_models_in_list),
      streamed_body: asObject(data.streamed_body) || {},
    }) as Partial<ConfigData>
  }
  if (key === 'classifier') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      domain: asObject(data.domain) || {},
      pii: asObject(data.pii) || {},
      mcp: asObject(data.mcp) || {},
      preference: asObject(data.preference) || {},
    }) as Partial<ConfigData>
  }
  if (key === 'hallucination_mitigation') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      enabled: Boolean(data.enabled),
      on_hallucination_detected: data.on_hallucination_detected,
      fact_check: asObject(data.fact_check) || {},
      detector: asObject(data.detector) || {},
      explainer: asObject(data.explainer) || {},
    }) as Partial<ConfigData>
  }
  if (key === 'embedding_models') {
    const { bert, ...semanticFields } = data
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      semantic: semanticFields,
      bert: asObject(bert) || {},
    }) as Partial<ConfigData>
  }
  if (key === 'model_selection') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      enabled: Boolean(data.enabled),
      method: data.default_algorithm,
      elo: asObject(data.elo) || {},
      router_dc: asObject(data.router_dc) || {},
      automix: asObject(data.automix) || {},
      hybrid: asObject(data.hybrid) || {},
      ml: {
        models_path: data.models_path,
        knn: asObject(data.knn) || {},
        kmeans: asObject(data.kmeans) || {},
        svm: asObject(data.svm) || {},
      },
    }) as Partial<ConfigData>
  }
  return buildNestedPatch(GLOBAL_SECTION_PATHS[key], data) as Partial<ConfigData>
}

export function buildEffectiveRouterConfig(
  routerDefaults: CanonicalGlobalConfig | null,
  config: ConfigData | null,
): RouterConfigSectionData {
  return {
    router_core: getSectionValue(routerDefaults, 'router_core') ?? getSectionValue(config, 'router_core'),
    response_api: getSectionValue(routerDefaults, 'response_api') ?? getSectionValue(config, 'response_api'),
    router_replay: getSectionValue(routerDefaults, 'router_replay') ?? getSectionValue(config, 'router_replay'),
    authz: getSectionValue(routerDefaults, 'authz') ?? getSectionValue(config, 'authz'),
    ratelimit: getSectionValue(routerDefaults, 'ratelimit') ?? getSectionValue(config, 'ratelimit'),
    memory: getSectionValue(routerDefaults, 'memory') ?? getSectionValue(config, 'memory'),
    semantic_cache: getSectionValue(routerDefaults, 'semantic_cache') ?? getSectionValue(config, 'semantic_cache'),
    vector_store: getSectionValue(routerDefaults, 'vector_store') ?? getSectionValue(config, 'vector_store'),
    tools: getSectionValue(routerDefaults, 'tools') ?? getSectionValue(config, 'tools'),
    prompt_guard: getSectionValue(routerDefaults, 'prompt_guard') ?? getSectionValue(config, 'prompt_guard'),
    classifier: getSectionValue(routerDefaults, 'classifier') ?? getSectionValue(config, 'classifier'),
    hallucination_mitigation: getSectionValue(routerDefaults, 'hallucination_mitigation') ?? getSectionValue(config, 'hallucination_mitigation'),
    feedback_detector: getSectionValue(routerDefaults, 'feedback_detector') ?? getSectionValue(config, 'feedback_detector'),
    external_models: getSectionValue(routerDefaults, 'external_models') ?? getSectionValue(config, 'external_models'),
    system_models: getSectionValue(routerDefaults, 'system_models') ?? getSectionValue(config, 'system_models'),
    embedding_models: getSectionValue(routerDefaults, 'embedding_models') ?? getSectionValue(config, 'embedding_models'),
    prompt_compression: getSectionValue(routerDefaults, 'prompt_compression') ?? getSectionValue(config, 'prompt_compression'),
    modality_detector: getSectionValue(routerDefaults, 'modality_detector') ?? getSectionValue(config, 'modality_detector'),
    observability: getSectionValue(routerDefaults, 'observability') ?? getSectionValue(config, 'observability'),
    looper: getSectionValue(routerDefaults, 'looper') ?? getSectionValue(config, 'looper'),
    clear_route_cache: getSectionValue(routerDefaults, 'clear_route_cache') ?? getSectionValue(config, 'clear_route_cache'),
    model_selection: getSectionValue(routerDefaults, 'model_selection') ?? getSectionValue(config, 'model_selection'),
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
