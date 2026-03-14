import type { FieldConfig } from '../components/EditModal'
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
  | 'response_api'
  | 'router_replay'
  | 'memory'
  | 'semantic_cache'
  | 'tools'
  | 'prompt_guard'
  | 'classifier'
  | 'hallucination_mitigation'
  | 'feedback_detector'
  | 'external_models'
  | 'embedding_models'
  | 'observability'
  | 'looper'
  | 'clear_route_cache'
  | 'model_selection'
  | 'api'
  | 'bert_model'

export type RouterConfigSectionData = Partial<Record<RouterSystemKey, unknown>>

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
  title: string
  eyebrow: string
  description: string
  data: unknown
  sourceLabel: string
  sourceTone: 'active' | 'inactive' | 'info'
  status: RouterSectionBadge
  badges: RouterSectionBadge[]
  summary: RouterSectionSummaryItem[]
  editData: Record<string, unknown>
  editFields: FieldConfig[]
  save: (data: Record<string, unknown>) => Partial<ConfigData>
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

const GLOBAL_SECTION_PATHS: Record<RouterSystemKey, string[]> = {
  response_api: ['services', 'response_api'],
  router_replay: ['services', 'router_replay'],
  memory: ['stores', 'memory'],
  semantic_cache: ['stores', 'semantic_cache'],
  tools: ['integrations', 'tools'],
  prompt_guard: ['model_catalog', 'modules', 'prompt_guard'],
  classifier: ['model_catalog', 'modules', 'classifier'],
  hallucination_mitigation: ['model_catalog', 'modules', 'hallucination_mitigation'],
  feedback_detector: ['model_catalog', 'modules', 'feedback_detector'],
  external_models: ['model_catalog', 'external'],
  embedding_models: ['model_catalog', 'embeddings'],
  observability: ['services', 'observability'],
  looper: ['integrations', 'looper'],
  clear_route_cache: ['router', 'clear_route_cache'],
  model_selection: ['router', 'model_selection'],
  api: ['services', 'api'],
  bert_model: ['model_catalog', 'embeddings', 'bert'],
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
  bert_model: 'bert_model',
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
    case 'response_api':
      return [
        { label: 'Store backend', value: stringOrFallback(section?.store_backend) },
        { label: 'TTL', value: section?.ttl_seconds ? `${section.ttl_seconds}s` : 'Not set' },
        { label: 'Max responses', value: stringOrFallback(section?.max_responses) },
      ]
    case 'router_replay':
      return [
        { label: 'Store backend', value: stringOrFallback(section?.store_backend) },
        { label: 'Retention', value: section?.ttl_seconds ? `${section.ttl_seconds}s` : 'Not set' },
        { label: 'Async writes', value: stringOrFallback(section?.async_writes, 'Disabled') },
      ]
    case 'memory':
      return [
        { label: 'Milvus address', value: stringOrFallback(asObject(section?.milvus)?.address) },
        { label: 'Embedding model', value: stringOrFallback(asObject(section?.embedding)?.model) },
        { label: 'Similarity threshold', value: percentOrFallback(section?.default_similarity_threshold) },
      ]
    case 'semantic_cache':
      return [
        { label: 'Backend', value: stringOrFallback(section?.backend_type) },
        { label: 'Similarity threshold', value: percentOrFallback(section?.similarity_threshold) },
        { label: 'Retention', value: section?.ttl_seconds ? `${section.ttl_seconds}s` : 'Not set' },
      ]
    case 'tools':
      return [
        { label: 'Top K', value: stringOrFallback(section?.top_k) },
        { label: 'Similarity threshold', value: percentOrFallback(section?.similarity_threshold) },
        { label: 'Tool DB path', value: stringOrFallback(section?.tools_db_path) },
      ]
    case 'prompt_guard':
      return [
        { label: 'Model Ref', value: stringOrFallback(section?.model_ref ?? section?.model_id) },
        { label: 'Threshold', value: percentOrFallback(section?.threshold) },
        { label: 'Runtime', value: section?.use_cpu ? 'CPU' : 'GPU' },
      ]
    case 'classifier': {
      const classifier = section
      return [
        { label: 'Domain model', value: stringOrFallback(asObject(classifier?.domain)?.model_ref ?? asObject(classifier?.domain)?.model_id) },
        { label: 'PII model', value: stringOrFallback(asObject(classifier?.pii)?.model_ref ?? asObject(classifier?.pii)?.model_id) },
        { label: 'Preference mode', value: asObject(classifier?.preference)?.use_contrastive ? 'Contrastive' : 'External / unset' },
      ]
    }
    case 'hallucination_mitigation':
      return [
        { label: 'Fact-check model', value: stringOrFallback(asObject(section?.fact_check)?.model_ref ?? asObject(section?.fact_check)?.model_id) },
        { label: 'Detector model', value: stringOrFallback(asObject(section?.detector)?.model_ref ?? asObject(section?.detector)?.model_id) },
        { label: 'Explainer model', value: stringOrFallback(asObject(section?.explainer)?.model_ref ?? asObject(section?.explainer)?.model_id) },
      ]
    case 'feedback_detector':
      return [
        { label: 'Model Ref', value: stringOrFallback(section?.model_ref ?? section?.model_id) },
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
    case 'embedding_models':
      {
        const semantic = asObject(section?.semantic)
        const bert = asObject(section?.bert)
        return [
          { label: 'Semantic path', value: stringOrFallback(semantic?.mmbert_model_path ?? semantic?.qwen3_model_path ?? semantic?.gemma_model_path ?? semantic?.multimodal_model_path) },
          { label: 'BERT path', value: stringOrFallback(bert?.model_id) },
          { label: 'Runtime', value: semantic?.use_cpu ? 'CPU' : 'GPU' },
        ]
      }
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
      return [
        { label: 'Max batch size', value: stringOrFallback(asObject(section?.batch_classification)?.max_batch_size) },
        { label: 'Concurrency threshold', value: stringOrFallback(asObject(section?.batch_classification)?.concurrency_threshold) },
        { label: 'Metrics', value: asObject(asObject(section?.batch_classification)?.metrics)?.enabled ? 'Enabled' : 'Disabled' },
      ]
    case 'bert_model':
      return [
        { label: 'Model ID', value: stringOrFallback(section?.model_id) },
        { label: 'Threshold', value: percentOrFallback(section?.threshold) },
        { label: 'Runtime', value: section?.use_cpu ? 'CPU' : 'GPU' },
      ]
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
    const hnsw = asObject(section?.hnsw_config)
    if (hnsw?.preload_embeddings !== undefined) {
      badges.push({
        label: hnsw.preload_embeddings ? 'Preload embeddings' : 'Lazy embeddings',
        tone: hnsw.preload_embeddings ? 'active' : 'info',
      })
    }
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
    case 'response_api':
      return [
        { name: 'enabled', label: 'Enable Response API', type: 'boolean' },
        { name: 'store_backend', label: 'Store Backend', type: 'select', options: ['memory', 'milvus', 'redis'], required: true },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '86400' },
        { name: 'max_responses', label: 'Max Responses', type: 'number', placeholder: '1000' },
      ]
    case 'router_replay':
      return [
        { name: 'store_backend', label: 'Store Backend', type: 'select', options: ['memory', 'redis', 'postgres', 'milvus'], required: true },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '2592000' },
        { name: 'async_writes', label: 'Async Writes', type: 'boolean' },
      ]
    case 'memory':
      return [
        { name: 'enabled', label: 'Enable Memory', type: 'boolean' },
        { name: 'auto_store', label: 'Auto Store Facts', type: 'boolean' },
        { name: 'milvus', label: 'Milvus Config (JSON)', type: 'json', placeholder: '{"address":"","collection":"agentic_memory","dimension":384}' },
        { name: 'embedding', label: 'Embedding Config (JSON)', type: 'json', placeholder: '{"model":"all-MiniLM-L6-v2","dimension":384}' },
        { name: 'default_retrieval_limit', label: 'Default Retrieval Limit', type: 'number', placeholder: '5' },
        { name: 'default_similarity_threshold', label: 'Similarity Threshold', type: 'percentage', placeholder: '70' },
        { name: 'extraction_batch_size', label: 'Extraction Batch Size', type: 'number', placeholder: '10' },
      ]
    case 'semantic_cache':
      return [
        { name: 'enabled', label: 'Enable Semantic Cache', type: 'boolean' },
        { name: 'backend_type', label: 'Backend Type', type: 'select', options: ['memory', 'milvus', 'hybrid', 'redis', 'memcached'], required: true },
        { name: 'similarity_threshold', label: 'Similarity Threshold', type: 'percentage', placeholder: '80' },
        { name: 'max_entries', label: 'Max Entries', type: 'number', placeholder: '1000' },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '3600' },
        { name: 'eviction_policy', label: 'Eviction Policy', type: 'select', options: ['fifo', 'lru', 'lfu'] },
        { name: 'use_hnsw', label: 'Enable HNSW', type: 'boolean' },
        { name: 'hnsw_m', label: 'HNSW M', type: 'number', placeholder: '16' },
        { name: 'hnsw_ef_construction', label: 'HNSW EF Construction', type: 'number', placeholder: '200' },
        { name: 'embedding_model', label: 'Embedding Model Override', type: 'text', placeholder: 'mmbert' },
        { name: 'max_memory_entries', label: 'Hybrid Max Memory Entries', type: 'number', placeholder: '100000' },
        { name: 'backend_config_path', label: 'Backend Config Path', type: 'text', placeholder: 'config/milvus.yaml' },
      ]
    case 'tools':
      return [
        { name: 'enabled', label: 'Enable Tool Auto Selection', type: 'boolean' },
        { name: 'top_k', label: 'Top K', type: 'number', placeholder: '3' },
        { name: 'similarity_threshold', label: 'Similarity Threshold', type: 'percentage', placeholder: '20' },
        { name: 'tools_db_path', label: 'Tools DB Path', type: 'text', placeholder: 'config/tools_db.json' },
        { name: 'fallback_to_empty', label: 'Fallback To Empty', type: 'boolean' },
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
        { name: 'category_model', label: 'Domain Model (JSON)', type: 'json' },
        { name: 'pii_model', label: 'PII Model (JSON)', type: 'json' },
        { name: 'mcp_category_model', label: 'MCP Model (JSON)', type: 'json' },
        { name: 'preference_model', label: 'Preference Model (JSON)', type: 'json' },
      ]
    case 'hallucination_mitigation':
      return [
        { name: 'enabled', label: 'Enable Hallucination Mitigation', type: 'boolean' },
        { name: 'on_hallucination_detected', label: 'On Detection Action', type: 'text', placeholder: 'block' },
        { name: 'fact_check_model', label: 'Fact Check Module (JSON)', type: 'json' },
        { name: 'hallucination_model', label: 'Detector Module (JSON)', type: 'json' },
        { name: 'nli_model', label: 'Explainer Module (JSON)', type: 'json' },
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
    case 'embedding_models':
      return [
        { name: 'qwen3_model_path', label: 'Qwen3 Model Path', type: 'text', placeholder: 'models/mom-embedding-pro' },
        { name: 'gemma_model_path', label: 'Gemma Model Path', type: 'text', placeholder: 'models/mom-embedding-flash' },
        { name: 'mmbert_model_path', label: 'mmBERT Model Path', type: 'text', placeholder: 'models/mom-embedding-ultra' },
        { name: 'multimodal_model_path', label: 'Multimodal Model Path', type: 'text', placeholder: 'models/mom-embedding-multimodal' },
        { name: 'use_cpu', label: 'Use CPU', type: 'boolean' },
        { name: 'hnsw_config', label: 'HNSW Config (JSON)', type: 'json' },
        { name: 'bert', label: 'BERT Catalog Entry (JSON)', type: 'json' },
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
      return [{ name: 'batch_classification', label: 'Batch Classification (JSON)', type: 'json' }]
    case 'bert_model':
      return [
        { name: 'model_id', label: 'Model ID', type: 'text', required: true, placeholder: 'sentence-transformers/all-MiniLM-L6-v2' },
        { name: 'threshold', label: 'Similarity Threshold', type: 'percentage', placeholder: '80' },
        { name: 'use_cpu', label: 'Use CPU', type: 'boolean' },
      ]
  }

  return []
}

function editDataForKey(key: RouterSystemKey, data: unknown): Record<string, unknown> {
  if (key === 'clear_route_cache') {
    return { value: Boolean(data) }
  }
  if (key === 'external_models') {
    return { items: Array.isArray(data) ? data : cloneDefaultSection(key) }
  }
  if (key === 'classifier') {
    const classifier = asObject(data)
    return {
      category_model: asObject(classifier?.domain) || {},
      pii_model: asObject(classifier?.pii) || {},
      mcp_category_model: asObject(classifier?.mcp) || {},
      preference_model: asObject(classifier?.preference) || {},
    }
  }
  if (key === 'hallucination_mitigation') {
    const hallucination = asObject(data)
    return {
      enabled: hallucination?.enabled,
      on_hallucination_detected: hallucination?.on_hallucination_detected,
      fact_check_model: asObject(hallucination?.fact_check) || {},
      hallucination_model: asObject(hallucination?.detector) || {},
      nli_model: asObject(hallucination?.explainer) || {},
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

function saveForKey(key: RouterSystemKey, data: Record<string, unknown>): Partial<ConfigData> {
  if (key === 'clear_route_cache') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], Boolean(data.value)) as Partial<ConfigData>
  }
  if (key === 'external_models') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], Array.isArray(data.items) ? data.items : []) as Partial<ConfigData>
  }
  if (key === 'classifier') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      domain: asObject(data.category_model) || {},
      pii: asObject(data.pii_model) || {},
      mcp: asObject(data.mcp_category_model) || {},
      preference: asObject(data.preference_model) || {},
    }) as Partial<ConfigData>
  }
  if (key === 'hallucination_mitigation') {
    return buildNestedPatch(GLOBAL_SECTION_PATHS[key], {
      enabled: Boolean(data.enabled),
      on_hallucination_detected: data.on_hallucination_detected,
      fact_check: asObject(data.fact_check_model) || {},
      detector: asObject(data.hallucination_model) || {},
      explainer: asObject(data.nli_model) || {},
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
    response_api: getSectionValue(routerDefaults, 'response_api') ?? getSectionValue(config, 'response_api'),
    router_replay: getSectionValue(routerDefaults, 'router_replay') ?? getSectionValue(config, 'router_replay'),
    memory: getSectionValue(routerDefaults, 'memory') ?? getSectionValue(config, 'memory'),
    semantic_cache: getSectionValue(routerDefaults, 'semantic_cache') ?? getSectionValue(config, 'semantic_cache'),
    tools: getSectionValue(routerDefaults, 'tools') ?? getSectionValue(config, 'tools'),
    prompt_guard: getSectionValue(routerDefaults, 'prompt_guard') ?? getSectionValue(config, 'prompt_guard'),
    classifier: getSectionValue(routerDefaults, 'classifier') ?? getSectionValue(config, 'classifier'),
    hallucination_mitigation: getSectionValue(routerDefaults, 'hallucination_mitigation') ?? getSectionValue(config, 'hallucination_mitigation'),
    feedback_detector: getSectionValue(routerDefaults, 'feedback_detector') ?? getSectionValue(config, 'feedback_detector'),
    external_models: getSectionValue(routerDefaults, 'external_models') ?? getSectionValue(config, 'external_models'),
    embedding_models: getSectionValue(routerDefaults, 'embedding_models') ?? getSectionValue(config, 'embedding_models'),
    observability: getSectionValue(routerDefaults, 'observability') ?? getSectionValue(config, 'observability'),
    looper: getSectionValue(routerDefaults, 'looper') ?? getSectionValue(config, 'looper'),
    clear_route_cache: getSectionValue(routerDefaults, 'clear_route_cache') ?? getSectionValue(config, 'clear_route_cache'),
    model_selection: getSectionValue(routerDefaults, 'model_selection') ?? getSectionValue(config, 'model_selection'),
    api: getSectionValue(routerDefaults, 'api') ?? getSectionValue(config, 'api'),
    bert_model: getSectionValue(routerDefaults, 'bert_model') ?? getSectionValue(config, 'bert_model'),
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
