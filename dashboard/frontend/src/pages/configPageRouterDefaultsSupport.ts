import type { FieldConfig } from '../components/EditModal'
import {
  DEFAULT_SECTIONS,
  OPTIONAL_ROUTER_KEYS,
  PYTHON_ROUTER_KEYS,
  SECTION_META,
} from './configPageRouterDefaultsCatalog'
import type {
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

export type RouterConfigSectionData = Partial<
  Pick<
    ConfigData,
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
  >
>

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
  routerDefaults: ConfigData | null
  toolsData: Tool[]
  toolsLoading: boolean
  toolsError: string | null
}

function cloneDefaultSection(key: RouterSystemKey): unknown {
  return JSON.parse(JSON.stringify(DEFAULT_SECTIONS[key]))
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

function sourceBadge(key: RouterSystemKey, routerDefaults: ConfigData | null, data: unknown): { label: string; tone: 'active' | 'inactive' | 'info' } {
  if (routerDefaults && routerDefaults[key] !== undefined) {
    return { label: '.vllm-sr/router-defaults.yaml', tone: 'active' }
  }
  if (data !== undefined) {
    return { label: 'config.yaml fallback', tone: 'info' }
  }
  return { label: 'Template section', tone: 'inactive' }
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
        { label: 'Model ID', value: stringOrFallback(section?.model_id) },
        { label: 'Threshold', value: percentOrFallback(section?.threshold) },
        { label: 'Runtime', value: section?.use_cpu ? 'CPU' : 'GPU' },
      ]
    case 'classifier': {
      const classifier = section
      return [
        { label: 'Category model', value: stringOrFallback(asObject(classifier?.category_model)?.model_id) },
        { label: 'PII model', value: stringOrFallback(asObject(classifier?.pii_model)?.model_id) },
        { label: 'Preference mode', value: asObject(classifier?.preference_model)?.use_contrastive ? 'Contrastive' : 'External / unset' },
      ]
    }
    case 'hallucination_mitigation':
      return [
        { label: 'Fact-check model', value: stringOrFallback(asObject(section?.fact_check_model)?.model_id) },
        { label: 'Detector model', value: stringOrFallback(asObject(section?.hallucination_model)?.model_id) },
        { label: 'Explainer model', value: stringOrFallback(asObject(section?.nli_model)?.model_id) },
      ]
    case 'feedback_detector':
      return [
        { label: 'Model ID', value: stringOrFallback(section?.model_id) },
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
      return [
        { label: 'Primary path', value: stringOrFallback(section?.mmbert_model_path ?? section?.qwen3_model_path ?? section?.gemma_model_path ?? section?.multimodal_model_path) },
        { label: 'Runtime', value: section?.use_cpu ? 'CPU' : 'GPU' },
        { label: 'HNSW model type', value: stringOrFallback(asObject(section?.hnsw_config)?.model_type) },
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
      return [
        { label: 'Default algorithm', value: stringOrFallback(section?.default_algorithm) },
        { label: 'Models path', value: stringOrFallback(section?.models_path) },
        { label: 'Custom training', value: asObject(section?.custom_training)?.enabled ? 'Enabled' : 'Disabled' },
      ]
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
        { name: 'model_id', label: 'Model ID', type: 'text', required: true, placeholder: 'models/mmbert32k-jailbreak-detector-merged' },
        { name: 'threshold', label: 'Threshold', type: 'percentage', placeholder: '70' },
        { name: 'use_cpu', label: 'Use CPU', type: 'boolean' },
        { name: 'use_mmbert_32k', label: 'Use mmBERT 32K', type: 'boolean' },
        { name: 'use_modernbert', label: 'Use ModernBERT', type: 'boolean' },
        { name: 'jailbreak_mapping_path', label: 'Mapping Path', type: 'text', placeholder: 'models/.../jailbreak_type_mapping.json' },
      ]
    case 'classifier':
      return [
        { name: 'category_model', label: 'Category Model (JSON)', type: 'json' },
        { name: 'pii_model', label: 'PII Model (JSON)', type: 'json' },
        { name: 'mcp_category_model', label: 'MCP Category Model (JSON)', type: 'json' },
        { name: 'preference_model', label: 'Preference Model (JSON)', type: 'json' },
      ]
    case 'hallucination_mitigation':
      return [
        { name: 'enabled', label: 'Enable Hallucination Mitigation', type: 'boolean' },
        { name: 'fact_check_model', label: 'Fact Check Model (JSON)', type: 'json' },
        { name: 'hallucination_model', label: 'Hallucination Model (JSON)', type: 'json' },
        { name: 'nli_model', label: 'NLI Model (JSON)', type: 'json' },
      ]
    case 'feedback_detector':
      return [
        { name: 'enabled', label: 'Enable Feedback Detector', type: 'boolean' },
        { name: 'model_id', label: 'Model ID', type: 'text', required: true, placeholder: 'models/mmbert32k-feedback-detector-merged' },
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
        { name: 'default_algorithm', label: 'Default Algorithm', type: 'select', options: ['knn', 'kmeans', 'svm'], required: true },
        { name: 'llm_candidates_path', label: 'LLM Candidates Path', type: 'text', placeholder: 'config/model_selection/llm_candidates.json' },
        { name: 'models_path', label: 'Models Path', type: 'text', placeholder: 'models/model_selection' },
        { name: 'training_data_path', label: 'Training Data Path', type: 'text', placeholder: 'config/model_selection/routing_training_data.jsonl' },
        { name: 'knn', label: 'KNN Config (JSON)', type: 'json' },
        { name: 'kmeans', label: 'KMeans Config (JSON)', type: 'json' },
        { name: 'svm', label: 'SVM Config (JSON)', type: 'json' },
        { name: 'custom_training', label: 'Custom Training (JSON)', type: 'json' },
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
  const objectData = asObject(data)
  return objectData ? { ...objectData } : asObject(cloneDefaultSection(key)) || {}
}

function saveForKey(key: RouterSystemKey, data: Record<string, unknown>): Partial<ConfigData> {
  if (key === 'clear_route_cache') {
    return { clear_route_cache: Boolean(data.value) }
  }
  if (key === 'external_models') {
    return { external_models: Array.isArray(data.items) ? data.items : [] }
  }
  return { [key]: data } as Partial<ConfigData>
}

export function buildEffectiveRouterConfig(
  routerDefaults: ConfigData | null,
  config: ConfigData | null,
): RouterConfigSectionData {
  return {
    response_api: routerDefaults?.response_api ?? config?.response_api,
    router_replay: routerDefaults?.router_replay ?? config?.router_replay,
    memory: routerDefaults?.memory ?? config?.memory,
    semantic_cache: routerDefaults?.semantic_cache ?? config?.semantic_cache,
    tools: routerDefaults?.tools ?? config?.tools,
    prompt_guard: routerDefaults?.prompt_guard ?? config?.prompt_guard,
    classifier: routerDefaults?.classifier ?? config?.classifier,
    hallucination_mitigation: routerDefaults?.hallucination_mitigation ?? config?.hallucination_mitigation,
    feedback_detector: routerDefaults?.feedback_detector ?? config?.feedback_detector,
    external_models: routerDefaults?.external_models ?? config?.external_models,
    embedding_models: routerDefaults?.embedding_models ?? config?.embedding_models,
    observability: routerDefaults?.observability ?? config?.observability,
    looper: routerDefaults?.looper ?? config?.looper,
    clear_route_cache: routerDefaults?.clear_route_cache ?? config?.clear_route_cache,
    model_selection: routerDefaults?.model_selection ?? config?.model_selection,
    api: routerDefaults?.api ?? config?.api,
    bert_model: routerDefaults?.bert_model ?? config?.bert_model,
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
