import type { Endpoint } from '../components/EndpointsEditor'
import type { DecisionConditionType } from '../types/config'

export interface VLLMEndpoint {
  name: string
  address: string
  port: number
  weight: number
  health_check_path: string
  protocol?: 'http' | 'https'
  provider_profile?: string
  type?: string
  api_key?: string
  api_key_env?: string
}

export interface ModelConfig {
  model_id: string
  use_modernbert?: boolean
  use_mmbert_32k?: boolean
  threshold: number
  use_cpu: boolean
  use_contrastive?: boolean
  embedding_model?: string
  category_mapping_path?: string
  pii_mapping_path?: string
  jailbreak_mapping_path?: string
}

export interface MCPCategoryModel {
  enabled: boolean
  transport_type: string
  command?: string
  args?: string[]
  env?: Record<string, string>
  url?: string
  tool_name?: string
  threshold: number
  timeout_seconds?: number
}

export interface ModelScore {
  model: string
  score: number
  use_reasoning: boolean
  reasoning_description?: string
  reasoning_effort?: string
}

export interface Category {
  name: string
  system_prompt?: string
  description?: string
  mmlu_categories?: string[]
  model_scores?: ModelScore[] | Record<string, number>
}

export interface ToolFunction {
  name: string
  description: string
  parameters: {
    type: string
    properties: Record<string, ToolParameterSchema>
    required?: string[]
  }
}

export interface ToolParameterSchema {
  type?: string
  description?: string
  [key: string]: unknown
}

export interface Tool {
  tool: {
    type: string
    function: ToolFunction
  }
  description: string
  category?: string
  tags?: string[]
}

export interface ReasoningFamily {
  type: string
  parameter: string
}

export interface ModelPricing {
  currency?: string
  prompt_per_1m?: number
  completion_per_1m?: number
}

export interface LoRAAdapter {
  name: string
  description?: string
}

export interface ModelConfigEntry {
  model_id?: string
  reasoning_family?: string
  preferred_endpoints?: string[]
  access_key?: string
  pricing?: ModelPricing
  api_format?: string
  external_model_ids?: Record<string, string>
  param_size?: string
  context_window_size?: number
  description?: string
  capabilities?: string[]
  loras?: LoRAAdapter[]
  tags?: string[]
  quality_score?: number
  modality?: string
}

export interface BackendRefEntry {
  name?: string
  endpoint?: string
  protocol?: 'http' | 'https'
  weight?: number
  type?: string
  base_url?: string
  provider?: string
  auth_header?: string
  auth_prefix?: string
  extra_headers?: Record<string, string>
  api_version?: string
  chat_path?: string
  api_key?: string
  api_key_env?: string
}

export interface ProviderModelConfig {
  name: string
  reasoning_family?: string
  provider_model_id?: string
  api_format?: string
  external_model_ids?: Record<string, string>
  backend_refs?: BackendRefEntry[]
  endpoints?: Array<{
    name: string
    weight: number
    endpoint: string
    protocol: 'http' | 'https'
  }>
  access_key?: string
  pricing?: ModelPricing
}

export interface ProviderDefaultsConfig {
  default_model?: string
  reasoning_families?: Record<string, ReasoningFamily>
  default_reasoning_effort?: string
}

export interface ProvidersConfig {
  defaults?: ProviderDefaultsConfig
  models: ProviderModelConfig[]
}

export interface RoutingModelCard {
  name: string
  param_size?: string
  context_window_size?: number
  description?: string
  capabilities?: string[]
  loras?: LoRAAdapter[]
  tags?: string[]
  quality_score?: number
  modality?: string
}

export interface DecisionCondition {
  type: string
  name: string
}

export interface DecisionRuleSet {
  operator: 'AND' | 'OR' | 'NOT'
  conditions: DecisionCondition[]
}

export interface DecisionModelRef {
  model: string
  use_reasoning: boolean
  reasoning_description?: string
  reasoning_effort?: string
  lora_name?: string
  weight?: number
}

export interface DecisionPluginConfig {
  type: string
  configuration: Record<string, unknown>
}

export interface DecisionConfig {
  name: string
  description: string
  priority: number
  rules: DecisionRuleSet
  modelRefs: DecisionModelRef[]
  plugins?: DecisionPluginConfig[]
}

export interface RoutingConfig {
  modelCards?: RoutingModelCard[]
  signals?: ConfigData['signals']
  decisions?: DecisionConfig[]
}

export interface NormalizedModel {
  name: string
  reasoning_family?: string
  provider_model_id?: string
  api_format?: string
  external_model_ids?: Record<string, string>
  backend_refs?: BackendRefEntry[]
  endpoints: Endpoint[]
  param_size?: string
  context_window_size?: number
  description?: string
  capabilities?: string[]
  loras?: LoRAAdapter[]
  tags?: string[]
  quality_score?: number
  modality?: string
  pricing?: {
    currency?: string
    prompt_per_1m?: number
    completion_per_1m?: number
  }
}

export interface TracingConfig {
  enabled: boolean
  provider: string
  exporter: {
    type: string
    endpoint?: string
    insecure?: boolean
  }
  sampling: {
    type: string
    rate?: number
  }
  resource: {
    service_name: string
    service_version: string
    deployment_environment: string
  }
}

export interface APIConfig {
  batch_classification?: {
    max_batch_size: number
    concurrency_threshold: number
    max_concurrency: number
    metrics?: {
      enabled: boolean
      detailed_goroutine_tracking?: boolean
      high_resolution_timing?: boolean
      sample_rate?: number
      duration_buckets?: number[]
      size_buckets?: number[]
    }
  }
}

export interface ResponseAPIConfig {
  enabled?: boolean
  store_backend?: string
  ttl_seconds?: number
  max_responses?: number
}

export interface RouterReplayConfig {
  store_backend?: string
  ttl_seconds?: number
  async_writes?: boolean
}

export interface MemoryMilvusConfig {
  address?: string
  collection?: string
  dimension?: number
  num_partitions?: number
}

export interface MemoryEmbeddingConfig {
  model?: string
  dimension?: number
}

export interface MemoryConfig {
  enabled?: boolean
  auto_store?: boolean
  milvus?: MemoryMilvusConfig
  embedding?: MemoryEmbeddingConfig
  default_retrieval_limit?: number
  default_similarity_threshold?: number
  extraction_batch_size?: number
}

export interface SemanticCacheConfig {
  enabled?: boolean
  backend_type?: string
  similarity_threshold?: number
  max_entries?: number
  ttl_seconds?: number
  eviction_policy?: string
  use_hnsw?: boolean
  hnsw_m?: number
  hnsw_ef_construction?: number
  embedding_model?: string
  max_memory_entries?: number
}

export interface FactCheckModelModuleConfig {
  model_id?: string
  model_ref?: string
  threshold?: number
  use_cpu?: boolean
  use_mmbert_32k?: boolean
}

export interface HallucinationDetectorModuleConfig {
  model_id?: string
  model_ref?: string
  threshold?: number
  use_cpu?: boolean
  min_span_length?: number
  min_span_confidence?: number
  context_window_size?: number
  enable_nli_filtering?: boolean
  nli_entailment_threshold?: number
}

export interface NLIExplainerModuleConfig {
  model_id?: string
  model_ref?: string
  threshold?: number
  use_cpu?: boolean
}

export interface HallucinationMitigationConfig {
  enabled?: boolean
  fact_check_model?: FactCheckModelModuleConfig
  hallucination_model?: HallucinationDetectorModuleConfig
  nli_model?: NLIExplainerModuleConfig
}

export interface FeedbackDetectorConfig {
  enabled?: boolean
  model_id?: string
  threshold?: number
  use_cpu?: boolean
  use_mmbert_32k?: boolean
  use_modernbert?: boolean
}

export interface EmbeddingOptimizationConfig {
  model_type?: string
  preload_embeddings?: boolean
  target_dimension?: number
  target_layer?: number
  enable_soft_matching?: boolean
  min_score_threshold?: number
}

export interface EmbeddingModelsConfig {
  qwen3_model_path?: string
  gemma_model_path?: string
  mmbert_model_path?: string
  multimodal_model_path?: string
  use_cpu?: boolean
  embedding_config?: EmbeddingOptimizationConfig
}

export interface ObservabilityConfig {
  tracing?: TracingConfig
  metrics?: { enabled?: boolean }
}

export interface LooperConfig {
  enabled?: boolean
  endpoint?: string
  timeout_seconds?: number
  headers?: Record<string, string>
}

export interface StreamedBodyConfig {
  enabled?: boolean
  max_bytes?: number
  timeout_sec?: number
}

export interface IdentityConfig {
  user_id_header?: string
  user_groups_header?: string
}

export interface AuthzProviderConfig {
  type: string
  headers?: Record<string, string>
}

export interface AuthzConfig {
  fail_open?: boolean
  identity?: IdentityConfig
  providers?: AuthzProviderConfig[]
}

export interface RateLimitMatch {
  user?: string
  group?: string
  model?: string
}

export interface RateLimitRule {
  name: string
  match: RateLimitMatch
  requests_per_unit?: number
  tokens_per_unit?: number
  unit: string
}

export interface RateLimitProviderConfig {
  type: string
  address?: string
  domain?: string
  rules?: RateLimitRule[]
}

export interface RateLimitConfig {
  fail_open?: boolean
  providers?: RateLimitProviderConfig[]
}

export interface VectorStoreMemoryConfig {
  max_entries_per_store?: number
}

export interface LlamaStackVectorStoreConfig {
  endpoint: string
  auth_token?: string
  embedding_model?: string
  request_timeout_seconds?: number
  search_type?: string
}

export interface VectorStoreConfig {
  enabled?: boolean
  backend_type?: string
  file_storage_dir?: string
  max_file_size_mb?: number
  embedding_model?: string
  embedding_dimension?: number
  ingestion_workers?: number
  supported_formats?: string[]
  milvus?: MilvusConfig
  memory?: VectorStoreMemoryConfig
  llama_stack?: LlamaStackVectorStoreConfig
}

export interface PromptCompressionConfig {
  enabled?: boolean
  max_tokens?: number
  min_length?: number
  skip_signals?: string[]
  textrank_weight?: number
  position_weight?: number
  tfidf_weight?: number
  position_depth?: number
}

export interface ModalityClassifierConfig {
  model_path?: string
  use_cpu?: boolean
}

export interface ModalityDetectionConfig {
  method?: string
  classifier?: ModalityClassifierConfig
  keywords?: string[]
  both_keywords?: string[]
  confidence_threshold?: number
  lower_threshold_ratio?: number
}

export interface ModalityDetectorConfig {
  enabled?: boolean
  prompt_prefixes?: string[]
  method?: string
  classifier?: ModalityClassifierConfig
  keywords?: string[]
  both_keywords?: string[]
  confidence_threshold?: number
  lower_threshold_ratio?: number
}

export interface ExternalModelEndpointConfig {
  address?: string
  port?: number
  protocol?: string
  name?: string
  use_chat_template?: boolean
  prompt_template?: string
}

export interface ExternalModelConfig {
  llm_provider: string
  model_role: string
  llm_endpoint?: ExternalModelEndpointConfig
  llm_model_name?: string
  llm_timeout_seconds?: number
  parser_type?: string
  threshold?: number
  access_key?: string
  max_tokens?: number
  temperature?: number
}

export interface ToolFilteringWeights {
  embed?: number
  lexical?: number
  tag?: number
  name?: number
  category?: number
}

export interface AdvancedToolFilteringConfig {
  enabled?: boolean
  candidate_pool_size?: number
  min_lexical_overlap?: number
  min_combined_score?: number
  weights?: ToolFilteringWeights
  use_category_filter?: boolean
  category_confidence_threshold?: number
  allow_tools?: string[]
  block_tools?: string[]
}

export interface CanonicalSystemModels {
  prompt_guard?: string
  domain_classifier?: string
  pii_classifier?: string
  fact_check_classifier?: string
  hallucination_detector?: string
  hallucination_explainer?: string
  feedback_detector?: string
}

export interface ModelSelectionConfig {
  enabled?: boolean
  method?: string
  elo?: {
    initial_rating?: number
    k_factor?: number
    category_weighted?: boolean
    decay_factor?: number
    min_comparisons?: number
    cost_scaling_factor?: number
    storage_path?: string
    auto_save_interval?: string
  }
  router_dc?: {
    temperature?: number
    dimension_size?: number
    min_similarity?: number
    use_query_contrastive?: boolean
    use_model_contrastive?: boolean
    require_descriptions?: boolean
    use_capabilities?: boolean
  }
  automix?: {
    verification_threshold?: number
    max_escalations?: number
    cost_aware_routing?: boolean
    cost_quality_tradeoff?: number
    discount_factor?: number
    use_logprob_verification?: boolean
  }
  hybrid?: {
    elo_weight?: number
    router_dc_weight?: number
    automix_weight?: number
    cost_weight?: number
    quality_gap_threshold?: number
    normalize_scores?: boolean
  }
  ml?: {
    models_path?: string
    embedding_dim?: number
    knn?: { k?: number; pretrained_path?: string }
    kmeans?: { num_clusters?: number; efficiency_weight?: number; pretrained_path?: string }
    svm?: { kernel?: string; gamma?: number; pretrained_path?: string }
    mlp?: { device?: string; pretrained_path?: string }
  }
}

export interface CanonicalClassifierConfig {
  domain?: (ModelConfig & { model_ref?: string; fallback_category?: string })
  mcp?: MCPCategoryModel
  pii?: (ModelConfig & { model_ref?: string })
  preference?: {
    use_contrastive?: boolean
    embedding_model?: string
  }
}

export interface CanonicalHallucinationModuleConfig {
  enabled?: boolean
  on_hallucination_detected?: string
  fact_check?: FactCheckModelModuleConfig
  detector?: HallucinationDetectorModuleConfig
  explainer?: NLIExplainerModuleConfig
}

export interface CanonicalEmbeddingCatalogConfig {
  semantic?: EmbeddingModelsConfig
  bert?: ModelConfig
}

export interface CanonicalGlobalConfig {
  router?: {
    config_source?: string
    strategy?: string
    auto_model_name?: string
    include_config_models_in_list?: boolean
    clear_route_cache?: boolean
    streamed_body?: StreamedBodyConfig
    model_selection?: ModelSelectionConfig
  }
  services?: {
    api?: APIConfig
    response_api?: ResponseAPIConfig
    observability?: ObservabilityConfig
    authz?: AuthzConfig
    ratelimit?: RateLimitConfig
    router_replay?: RouterReplayConfig
  }
  stores?: {
    semantic_cache?: SemanticCacheConfig
    memory?: MemoryConfig
    vector_store?: VectorStoreConfig
  }
  integrations?: {
    tools?: {
      enabled?: boolean
      top_k?: number
      similarity_threshold?: number
      tools_db_path?: string
      fallback_to_empty?: boolean
      advanced_filtering?: AdvancedToolFilteringConfig
    }
    looper?: LooperConfig
  }
  model_catalog?: {
    embeddings?: CanonicalEmbeddingCatalogConfig
    system?: CanonicalSystemModels
    external?: ExternalModelConfig[]
    modules?: {
      prompt_compression?: PromptCompressionConfig
      prompt_guard?: (ModelConfig & { enabled?: boolean; model_ref?: string; use_vllm?: boolean })
      classifier?: CanonicalClassifierConfig
      hallucination_mitigation?: CanonicalHallucinationModuleConfig
      feedback_detector?: (FeedbackDetectorConfig & { model_ref?: string })
      modality_detector?: ModalityDetectorConfig
    }
  }
}

export interface KeywordSignal {
  name: string
  operator: 'AND' | 'OR'
  keywords: string[]
  case_sensitive: boolean
}

export interface EmbeddingSignal {
  name: string
  threshold: number
  candidates: string[]
  aggregation_method: string
}

export interface DomainSignal {
  name: string
  description: string
  mmlu_categories?: string[]
}

export interface ModalitySignal {
  name: string
  description?: string
}

export interface Subject {
  kind: 'User' | 'Group'
  name: string
}

export interface RoleBindingSignal {
  name: string
  role: string
  subjects: Subject[]
  description?: string
}

export interface FactCheckSignal {
  name: string
  description: string
}

export interface UserFeedbackSignal {
  name: string
  description: string
}

export interface PreferenceSignal {
  name: string
  description: string
  examples?: string[]
  threshold?: number
}

export interface LanguageSignal {
  name: string
  description?: string
}

export interface ContextSignal {
  name: string
  min_tokens: string
  max_tokens: string
  description?: string
}

export interface ComplexitySignal {
  name: string
  threshold: number
  hard: { candidates: string[] }
  easy: { candidates: string[] }
  description?: string
  composer?: {
    operator: 'AND' | 'OR' | 'NOT'
    conditions: Array<{ type: string; name: string }>
  }
}

export interface JailbreakSignal {
  name: string
  threshold?: number
  method?: string
  include_history?: boolean
  jailbreak_patterns?: string[]
  benign_patterns?: string[]
  description?: string
}

export interface PIISignal {
  name: string
  threshold?: number
  pii_types_allowed?: string[]
  include_history?: boolean
  description?: string
}

export interface ConfigData {
  version?: string
  listeners?: Array<{
    name: string
    address: string
    port: number
    timeout?: string
  }>
  signals?: {
    keywords?: KeywordSignal[]
    embeddings?: EmbeddingSignal[]
    domains?: DomainSignal[]
    fact_check?: FactCheckSignal[]
    user_feedbacks?: UserFeedbackSignal[]
    preferences?: PreferenceSignal[]
    language?: LanguageSignal[]
    context?: ContextSignal[]
    complexity?: ComplexitySignal[]
    modality?: ModalitySignal[]
    role_bindings?: RoleBindingSignal[]
    jailbreak?: JailbreakSignal[]
    pii?: PIISignal[]
  }
  decisions?: DecisionConfig[]
  providers?: ProvidersConfig
  routing?: RoutingConfig
  global?: CanonicalGlobalConfig
  bert_model?: ModelConfig
  semantic_cache?: SemanticCacheConfig
  tools?: {
    enabled: boolean
    top_k: number
    similarity_threshold: number
    tools_db_path: string
    fallback_to_empty: boolean
  }
  prompt_guard?: ModelConfig & { enabled: boolean }
  vllm_endpoints?: VLLMEndpoint[]
  classifier?: {
    category_model?: ModelConfig
    mcp_category_model?: MCPCategoryModel
    pii_model?: ModelConfig
    preference_model?: ModelConfig
  }
  categories?: (Category & { mmlu_categories?: string[] })[]
  default_reasoning_effort?: string
  default_model?: string
  model_config?: Record<string, ModelConfigEntry>
  reasoning_families?: Record<string, ReasoningFamily>
  response_api?: ResponseAPIConfig
  router_replay?: RouterReplayConfig
  memory?: MemoryConfig
  hallucination_mitigation?: HallucinationMitigationConfig
  feedback_detector?: FeedbackDetectorConfig
  external_models?: Array<Record<string, unknown>>
  embedding_models?: EmbeddingModelsConfig
  api?: APIConfig
  observability?: ObservabilityConfig
  looper?: LooperConfig
  clear_route_cache?: boolean
  model_selection?: ModelSelectionConfig
  keyword_rules?: KeywordSignal[]
  embedding_rules?: EmbeddingSignal[]
  fact_check_rules?: FactCheckSignal[]
  user_feedback_rules?: UserFeedbackSignal[]
  preference_rules?: PreferenceSignal[]
  language_rules?: LanguageSignal[]
  context_rules?: ContextSignal[]
  complexity_rules?: ComplexitySignal[]
  jailbreak?: JailbreakSignal[]
  pii?: PIISignal[]
  [key: string]: unknown
}

export type SignalType =
  | 'Keywords'
  | 'Embeddings'
  | 'Domain'
  | 'Preference'
  | 'Fact Check'
  | 'User Feedback'
  | 'Language'
  | 'Context'
  | 'Complexity'
  | 'Modality'
  | 'Authz'
  | 'Jailbreak'
  | 'PII'

export interface DecisionFormState {
  name: string
  description: string
  priority: number
  operator: 'AND' | 'OR' | 'NOT'
  conditions: DecisionCondition[]
  modelRefs: DecisionModelRef[]
  plugins: { type: string; configuration: string | Record<string, unknown> }[]
}

export interface AddSignalFormState {
  type: SignalType
  name: string
  description: string
  operator: 'AND' | 'OR'
  keywords: string
  case_sensitive: boolean
  threshold: number
  candidates: string
  aggregation_method: string
  mmlu_categories: string
  min_tokens?: string
  max_tokens?: string
  preference_examples?: string
  preference_threshold?: number
  complexity_threshold?: number
  role?: string
  subjects?: string
  hard_candidates?: string
  easy_candidates?: string
  composer_operator?: 'AND' | 'OR' | 'NOT'
  composer_conditions?: string
  jailbreak_threshold?: number
  jailbreak_method?: string
  include_history?: boolean
  jailbreak_patterns?: string
  benign_patterns?: string
  pii_threshold?: number
  pii_types_allowed?: string
  pii_include_history?: boolean
}

export const formatThreshold = (value: number): string => {
  return `${Math.round(value * 100)}%`
}

export const normalizeModelScores = (
  modelScores: ModelScore[] | Record<string, number> | undefined
): ModelScore[] => {
  if (!modelScores) return []
  if (Array.isArray(modelScores)) return modelScores
  return Object.entries(modelScores).map(([model, score]) => ({
    model,
    score: typeof score === 'number' ? score : 0,
    use_reasoning: false,
  }))
}

export const normalizeEndpointProtocol = (protocol: unknown): Endpoint['protocol'] =>
  protocol === 'https' ? 'https' : 'http'

export const normalizeEndpoint = (
  endpoint: Partial<Endpoint> | undefined,
  index: number
): Endpoint => ({
  name: endpoint?.name?.trim() || `endpoint-${index + 1}`,
  endpoint: endpoint?.endpoint?.trim() || '',
  protocol: normalizeEndpointProtocol(endpoint?.protocol),
  weight:
    typeof endpoint?.weight === 'number' && Number.isFinite(endpoint.weight) ? endpoint.weight : 1,
})

export const normalizeEndpoints = (endpoints: Partial<Endpoint>[] | undefined): Endpoint[] =>
  Array.isArray(endpoints) ? endpoints.map((endpoint, index) => normalizeEndpoint(endpoint, index)) : []

export const normalizeProviderModelEndpoints = (
  model: {
    endpoints?: Partial<Endpoint>[]
    backend_refs?: BackendRefEntry[]
  }
): Endpoint[] => {
  if (Array.isArray(model.backend_refs) && model.backend_refs.length > 0) {
    return model.backend_refs.map((backend, index) => {
      const baseURL = typeof backend.base_url === 'string' ? backend.base_url.trim() : ''
      const endpoint =
        typeof backend.endpoint === 'string' && backend.endpoint.trim()
          ? backend.endpoint.trim()
          : baseURL
      const protocol =
        backend.protocol ||
        (baseURL.startsWith('https://') ? 'https' : 'http')
      return normalizeEndpoint({
        name: backend.name,
        endpoint,
        protocol,
        weight: backend.weight,
      }, index)
    })
  }
  return normalizeEndpoints(model.endpoints)
}

export const mergeProviderBackendRefs = (
  existingRefs: BackendRefEntry[] | undefined,
  endpoints: Endpoint[],
  accessKey?: string
): BackendRefEntry[] => {
  const existing = Array.isArray(existingRefs) ? existingRefs : []

  return endpoints.map((ep, index) => {
    const matched =
      existing.find((ref) => ref.name === ep.name) ||
      existing[index]

    const merged: BackendRefEntry = {
      ...(matched || {}),
      name: ep.name,
      protocol: ep.protocol,
      weight: ep.weight,
    }

    const matchedDisplayEndpoint =
      typeof matched?.endpoint === 'string' && matched.endpoint.trim()
        ? matched.endpoint.trim()
        : typeof matched?.base_url === 'string'
          ? matched.base_url.trim()
          : ''

    if (matched?.base_url && !matched?.endpoint) {
      if (ep.endpoint.trim() && ep.endpoint.trim() !== matchedDisplayEndpoint) {
        merged.base_url = ep.endpoint.trim()
      } else {
        merged.base_url = matched.base_url
      }
      delete merged.endpoint
    } else {
      merged.endpoint = ep.endpoint
    }

    if (accessKey?.trim()) {
      merged.api_key = accessKey.trim()
      delete merged.api_key_env
    } else if (!matched?.api_key && matched?.api_key_env) {
      delete merged.api_key
    }

    return merged
  })
}

export const TABLE_COLUMN_WIDTH = {
  compact: '140px',
  medium: '160px',
} as const

const SIGNAL_SECTION_KEYS = [
  'keywords',
  'embeddings',
  'domains',
  'fact_check',
  'user_feedbacks',
  'preferences',
  'language',
  'context',
  'complexity',
  'jailbreak',
  'pii',
] as const

type ConfigSignalSections = NonNullable<ConfigData['signals']>

export const collectConfiguredSignalNames = (signals?: ConfigData['signals']) => {
  if (!signals) {
    return new Set<string>()
  }

  return new Set(
    SIGNAL_SECTION_KEYS.flatMap((key) => ((signals as ConfigSignalSections)[key] || []).map((entry) => entry.name))
  )
}

export const clonePresetSignals = (signals?: Record<string, unknown>) => {
  if (!signals) {
    return undefined
  }

  return Object.fromEntries(
    Object.entries(signals).map(([key, value]) => [
      key,
      Array.isArray(value) ? value.map((item) => ({ ...item })) : value,
    ]),
  )
}

export const clonePresetDecisions = (decisions: DecisionConfig[]) =>
  decisions.map((decision) => ({
    ...decision,
    rules: {
      ...decision.rules,
      conditions: decision.rules.conditions.map((condition) => ({ ...condition })),
    },
    modelRefs: decision.modelRefs.map((modelRef) => ({ ...modelRef })),
    plugins: decision.plugins?.map((plugin) => ({
      ...plugin,
      configuration: { ...plugin.configuration },
    })),
  }))

export type ConfigDecisionConditionType = DecisionConditionType

export const getDefaultModelName = (
  config: ConfigData | null,
  isPythonCLI: boolean
): string => {
  if (isPythonCLI) {
    return config?.providers?.defaults?.default_model || ''
  }
  return config?.default_model || ''
}

export const getReasoningFamiliesMap = (
  config: ConfigData | null,
  isPythonCLI: boolean
): Record<string, ReasoningFamily> => {
  if (isPythonCLI) {
    return config?.providers?.defaults?.reasoning_families || {}
  }
  return config?.reasoning_families || {}
}

export const getNormalizedModels = (
  config: ConfigData | null,
  isPythonCLI: boolean
): NormalizedModel[] => {
  if (isPythonCLI && config?.providers?.models) {
    const cards = config?.routing?.modelCards || []
    const cardByName = new Map(cards.map((card) => [card.name, card]))
    const models = config.providers.models.map((m): NormalizedModel => ({
      name: m.name,
      reasoning_family: m.reasoning_family,
      provider_model_id: m.provider_model_id,
      api_format: m.api_format,
      external_model_ids: m.external_model_ids,
      backend_refs: m.backend_refs,
      endpoints: normalizeProviderModelEndpoints(m),
      param_size: cardByName.get(m.name)?.param_size,
      context_window_size: cardByName.get(m.name)?.context_window_size,
      description: cardByName.get(m.name)?.description,
      capabilities: cardByName.get(m.name)?.capabilities,
      loras: cardByName.get(m.name)?.loras,
      tags: cardByName.get(m.name)?.tags,
      quality_score: cardByName.get(m.name)?.quality_score,
      modality: cardByName.get(m.name)?.modality,
      pricing: m.pricing,
    }))

    for (const card of cards) {
      if (models.some((model) => model.name === card.name)) {
        continue
      }
      models.push({
        name: card.name,
        reasoning_family: undefined,
        provider_model_id: undefined,
        api_format: undefined,
        external_model_ids: undefined,
        backend_refs: undefined,
        endpoints: [],
        param_size: card.param_size,
        context_window_size: card.context_window_size,
        description: card.description,
        capabilities: card.capabilities,
        loras: card.loras,
        tags: card.tags,
        quality_score: card.quality_score,
        modality: card.modality,
        pricing: undefined,
      })
    }

    return models
  }

  if (config?.model_config) {
    return (Object.entries(config.model_config) as [string, ModelConfigEntry][]).map(([name, cfg]) => ({
      name,
      reasoning_family: cfg.reasoning_family,
      endpoints: cfg.preferred_endpoints?.map((ep: string) => {
        const endpoint = config.vllm_endpoints?.find((entry: VLLMEndpoint) => entry.name === ep)
        return endpoint ? normalizeEndpoint({
          name: ep,
          weight: endpoint.weight || 1,
          endpoint: `${endpoint.address}:${endpoint.port}`,
          protocol: 'http',
        }, 0) : null
      }).filter((entry): entry is NonNullable<typeof entry> => entry !== null) || [],
      access_key: undefined,
      pricing: cfg.pricing,
    }))
  }

  return []
}
