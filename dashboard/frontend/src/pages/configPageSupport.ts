import type { Endpoint } from '../components/EndpointsEditor'
import bundledCatalog from '../modelCatalogDocument'
import type { DecisionConditionType } from '../types/config'
import type { BuiltInModelCatalog, CatalogBenchmark, CatalogIndex } from '../types/modelCatalog'

export interface ListenerConfig {
  name: string
  address: string
  port: number
  timeout?: string
}

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

export interface PreferenceModelConfig {
  use_contrastive?: boolean
  embedding_model?: string
  model_id?: string
  threshold?: number
  use_cpu?: boolean
}

export interface ModelScore {
  model: string
  score: number
  use_reasoning: boolean
  reasoning_description?: string
  reasoning_mode?: 'enabled' | 'disabled' | 'adaptive'
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
  activation_parameter?: string
  effort_flags?: Record<string, string>
  levels?: string[]
  default?: string
  modes?: Array<'enabled' | 'disabled' | 'adaptive'>
  default_mode?: 'enabled' | 'disabled' | 'adaptive'
  disabled?: string
}

export interface ModelPricing {
  currency?: string
  prompt_per_1m?: number
  cached_input_per_1m?: number
  cache_write_per_1m?: number
  completion_per_1m?: number
}

export interface ProviderReliability {
  lb_policy?: string
  retry_count?: number
  retry_on?: string
  consecutive_5xx?: number
  base_ejection_time?: string
  max_ejection_percent?: number
  health_check_path?: string
  health_check_interval?: string
  health_check_timeout?: string
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

export interface ModelReasoningConfig {
  family?: string
  type?: string
  parameter?: string
  activation_parameter?: string
  effort_flags?: Record<string, string>
  levels?: string[]
  default?: string
  modes?: Array<'enabled' | 'disabled' | 'adaptive'>
  default_mode?: 'enabled' | 'disabled' | 'adaptive'
  disabled?: string
}

export interface EvaluationRecordConfig {
  model: string
  benchmark: string
  benchmark_profile?: string
  reasoning_effort?: string
  metrics: Record<string, number>
  source?: string
  measured_at?: string
  metadata?: Record<string, string | number | boolean | null>
}

export interface ProviderModelConfig {
  name: string
  catalog?: string
  reasoning?: ModelReasoningConfig
  /** @deprecated Use reasoning. */
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
  reliability?: ProviderReliability
}

export interface ProviderDefaultsConfig {
  model?: string
  reasoning_effort?: string
  /** @deprecated Reasoning definitions are now catalog-backed or inline on a provider model. */
  reasoning_families?: Record<string, ReasoningFamily>
}

export interface ProvidersConfig {
  defaults?: ProviderDefaultsConfig
  models: ProviderModelConfig[]
}

export interface RoutingModelCard {
  name: string
  display_name?: string
  publisher?: string
  presentation?: {
    logo: string
    monogram: string
    monochrome: boolean
  }
  distribution?: {
    type: 'proprietary_api' | 'open_weights' | 'router_recipe'
    source: string
    license?: string
  }
  family?: string
  revision?: string
  released_at?: string
  knowledge_cutoff?: string
  lifecycle?: 'experimental' | 'active' | 'deprecated' | 'removed'
  param_size?: string
  context_window_size?: number
  max_output_tokens?: number
  description?: string
  capabilities?: string[]
  modalities?: { input: string[]; output: string[] }
  loras?: LoRAAdapter[]
  tags?: string[]
  modality?: string
}

export interface DecisionCondition {
  type?: string
  name?: string
  label?: string
  predicate?: NumericPredicate
  on_error?: 'no_match' | 'match'
  operator?: 'AND' | 'OR' | 'NOT'
  conditions?: DecisionCondition[]
}

export interface DecisionRuleSet {
  operator?: 'AND' | 'OR' | 'NOT'
  conditions?: DecisionCondition[]
  on_unknown?: 'no_match' | 'match' | 'fail_request'
}

export interface DecisionModelRef {
  model: string
  use_reasoning: boolean
  reasoning_description?: string
  reasoning_mode?: '' | 'enabled' | 'disabled' | 'adaptive'
  reasoning_effort?: string
  lora_name?: string
  weight?: number
}

export interface DecisionPluginConfig {
  type: string
  configuration: DecisionPluginConfiguration
}

export interface DecisionConfig {
  name: string
  description: string
  priority: number
  rules: DecisionRuleSet
  modelRefs: DecisionModelRef[]
  plugins?: DecisionPluginConfig[]
  algorithm?: Record<string, unknown>
  action?: { type: string; destination: string }
  adaptations?: Record<string, unknown>
  output_contract_spec?: Record<string, unknown>
  candidateIterations?: Array<Record<string, unknown>>
  emits?: Array<Record<string, unknown>>
  tier?: number
  annotations?: Record<string, unknown>
  output_contract?: string
}

export const ROUTING_STRATEGIES = ['priority', 'confidence'] as const
export type RoutingStrategy = (typeof ROUTING_STRATEGIES)[number]
export const DEFAULT_ROUTING_STRATEGY: RoutingStrategy = 'priority'

export interface RoutingConfig {
  model_bindings?: Record<string, Record<string, string>>
  modelCards?: RoutingModelCard[]
  signals?: ConfigSignals
  projections?: ConfigProjections
  decisions?: DecisionConfig[]
  strategy?: RoutingStrategy
}

export interface EntrypointConfig {
  model_names: string[]
  recipe: string
}

export interface RecipeRoutingConfig {
  model_bindings?: Record<string, Record<string, string>>
  signals?: ConfigSignals
  projections?: ConfigProjections
  decisions?: DecisionConfig[]
  strategy?: RoutingStrategy
}

export interface RecipeConfig {
  name: string
  description?: string
  routing: RecipeRoutingConfig
}

export interface NormalizedModel {
  name: string
  catalog?: string
  reasoning?: ModelReasoningConfig
  reasoning_family?: string
  reasoning_modes?: Array<'enabled' | 'disabled' | 'adaptive'>
  reasoning_efforts?: string[]
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
  card_override?: RoutingModelCard
  modality?: string
  pricing?: {
    currency?: string
    prompt_per_1m?: number
    cached_input_per_1m?: number
    cache_write_per_1m?: number
    completion_per_1m?: number
  }
  reliability?: ProviderReliability
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
    metrics?: {
      enabled?: boolean
      detailed_goroutine_tracking?: boolean
      high_resolution_timing?: boolean
      sample_rate?: number
      batch_size_ranges?: Array<{
        min: number
        max: number
        label: string
      }>
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
  enabled?: boolean
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

export interface MemoryConfig {
  enabled?: boolean
  auto_store?: boolean
  milvus?: MemoryMilvusConfig
  embedding_model?: string
  default_retrieval_limit?: number
  default_similarity_threshold?: number
  hybrid_search?: boolean
  hybrid_mode?: string
  adaptive_threshold?: boolean
  reflection?: MemoryReflectionConfig
}

export interface SemanticCacheConfig {
  enabled?: boolean
  backend_type?: string
  similarity_threshold?: number
  max_entries?: number
  ttl_seconds?: number
  eviction_policy?: string
  embedding_model?: string
  redis?: SemanticCacheRedisConfig
  milvus?: VectorStoreMilvusConfig
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
  backend?: 'candle' | 'openvino' | 'openai_compatible'
  model_type?: string
  preload_embeddings?: boolean
  target_dimension?: number
  target_layer?: number
  enable_soft_matching?: boolean
  top_k?: number
  min_score_threshold?: number
}

export interface EmbeddingEndpointConfig {
  base_url?: string
  model?: string
  api_key_env?: string
  timeout_seconds?: number
  max_retries?: number
  max_response_bytes?: number
  dimensions?: number
}

export interface EmbeddingModelsConfig {
  qwen3_model_path?: string
  gemma_model_path?: string
  mmbert_model_path?: string
  multimodal_model_path?: string
  bert_model_path?: string
  use_cpu?: boolean
  embedding_config?: EmbeddingOptimizationConfig
  endpoint?: EmbeddingEndpointConfig
}

export interface ObservabilityConfig {
  tracing?: TracingConfig
  metrics?: {
    enabled?: boolean
    windowed_metrics?: {
      enabled?: boolean
      time_windows?: string[]
      update_interval?: string
      queue_depth_estimation?: boolean
      max_models?: number
    }
  }
}

export interface LooperConfig {
  endpoint?: string
  model_endpoints?: Record<string, string>
  grpc_max_msg_size_mb?: number
  timeout_seconds?: number
  retry_count?: number
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
  ingestion_drain_timeout_seconds?: number
  supported_formats?: string[]
  milvus?: VectorStoreMilvusConfig
  memory?: VectorStoreMemoryConfig
  llama_stack?: LlamaStackVectorStoreConfig
}

export interface PromptCompressionConfig {
  enabled?: boolean
  profile?: string
  max_tokens?: number
  min_length?: number
  skip_signals?: string[]
  textrank_weight?: number
  position_weight?: number
  tfidf_weight?: number
  novelty_weight?: number
  position_depth?: number
  preserve_first_n?: number
  preserve_last_n?: number
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
    experience_weight?: number
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
  domain?: ModelConfig & { model_ref?: string; fallback_category?: string }
  mcp?: MCPCategoryModel
  pii?: ModelConfig & { model_ref?: string }
  preference?: {
    use_contrastive?: boolean
    embedding_model?: string
  }
}

export interface CanonicalHallucinationModuleConfig {
  enabled?: boolean
  fact_check?: FactCheckModelModuleConfig
  detector?: HallucinationDetectorModuleConfig
  explainer?: NLIExplainerModuleConfig
}

export interface CanonicalEmbeddingCatalogConfig {
  semantic?: EmbeddingModelsConfig
}

export interface RouterCoreConfig {
  config_source?: string
  strategy?: string
  auto_model_name?: string
  auto_model_names?: string[]
  include_config_models_in_list?: boolean
  clear_route_cache?: boolean
  streamed_body?: StreamedBodyConfig
  skip_processing?: { enabled?: boolean }
  model_selection?: ModelSelectionConfig
  learning?: RouterLearningConfig
}

export interface CanonicalServiceGlobalConfig {
  api?: APIConfig
  response_api?: ResponseAPIConfig
  observability?: ObservabilityConfig
  authz?: AuthzConfig
  ratelimit?: RateLimitConfig
  management_api?: Record<string, unknown>
  router_replay?: RouterReplayConfig
  startup_status?: Record<string, unknown>
}

export interface CanonicalStoreGlobalConfig {
  response_cache?: SemanticCacheConfig
  /** @deprecated Use response_cache. */
  semantic_cache?: SemanticCacheConfig
  memory?: MemoryConfig
  vector_store?: VectorStoreConfig
}

export interface ToolIntegrationConfig {
  enabled?: boolean
  top_k?: number
  similarity_threshold?: number
  tools_db_path?: string
  fallback_to_empty?: boolean
  advanced_filtering?: AdvancedToolFilteringConfig
}

export interface CanonicalIntegrationGlobalConfig {
  tools?: ToolIntegrationConfig
  looper?: LooperConfig
}

export interface CanonicalModelModulesConfig {
  prompt_compression?: PromptCompressionConfig
  prompt_guard?: ModelConfig & { enabled?: boolean; model_ref?: string; use_vllm?: boolean }
  classifier?: CanonicalClassifierConfig
  complexity?: Record<string, unknown>
  hallucination_mitigation?: CanonicalHallucinationModuleConfig
  feedback_detector?: FeedbackDetectorConfig & { model_ref?: string }
  modality_detector?: ModalityDetectorConfig
}

export interface CanonicalModelCatalogConfig {
  embeddings?: CanonicalEmbeddingCatalogConfig
  system?: CanonicalSystemModels
  external?: ExternalModelConfig[]
  kbs?: Array<Record<string, unknown>>
  modules?: CanonicalModelModulesConfig
  admission?: Record<string, Record<string, unknown>>
}

export interface RouterLearningConfig {
  enabled?: boolean
  adaptation?: {
    enabled?: boolean
    candidate_set?: 'decision' | 'tier' | 'global'
    strategy?: string
  }
  protection?: {
    enabled?: boolean
    scope?: 'conversation' | 'session'
    identity?: {
      headers?: {
        session?: string
        conversation?: string
      }
    }
    tuning?: {
      idle_timeout_seconds?: number
      min_turns_before_switch?: number
      switch_margin?: number
      stability_weight?: number
    }
  }
  state_store?: {
    backend?: string
    ttl_seconds?: number
    timeout_ms?: number
    redis?: {
      address?: string
      password?: string
      database?: number
      key_prefix?: string
    }
  }
}

export interface CanonicalGlobalConfig {
  router?: RouterCoreConfig
  services?: CanonicalServiceGlobalConfig
  stores?: CanonicalStoreGlobalConfig
  integrations?: CanonicalIntegrationGlobalConfig
  model_catalog?: CanonicalModelCatalogConfig
}

export interface ConfigSignals {
  keywords?: KeywordSignal[]
  embeddings?: EmbeddingSignal[]
  domains?: DomainSignal[]
  fact_check?: FactCheckSignal[]
  user_feedbacks?: UserFeedbackSignal[]
  reasks?: ReaskSignal[]
  preferences?: PreferenceSignal[]
  language?: LanguageSignal[]
  context?: ContextSignal[]
  structure?: StructureSignal[]
  complexity?: ComplexitySignal[]
  modality?: ModalitySignal[]
  role_bindings?: RoleBindingSignal[]
  jailbreak?: JailbreakSignal[]
  hallucination?: HallucinationSignal[]
  pii?: PIISignal[]
  kb?: KBSignal[]
  metadata?: MetadataSignal[]
  classifiers?: ClassifierSignal[]
  conversation?: ConversationSignal[]
  events?: EventSignal[]
  input_modality?: InputModalitySignal[]
}

export interface ConfigProjections {
  partitions?: ProjectionPartition[]
  scores?: ProjectionScore[]
  mappings?: ProjectionMapping[]
}

export interface DecisionPluginConfiguration {
  [key: string]: unknown
}

export interface MemoryReflectionConfig {
  enabled?: boolean
  algorithm?: string
  max_inject_tokens?: number
  recency_decay_days?: number
  dedup_threshold?: number
  block_patterns?: string[]
}

export interface VectorStoreMilvusConnectionAuthConfig {
  enabled?: boolean
  username?: string
  password?: string
}

export interface VectorStoreMilvusConnectionTLSConfig {
  enabled?: boolean
  cert_file?: string
  key_file?: string
  ca_file?: string
}

export interface VectorStoreMilvusConnectionConfig {
  host?: string
  port?: number
  database?: string
  timeout?: number
  auth?: VectorStoreMilvusConnectionAuthConfig
  tls?: VectorStoreMilvusConnectionTLSConfig
}

export interface VectorStoreMilvusVectorFieldConfig {
  name?: string
  dimension?: number
  metric_type?: string
}

export interface VectorStoreMilvusIndexParamsConfig {
  M?: number
  efConstruction?: number
}

export interface VectorStoreMilvusIndexConfig {
  type?: string
  params?: VectorStoreMilvusIndexParamsConfig
}

export interface VectorStoreMilvusCollectionConfig {
  name?: string
  description?: string
  vector_field?: VectorStoreMilvusVectorFieldConfig
  index?: VectorStoreMilvusIndexConfig
}

export interface VectorStoreMilvusSearchParamsConfig {
  ef?: number
}

export interface VectorStoreMilvusSearchConfig {
  params?: VectorStoreMilvusSearchParamsConfig
  topk?: number
  consistency_level?: string
}

export interface VectorStoreMilvusLoggingConfig {
  level?: string
}

export interface VectorStoreMilvusDevelopmentConfig {
  drop_collection_on_startup?: boolean
  auto_create_collection?: boolean
}

export interface VectorStoreMilvusConfig {
  connection?: VectorStoreMilvusConnectionConfig
  collection?: VectorStoreMilvusCollectionConfig
  search?: VectorStoreMilvusSearchConfig
  logging?: VectorStoreMilvusLoggingConfig
  development?: VectorStoreMilvusDevelopmentConfig
}

export interface SemanticCacheRedisTLSConfig {
  enabled?: boolean
  cert_file?: string
  key_file?: string
  ca_file?: string
}

export interface SemanticCacheRedisConnectionConfig {
  host?: string
  port?: number
  database?: number
  password?: string
  timeout?: number
  tls?: SemanticCacheRedisTLSConfig
}

export interface SemanticCacheRedisVectorFieldConfig {
  name?: string
  dimension?: number
  metric_type?: string
}

export interface SemanticCacheRedisIndexParamsConfig {
  M?: number
  efConstruction?: number
}

export interface SemanticCacheRedisIndexConfig {
  name?: string
  prefix?: string
  vector_field?: SemanticCacheRedisVectorFieldConfig
  index_type?: string
  params?: SemanticCacheRedisIndexParamsConfig
}

export interface SemanticCacheRedisSearchConfig {
  topk?: number
}

export interface SemanticCacheRedisDevelopmentConfig {
  drop_index_on_startup?: boolean
  auto_create_index?: boolean
}

export interface SemanticCacheRedisLoggingConfig {
  level?: string
}

export interface SemanticCacheRedisConfig {
  connection?: SemanticCacheRedisConnectionConfig
  index?: SemanticCacheRedisIndexConfig
  search?: SemanticCacheRedisSearchConfig
  development?: SemanticCacheRedisDevelopmentConfig
  logging?: SemanticCacheRedisLoggingConfig
}

export interface KeywordSignal {
  name: string
  operator: 'AND' | 'OR' | 'all' | 'any'
  keywords: string[]
  case_sensitive?: boolean
  method?: 'regex' | 'bm25' | 'ngram'
  fuzzy_match?: boolean
  fuzzy_threshold?: number
  bm25_threshold?: number
  ngram_threshold?: number
  ngram_arity?: number
}

export interface EmbeddingSignal {
  name: string
  threshold: number
  candidates: string[]
  aggregation_method?: string
  query_modality?: 'text' | 'image' | 'audio'
}

export interface MetadataSignal {
  name: string
  description?: string
  key: string
  predicate: {
    equals?: string
    in?: string[]
    exists?: boolean
  }
}

export interface ClassifierSignal {
  name: string
  description?: string
  type: 'local' | 'llm' | 'sequence_classifier'
  model?: string
  model_path?: string
  labels: string[]
  instructions?: string
  use_cpu?: boolean
}

export interface DomainSignal {
  name: string
  description: string
  mmlu_categories?: string[]
  model_scores?: ModelScore[]
}

export interface EventSignal {
  name: string
  description?: string
  event_types?: string[]
  severities?: string[]
  action_codes?: string[]
  temporal?: boolean
}

export interface InputModalitySignal {
  name: string
  description?: string
  modality: 'text' | 'image' | 'audio' | 'video'
}

export interface ProjectionPartition {
  name: string
  semantics: string
  members: string[]
  temperature?: number
  default: string
}

export interface ProjectionScoreInput {
  type: string
  name?: string
  kb?: string
  metric?: string
  weight: number
  value_source?: string
  match?: number
  miss?: number
}

export interface ProjectionScore {
  name: string
  method: string
  inputs: ProjectionScoreInput[]
}

export interface ProjectionMappingCalibration {
  method: string
  slope?: number
}

export interface ProjectionMappingOutput {
  name: string
  lt?: number
  lte?: number
  gt?: number
  gte?: number
}

export interface ProjectionMapping {
  name: string
  source: string
  method: string
  calibration?: ProjectionMappingCalibration
  outputs: ProjectionMappingOutput[]
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

export interface KBSignal {
  name: string
  kb: string
  target: {
    kind: 'label' | 'group'
    value: string
  }
  match?: 'best' | 'threshold'
}

export interface FactCheckSignal {
  name: string
  description: string
}

export interface HallucinationSignal {
  name: string
  use_nli?: boolean
  description?: string
}

export interface UserFeedbackSignal {
  name: string
  description: string
}

export interface ReaskSignal {
  name: string
  description?: string
  threshold?: number
  lookback_turns?: number
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
  threshold?: number
}

export interface ContextSignal {
  name: string
  /** Inclusive lower bound. Defaults to 0 when omitted. */
  min_tokens?: string
  /** Inclusive upper bound. Omit for an open-ended band (no upper limit). */
  max_tokens?: string
  description?: string
}

export interface StructureSource {
  type: string
  pattern?: string
  keywords?: string[]
  case_sensitive?: boolean
  sequences?: string[][]
}

export interface StructureFeature {
  type: string
  source: StructureSource
}

export interface NumericPredicate {
  gt?: number
  gte?: number
  lt?: number
  lte?: number
}

export interface StructureSignal {
  name: string
  description?: string
  feature: StructureFeature
  predicate?: NumericPredicate
}

export interface ConversationSource {
  type: string
  role?: string
}

export interface ConversationFeature {
  type: string
  source: ConversationSource
}

export interface ConversationSignal {
  name: string
  description?: string
  feature: ConversationFeature
  predicate?: NumericPredicate
}

export interface ComplexitySignal {
  name: string
  threshold?: number
  hard_above?: number
  easy_below?: number
  hard_below?: number
  easy_above?: number
  hard?: { candidates?: string[]; image_candidates?: string[] }
  easy?: { candidates?: string[]; image_candidates?: string[] }
  description?: string
  composer?: {
    operator: 'AND' | 'OR' | 'NOT'
    conditions: DecisionCondition[]
  }
}

export interface JailbreakSignal {
  name: string
  threshold?: number
  method?: string
  include_history?: boolean
  direction?: 'request' | 'response'
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
  listeners?: ListenerConfig[]
  signals?: ConfigSignals
  projections?: ConfigProjections
  decisions?: DecisionConfig[]
  providers?: ProvidersConfig
  evaluation?: {
    benchmarks?: CatalogBenchmark[]
    indices?: CatalogIndex[]
    records?: EvaluationRecordConfig[]
  }
  routing?: RoutingConfig
  entrypoints?: EntrypointConfig[]
  recipes?: RecipeConfig[]
  global?: CanonicalGlobalConfig
  response_cache?: SemanticCacheConfig
  /** @deprecated Use response_cache. */
  semantic_cache?: SemanticCacheConfig
  tools?: ToolIntegrationConfig
  prompt_guard?: ModelConfig & { enabled: boolean }
  vllm_endpoints?: VLLMEndpoint[]
  classifier?: {
    category_model?: ModelConfig
    mcp_category_model?: MCPCategoryModel
    pii_model?: ModelConfig
    preference_model?: PreferenceModelConfig
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
  external_models?: ExternalModelConfig[]
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
  reask_rules?: ReaskSignal[]
  preference_rules?: PreferenceSignal[]
  language_rules?: LanguageSignal[]
  context_rules?: ContextSignal[]
  structure_rules?: StructureSignal[]
  complexity_rules?: ComplexitySignal[]
  jailbreak?: JailbreakSignal[]
  hallucination?: HallucinationSignal[]
  pii?: PIISignal[]
}

export type SignalType = string

export interface DecisionFormState {
  name: string
  description: string
  priority: number
  rules: DecisionRuleSet
  modelRefs: DecisionModelRef[]
  plugins: { type: string; configuration: string | DecisionPluginConfiguration }[]
  tier?: number
  output_contract?: string
  output_contract_spec: Record<string, unknown>
  action: Record<string, unknown>
  algorithm?: Record<string, unknown>
  adaptations: Record<string, unknown>
  declarative: Record<string, unknown>
}

export function mergeDecisionForSave(
  existing: DecisionConfig | undefined,
  update: DecisionConfig,
): DecisionConfig {
  return {
    ...(existing || {}),
    ...update,
  }
}

export const formatThreshold = (value: number): string => {
  return `${Math.round(value * 100)}%`
}

export const normalizeModelScores = (
  modelScores: ModelScore[] | Record<string, number> | undefined,
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
  index: number,
): Endpoint => ({
  name: endpoint?.name?.trim() || `endpoint-${index + 1}`,
  endpoint: endpoint?.endpoint?.trim() || '',
  protocol: normalizeEndpointProtocol(endpoint?.protocol),
  weight:
    typeof endpoint?.weight === 'number' && Number.isFinite(endpoint.weight) ? endpoint.weight : 1,
})

export const normalizeEndpoints = (endpoints: Partial<Endpoint>[] | undefined): Endpoint[] =>
  Array.isArray(endpoints)
    ? endpoints.map((endpoint, index) => normalizeEndpoint(endpoint, index))
    : []

export const normalizeProviderModelEndpoints = (model: {
  endpoints?: Partial<Endpoint>[]
  backend_refs?: BackendRefEntry[]
}): Endpoint[] => {
  if (Array.isArray(model.backend_refs) && model.backend_refs.length > 0) {
    return model.backend_refs.map((backend, index) => {
      const baseURL = typeof backend.base_url === 'string' ? backend.base_url.trim() : ''
      const endpoint =
        typeof backend.endpoint === 'string' && backend.endpoint.trim()
          ? backend.endpoint.trim()
          : baseURL
      const protocol = backend.protocol || (baseURL.startsWith('https://') ? 'https' : 'http')
      return normalizeEndpoint(
        {
          name: backend.name,
          endpoint,
          protocol,
          weight: backend.weight,
        },
        index,
      )
    })
  }
  return normalizeEndpoints(model.endpoints)
}

export const mergeProviderBackendRefs = (
  existingRefs: BackendRefEntry[] | undefined,
  endpoints: Endpoint[],
  accessKey?: string,
): BackendRefEntry[] => {
  const existing = Array.isArray(existingRefs) ? existingRefs : []

  return endpoints.map((ep, index) => {
    const matched = existing.find((ref) => ref.name === ep.name) || existing[index]

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

export type ConfigDecisionConditionType = DecisionConditionType

export const getDefaultModelName = (config: ConfigData | null, isPythonCLI: boolean): string => {
  if (isPythonCLI) {
    return config?.providers?.defaults?.model || ''
  }
  return config?.default_model || ''
}

export const getReasoningFamiliesMap = (
  config: ConfigData | null,
  isPythonCLI: boolean,
  catalog: BuiltInModelCatalog | null = bundledCatalog as unknown as BuiltInModelCatalog,
): Record<string, ReasoningFamily> => {
  if (isPythonCLI) {
    return Object.fromEntries(
      (catalog?.reasoning_families ?? []).map((family) => [
        family.id,
        {
          type: family.type,
          parameter: family.parameter,
          activation_parameter: family.activation_parameter,
          effort_flags: family.effort_flags ? { ...family.effort_flags } : undefined,
          levels: [...(family.levels ?? [])],
          default: family.default,
          modes: family.modes ? [...family.modes] : undefined,
          default_mode: family.default_mode,
          disabled: family.disabled,
        },
      ]),
    )
  }
  return config?.reasoning_families || {}
}
