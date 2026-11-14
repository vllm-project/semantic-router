import type { Endpoint } from '../components/EndpointsEditor'
import bundledCatalog from '../modelCatalogMetadata'
import type { Listener, SafetySignal, SystemOneSignal } from '../types/config'
import type { BuiltInModelCatalog, CatalogBenchmark, CatalogIndex } from '../types/modelCatalog'
import type { ModelBindings } from './modelBindings'

export type ListenerConfig = Listener

export interface ModelScore {
  model: string
  score: number
  use_reasoning: boolean
  reasoning_description?: string
  reasoning_mode?: 'enabled' | 'disabled' | 'adaptive'
  reasoning_effort?: string
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
  cache_write_1h_per_1m?: number
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
  connect_timeout?: string
  total_timeout?: string
  idle_timeout?: string
  per_try_timeout?: string
  first_byte_timeout?: string
  retriable_status_codes?: number[]
  retry_back_off_base?: string
  retry_back_off_max?: string
  retry_after_max?: string
  retry_budget_percent?: number
  retry_budget_min_concurrency?: number
}

export interface LoRAAdapter {
  name: string
  description?: string
}

/** A backend's credential, the connection block a deployment uses. */
export interface BackendRefAuth {
  // delegated sends the requesting user's credential
  // (global.services.authz.delegated_credential) instead of a key.
  credential?: 'delegated'
  header?: string
  prefix?: string
  api_key_env?: string
  api_key_file?: string
}

/** How an https backend is verified, by Envoy and by the Router's own calls. */
export interface BackendRefTLS {
  // Replaces the system roots for this backend.
  ca_file?: string
  server_name?: string
  cert_file?: string
  key_file?: string
}

export interface BackendRefEntry {
  name?: string
  /** host:port with protocol, or a full URL. */
  endpoint?: string
  protocol?: 'http' | 'https'
  weight?: number
  provider?: string
  api_version?: string
  chat_path?: string
  api_key?: string
  auth?: BackendRefAuth
  headers?: Record<string, string>
  tls?: BackendRefTLS
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
  deployment?: string
  reasoning?: ModelReasoningConfig
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
  /** The CA bundle of every https backend whose own tls names none. */
  tls?: { ca_file: string }
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

// The root is a combination, a single leaf condition, or empty (unconditional).
export interface DecisionRuleSet extends DecisionCondition {
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
  reliability?: Record<string, unknown>
  fallback?: Record<string, unknown>
  output_contract_spec?: Record<string, unknown>
  candidateIterations?: Array<Record<string, unknown>>
  retention?: Record<string, unknown>
  output?: DecisionOutputConfig
  tier?: number
  annotations?: Record<string, unknown>
  output_contract?: string
}

// The media a decision's answers carry: one kind or several, generated or an
// edit of the media the request provides. Without it a decision answers text.
export interface DecisionOutputConfig {
  kind?: DecisionOutputKind | DecisionOutputKind[]
  action?: 'generate' | 'edit'
}

export type DecisionOutputKind = 'text' | 'image' | 'audio' | 'video'

// The decision editor edits kind as a list; a single kind is written as the
// scalar the configuration documents.
export function decisionOutputForEditor(output?: DecisionOutputConfig): Record<string, unknown> {
  if (!output) return {}
  const { kind, ...rest } = output
  return kind === undefined ? rest : { ...rest, kind: Array.isArray(kind) ? kind : [kind] }
}

export function decisionOutputFromEditor(value: unknown): DecisionOutputConfig | undefined {
  if (!value || typeof value !== 'object') return undefined
  const { kind, action } = value as { kind?: unknown; action?: unknown }
  const kinds = (Array.isArray(kind) ? kind : [kind])
    .filter((entry): entry is string => typeof entry === 'string' && entry.trim() !== '')
    .map((entry) => entry.trim() as DecisionOutputKind)
  const output: DecisionOutputConfig = {}
  if (kinds.length > 0) output.kind = kinds.length === 1 ? kinds[0] : kinds
  if (action === 'generate' || action === 'edit') output.action = action
  return Object.keys(output).length > 0 ? output : undefined
}

export const ROUTING_STRATEGIES = ['priority', 'confidence'] as const
export type RoutingStrategy = (typeof ROUTING_STRATEGIES)[number]
export const DEFAULT_ROUTING_STRATEGY: RoutingStrategy = 'priority'

export interface RoutingConfig {
  model_bindings?: ModelBindings
  modelCards?: RoutingModelCard[]
  signals?: ConfigSignals
  projections?: ConfigProjections
  decisions?: DecisionConfig[]
  strategy?: RoutingStrategy
}

export interface EntrypointConfig {
  model_names: string[]
  recipe: string
  api?: 'chat' | 'systemone'
  /** Marks the entrypoint of the default recipe when there are several recipes. */
  default?: boolean
}

export interface RecipeRoutingConfig {
  model_bindings?: ModelBindings
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
  api_format_override?: string
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
  modalities?: { input: string[]; output: string[] }
  pricing?: {
    currency?: string
    prompt_per_1m?: number
    cached_input_per_1m?: number
    cache_write_per_1m?: number
    cache_write_1h_per_1m?: number
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
  auto_store?: boolean
  milvus?: MemoryMilvusConfig
  embedding?: StoreEmbeddingConfig
  default_retrieval_limit?: number
  default_similarity_threshold?: number
  hybrid_search?: boolean
  hybrid_mode?: string
  adaptive_threshold?: boolean
  reflection?: MemoryReflectionConfig
}

export interface SemanticCacheConfig {
  backend_type?: string
  max_entries?: number
  ttl_seconds?: number
  eviction_policy?: string
  embedding?: StoreEmbeddingConfig
  redis?: SemanticCacheRedisConfig
  milvus?: VectorStoreMilvusConfig
}

/** The embedding of a store indexed with another model than embeddings.default. */
export interface StoreEmbeddingConfig {
  deployment?: string
  dimension?: number
}

export interface ObservabilityConfig {
  tracing?: TracingConfig
  metrics?: {
    enabled?: boolean
    windowed_metrics?: {
      enabled?: boolean
      time_windows?: string[]
      update_interval?: string
      max_models?: number
    }
  }
}

export interface LooperConfig {
  grpc_max_msg_size_mb?: number
  max_response_bytes_mb?: number
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
  embedding?: StoreEmbeddingConfig
  ingestion_workers?: number
  ingestion_drain_timeout_seconds?: number
  supported_formats?: string[]
  milvus?: VectorStoreMilvusConfig
  memory?: VectorStoreMemoryConfig
  llama_stack?: LlamaStackVectorStoreConfig
}

export interface CanonicalSystemModels {
  /** The deployment that answers question-type signals without a binding. */
  decision_model?: { deployment: string }
}

export interface ModelSelectionConfig {
  enabled?: boolean
}

export interface CanonicalEmbeddingCatalogConfig {
  /** The deployment recall embeds text with. */
  default?: { deployment: string }
}

export interface RouterCoreConfig {
  config_source?: string
  strategy?: string
  list_backend_models?: boolean
  clear_route_cache?: boolean
  streamed_body?: StreamedBodyConfig
  skip_processing?: { enabled?: boolean }
  model_selection?: ModelSelectionConfig
  learning?: RouterLearningConfig
  allow_lossy_translation?: string[]
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

export interface CanonicalIntegrationGlobalConfig {
  looper?: LooperConfig
}

/** The API the Router speaks to a deployment. */
export type ModelDeploymentProvider =
  | 'model_runtime'
  | 'systemone'
  | 'openai_embeddings'
  | 'openai_chat'

/** A credential read from an environment variable or a file, never written inline. */
export interface DeploymentAuth {
  header?: string
  /** An empty prefix sends the bare key. */
  prefix?: string
  api_key_env?: string
  api_key_file?: string
}

export interface DeploymentTLS {
  ca_file?: string
  server_name?: string
  cert_file?: string
  key_file?: string
  insecure_skip_verify?: boolean
}

export interface DeploymentLimits {
  max_request_bytes?: number
  max_response_bytes?: number
}

export interface DeploymentCapabilities {
  /** systemone only. */
  question_types?: string[]
  /** systemone only. */
  presets?: string[]
  /** openai_embeddings only. */
  dimensions?: number
}

export interface ModelReplica {
  device?: string
  endpoint?: string
  /** The served model an attached worker answers with. */
  model?: string
}

/**
 * A model the Router starts (artifact, device, engine arguments, replicas) or
 * attaches, or a remote API. An attached endpoint and every remote provider
 * take the connection block: endpoint, model, auth, headers, timeout_ms,
 * max_retries, tls, limits and capabilities.
 */
export interface CanonicalModelDeployment {
  provider?: ModelDeploymentProvider
  artifact?: string
  revision?: string
  device?: string
  input?: { max_tokens?: number; overflow?: 'reject' | 'truncate' | 'window' }
  max_num_seqs?: number
  max_num_batched_tokens?: number
  batch_window_ms?: number
  dtype?: 'auto' | 'float32' | 'bfloat16' | 'float16'
  enable_shared_context?: boolean
  max_model_len?: number
  replicas?: ModelReplica[]
  public_name?: string
  endpoint?: string
  /** The remote model ID; on an attached model_runtime endpoint, the served model. */
  model?: string
  /** openai_chat only: how the chat model generates its answer. */
  generation?: { max_tokens?: number; temperature?: number }
  auth?: DeploymentAuth
  /** Non-secret headers; a credential header belongs in auth. */
  headers?: Record<string, string>
  timeout_ms?: number
  max_retries?: number
  tls?: DeploymentTLS
  limits?: DeploymentLimits
  capabilities?: DeploymentCapabilities
}

export interface CanonicalModelCatalogConfig {
  bindings?: ModelBindings
  deployments?: Record<string, CanonicalModelDeployment>
  embeddings?: CanonicalEmbeddingCatalogConfig
  system?: CanonicalSystemModels
  kbs?: Array<Record<string, unknown>>
  admission?: Record<string, Record<string, unknown>>
  signal_timeout_ms?: number
  on_signal_overload?: {
    action?: 'refuse' | 'retry'
    wait_ms?: number
    retry_after_seconds?: number
  }
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
        agent?: string
      }
    }
    tuning?: {
      idle_timeout_seconds?: number
      min_turns_before_switch?: number
      switch_margin?: number
      stability_weight?: number
    }
    release?: {
      enabled?: boolean
      header?: string
      min_user_turns_between?: number
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
  domains?: DomainSignal[]
  fact_check?: FactCheckSignal[]
  user_feedbacks?: UserFeedbackSignal[]
  language?: LanguageSignal[]
  context?: ContextSignal[]
  complexity?: ComplexitySignal[]
  modality?: ModalitySignal[]
  role_bindings?: RoleBindingSignal[]
  jailbreak?: JailbreakSignal[]
  safety?: SafetySignal[]
  hallucination?: HallucinationSignal[]
  refusal?: RefusalSignal[]
  turn_relation?: TurnRelationSignal[]
  pii?: PIISignal[]
  metadata?: MetadataSignal[]
  conversation?: ConversationSignal[]
  input_modality?: InputModalitySignal[]
  systemone?: SystemOneSignal[]
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
  description?: string
  operator: 'AND' | 'OR' | 'all' | 'any'
  keywords: string[]
  case_sensitive?: boolean
  method?: 'regex' | 'bm25' | 'ngram'
  fuzzy_match?: boolean
  fuzzy_threshold?: number
  bm25_threshold?: number
  ngram_threshold?: number
  ngram_arity?: number
  include_history?: boolean
  count?: NumericPredicate
  density?: NumericPredicate
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

export interface DomainSignal {
  name: string
  description: string
  labels?: string[]
  model_scores?: ModelScore[]
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
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
  labels?: string[]
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
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
  labels?: string[]
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
}

export interface HallucinationSignal {
  name: string
  description?: string
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
}

export interface RefusalSignal {
  name: string
  description?: string
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
}

export interface UserFeedbackSignal {
  name: string
  description: string
  labels?: string[]
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
}

export interface TurnRelationSignal {
  name: string
  description?: string
  /** continue, reask, shift or return; a rule without labels matches nothing itself. */
  labels?: string[]
  /** How many turns before the current one the rule reads (default 4). */
  input?: { previous_turns?: number }
  deployment?: string
  timeout_ms?: number
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

export interface NumericPredicate {
  gt?: number
  gte?: number
  lt?: number
  lte?: number
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
  /** The deployment that answers this rule; without one, its binding or the decision model does. */
  deployment?: string
  /** easy, medium or hard; a rule without labels matches nothing itself. */
  labels?: string[]
  timeout_ms?: number
  description?: string
}

export interface JailbreakSignal {
  name: string
  predicate?: NumericPredicate
  stage?: 'request' | 'response'
  input?: { history?: boolean }
  deployment?: string
  timeout_ms?: number
  on_unknown?: 'match' | 'no_match'
  description?: string
}

export interface PIISignal {
  name: string
  predicate?: NumericPredicate
  ignore_labels?: string[]
  input?: { history?: boolean; tool_results?: boolean; request?: boolean }
  deployment?: string
  timeout_ms?: number
  on_unknown?: 'match' | 'no_match'
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
  /** The name and description of the recipe the manager edits as routing. */
  default_recipe?: { name: string; description?: string }
  entrypoints?: EntrypointConfig[]
  recipes?: RecipeConfig[]
  global?: CanonicalGlobalConfig
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
  reliability: Record<string, unknown>
  fallback: Record<string, unknown>
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
      const endpoint = typeof backend.endpoint === 'string' ? backend.endpoint.trim() : ''
      const protocol = backend.protocol || (endpoint.startsWith('https://') ? 'https' : 'http')
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

export const TABLE_COLUMN_WIDTH = {
  compact: '140px',
  medium: '160px',
} as const

export const getDefaultModelName = (config: ConfigData | null): string =>
  config?.providers?.defaults?.model || ''

export const getReasoningFamiliesMap = (
  catalog: BuiltInModelCatalog | null = bundledCatalog as unknown as BuiltInModelCatalog,
): Record<string, ReasoningFamily> =>
  Object.fromEntries(
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
