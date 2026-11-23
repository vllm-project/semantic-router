import type { ModelBindings } from './modelBindings'

export interface ModelPricing {
  currency?: string
  prompt_per_1m?: number
  cached_input_per_1m?: number
  cache_write_per_1m?: number
  cache_write_1h_per_1m?: number
  completion_per_1m?: number
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

export interface BudgetMatch {
  user?: string
  group?: string
}

/** A spend limit per user a rule matches, over a calendar window in UTC. */
export interface UserBudget {
  name: string
  match?: BudgetMatch
  limit: number
  window: 'hour' | 'day' | 'week' | 'month'
}

/** global.services.budgets: spend budgets per agent run and per user. */
export interface BudgetsConfig {
  currency?: string
  unpriced?: 'refuse' | 'allow'
  output_reserve_tokens?: number
  run?: { limit: number; idle_timeout_seconds?: number }
  users?: UserBudget[]
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
  entrypoints_only?: boolean
  clear_route_cache?: boolean
  streamed_body?: StreamedBodyConfig
  skip_processing?: { enabled?: boolean }
  model_selection?: ModelSelectionConfig
  learning?: RouterLearningConfig
  allow_lossy_translation?: string[]
  allow_request_extension_fields?: string[]
}

export interface CanonicalServiceGlobalConfig {
  api?: APIConfig
  response_api?: ResponseAPIConfig
  observability?: ObservabilityConfig
  authz?: AuthzConfig
  ratelimit?: RateLimitConfig
  budgets?: BudgetsConfig
  management_api?: Record<string, unknown>
  router_replay?: RouterReplayConfig
  startup_status?: Record<string, unknown>
  masking_identity?: Record<string, unknown>
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
  pricing?: ModelPricing
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
      cache_economics?: {
        enabled?: boolean
        cache_ttl_seconds?: number
        default_remaining_turns?: number
      }
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
