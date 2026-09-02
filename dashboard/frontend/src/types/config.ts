/**
 * Configuration types for vLLM Semantic Router Dashboard
 *
 * Public config uses the canonical v0.3 layout:
 *   version / listeners / providers / routing / global
 *
 * Some interfaces below are dashboard view-model adapters that flatten
 * routing-owned data for editing and legacy read-only fallback handling.
 */

// =============================================================================
// PROVIDERS - Model and endpoint configuration
// =============================================================================

export interface ProviderEndpoint {
  name: string
  weight: number
  endpoint: string // e.g., "host.docker.internal:8000" or "api.openai.com"
  protocol: 'http' | 'https'
  base_url?: string
  provider?: 'openai' | 'anthropic'
  api_key?: string
  api_key_env?: string
}

export interface ProviderModel {
  name: string // e.g., "openai/gpt-oss-120b"
  reasoning_family?: string
  provider_model_id?: string
  backend_refs?: ProviderEndpoint[]
  endpoints?: ProviderEndpoint[]
  access_key?: string
  api_format?: 'anthropic'
  external_model_ids?: Record<string, string>
  pricing?: {
    currency?: string
    prompt_per_1m?: number
    cached_input_per_1m?: number
    cache_write_per_1m?: number
    completion_per_1m?: number
  }
  reliability?: {
    lb_policy?: 'round_robin' | 'least_request'
    retry_count?: number
    retry_on?: string
    consecutive_5xx?: number
    base_ejection_time?: string
    max_ejection_percent?: number
    health_check_path?: string
    health_check_interval?: string
    health_check_timeout?: string
  }
}

export interface ProviderDefaults {
  default_model?: string
  reasoning_families?: Record<string, ReasoningFamily>
  default_reasoning_effort?: string
}

export interface ReasoningFamily {
  type: 'reasoning_effort' | 'chat_template_kwargs'
  parameter: string // e.g., "reasoning_effort", "enable_thinking"
}

export interface Providers {
  models: ProviderModel[]
  defaults?: ProviderDefaults
}

// =============================================================================
// SIGNALS - Classification signals for routing
// =============================================================================

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
  aggregation_method: 'max' | 'avg' | 'min'
}

export interface DomainSignal {
  name: string
  description: string
  mmlu_categories?: string[]
}

export interface FactCheckSignal {
  name: string
  description: string
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

export interface ConversationSignal {
  name: string
  description?: string
  feature: {
    type: 'count' | 'exists'
    source: {
      type: string
      role?: string
    }
  }
  predicate?: NumericPredicate
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

export interface InputModalitySignal {
  name: string
  description?: string
  modality: 'text' | 'image' | 'audio' | 'video'
}

export interface ComplexityCandidates {
  candidates: string[]
}

export interface RuleComposer {
  operator: 'AND' | 'OR' | 'NOT'
  conditions: Array<{
    type: string
    name: string
  }>
}

export interface ComplexitySignal {
  name: string
  threshold: number
  hard: ComplexityCandidates
  easy: ComplexityCandidates
  description?: string
  composer?: RuleComposer
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

export interface JailbreakSignal {
  name: string
  threshold: number
  method?: string // "classifier" (default) or "contrastive"
  include_history?: boolean
  direction?: 'request' | 'response' // "request" (default) scores the prompt, "response" the model's output
  jailbreak_patterns?: string[] // Known jailbreak prompts (contrastive KB)
  benign_patterns?: string[] // Known benign prompts (contrastive KB)
  description?: string
}

export interface PIISignal {
  name: string
  threshold: number
  pii_types_allowed?: string[]
  include_history?: boolean
  description?: string
}

export interface Signals {
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
  pii?: PIISignal[]
  conversation?: ConversationSignal[]
  metadata?: MetadataSignal[]
  classifiers?: ClassifierSignal[]
  input_modality?: InputModalitySignal[]
}

// =============================================================================
// DECISIONS - Routing logic
// =============================================================================

export type DecisionConditionType =
  | 'keyword'
  | 'domain'
  | 'preference'
  | 'user_feedback'
  | 'reask'
  | 'embedding'
  | 'fact_check'
  | 'language'
  | 'context'
  | 'structure'
  | 'complexity'
  | 'modality'
  | 'authz'
  | 'jailbreak'
  | 'pii'
  | 'kb'
  | 'conversation'
  | 'event'
  | 'metadata'
  | 'classifier'
  | 'input_modality'
  | 'projection'
export interface DecisionCondition {
  type: DecisionConditionType
  name: string
  label?: string
  predicate?: NumericPredicate
  on_error?: 'no_match' | 'match'
}

export interface DecisionRules {
  operator: 'AND' | 'OR' | 'NOT'
  conditions: DecisionCondition[]
  on_unknown?: 'no_match' | 'match' | 'fail_request'
}

export interface ModelRef {
  model: string
  use_reasoning: boolean
  reasoning_description?: string
  reasoning_effort?: string
  lora_name?: string
  weight?: number
}

export interface PluginConfig {
  type:
    | 'response_cache'
    | 'memory'
    | 'system_prompt'
    | 'header_mutation'
    | 'hallucination'
    | 'router_replay'
    | 'rag'
    | 'fast_response'
    | 'tools'
    | 'request_params'
    | 'response_jailbreak'
    | 'context_compression'
  configuration: Record<string, unknown>
}

export interface Decision {
  name: string
  description?: string
  priority: number
  rules: DecisionRules
  modelRefs: ModelRef[]
  plugins?: PluginConfig[]
  algorithm?: {
    type: string
    on_error?: string
    prompt?: {
      model: string
      instructions: string
      timeout_seconds?: number
    }
    [key: string]: unknown
  }
  annotations?: Record<string, unknown>
}

// =============================================================================
// LISTENERS - Network configuration
// =============================================================================

export interface Listener {
  name: string
  address: string
  port: number
  timeout?: string
}

// =============================================================================
// COMPLETE CONFIG - Python CLI format
// =============================================================================

export interface PythonCLIConfig {
  version: string
  listeners: Listener[]
  signals?: Signals
  decisions: Decision[]
  providers: Providers
}

// =============================================================================
// LEGACY CONFIG - Go router format (for backward compatibility)
// =============================================================================

export interface LegacyVLLMEndpoint {
  name: string
  address: string
  port: number
  weight: number
  health_check_path?: string
}

export interface LegacyModelConfig {
  model_id: string
  use_modernbert?: boolean
  threshold: number
  use_cpu: boolean
  category_mapping_path?: string
  pii_mapping_path?: string
  jailbreak_mapping_path?: string
}

export interface LegacyCategory {
  name: string
  description?: string
  system_prompt?: string
  mmlu_categories?: string[]
}

export interface LegacyConfig {
  vllm_endpoints?: LegacyVLLMEndpoint[]
  model_config?: Record<string, unknown>
  categories?: LegacyCategory[]
  classifier?: {
    category_model?: LegacyModelConfig
    pii_model?: LegacyModelConfig
  }
  prompt_guard?: LegacyModelConfig & { enabled: boolean }
  response_cache?: {
    enabled: boolean
    backend_type?: string
    similarity_threshold: number
    max_entries: number
    ttl_seconds: number
    eviction_policy?: string
  }
  /** @deprecated Use response_cache. */
  semantic_cache?: {
    enabled: boolean
    backend_type?: string
    similarity_threshold?: number
    max_entries?: number
    ttl_seconds?: number
  }
  tools?: {
    enabled: boolean
    top_k: number
    similarity_threshold: number
    tools_db_path: string
    fallback_to_empty: boolean
  }
  default_model?: string
  reasoning_families?: Record<string, ReasoningFamily>
  default_reasoning_effort?: string
  observability?: {
    tracing?: TracingConfig
    metrics?: { enabled: boolean }
  }
  api?: APIConfig
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
    }
  }
}

// =============================================================================
// UNIFIED CONFIG - Supports both formats
// =============================================================================

export type ConfigFormat = 'python-cli' | 'legacy'

type UnknownConfigRecord = Record<string, unknown>

export interface UnifiedConfig extends Partial<PythonCLIConfig>, Partial<LegacyConfig> {
  // Both formats can have these at root level
  version?: string
  default_model?: string
  default_reasoning_effort?: string
  reasoning_families?: Record<string, ReasoningFamily>
  observability?: {
    tracing?: TracingConfig
    metrics?: { enabled: boolean }
  }
  api?: APIConfig
  tools?: {
    enabled: boolean
    top_k: number
    similarity_threshold: number
    tools_db_path: string
    fallback_to_empty: boolean
  }
}

// =============================================================================
// FORMAT DETECTION
// =============================================================================

/**
 * Detect the config format based on key indicators
 */
const asUnknownConfigRecord = (value: unknown): UnknownConfigRecord | null =>
  value && typeof value === 'object' ? (value as UnknownConfigRecord) : null

export function detectConfigFormat(config: unknown): ConfigFormat {
  const root = asUnknownConfigRecord(config)
  // Python CLI format has providers.models
  const providers = asUnknownConfigRecord(root?.providers)
  if (providers?.models) {
    return 'python-cli'
  }
  // Legacy format has vllm_endpoints or model_config at root
  if (root?.vllm_endpoints || root?.model_config) {
    return 'legacy'
  }
  // Default to python-cli as that's the future
  return 'python-cli'
}

/**
 * Check if config has decisions, regardless of format.
 * After deploy, the Router flattens Python CLI format to legacy struct fields
 * (vllm_endpoints, model_config, keyword_rules, etc.) but preserves the `decisions` array.
 * This helper detects that hybrid state so the UI can still render decisions.
 */
export function hasDecisions(config: unknown): boolean {
  const root = asUnknownConfigRecord(config)
  return Array.isArray(root?.decisions) && root.decisions.length > 0
}

/**
 * Check if config has flat signal fields (keyword_rules, embedding_rules, etc.)
 * that result from deploy flattening. These map to the nested signals.* structure.
 */
export function hasFlatSignals(config: unknown): boolean {
  const root = asUnknownConfigRecord(config)
  return !!(
    (Array.isArray(root?.keyword_rules) && root.keyword_rules.length > 0) ||
    (Array.isArray(root?.embedding_rules) && root.embedding_rules.length > 0) ||
    (Array.isArray(root?.categories) && root.categories.length > 0) ||
    (Array.isArray(root?.fact_check_rules) && root.fact_check_rules.length > 0) ||
    (Array.isArray(root?.user_feedback_rules) && root.user_feedback_rules.length > 0) ||
    (Array.isArray(root?.reask_rules) && root.reask_rules.length > 0) ||
    (Array.isArray(root?.preference_rules) && root.preference_rules.length > 0) ||
    (Array.isArray(root?.language_rules) && root.language_rules.length > 0) ||
    (Array.isArray(root?.context_rules) && root.context_rules.length > 0) ||
    (Array.isArray(root?.structure_rules) && root.structure_rules.length > 0) ||
    (Array.isArray(root?.complexity_rules) && root.complexity_rules.length > 0) ||
    (Array.isArray(root?.jailbreak) && root.jailbreak.length > 0) ||
    (Array.isArray(root?.pii) && root.pii.length > 0)
  )
}

/**
 * Check if config is in Python CLI format
 */
export function isPythonCLIFormat(config: unknown): config is PythonCLIConfig {
  return detectConfigFormat(config) === 'python-cli'
}

/**
 * Check if config is in legacy format
 */
export function isLegacyFormat(config: unknown): config is LegacyConfig {
  return detectConfigFormat(config) === 'legacy'
}
