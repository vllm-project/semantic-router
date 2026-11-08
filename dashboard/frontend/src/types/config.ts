/**
 * Configuration types for vLLM Semantic Router Dashboard
 *
 * Public config uses the canonical v0.5 layout:
 *   version / listeners / providers / global / recipes / entrypoints
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
  provider?: string
  api_key?: string
  api_key_env?: string
}

export interface ProviderModel {
  name: string // e.g., "openai/gpt-oss-120b"
  catalog?: string
  reasoning?: ReasoningConfig
  provider_model_id?: string
  backend_refs?: ProviderEndpoint[]
  endpoints?: ProviderEndpoint[]
  access_key?: string
  api_format?: 'openai' | 'responses' | 'anthropic' | 'systemone'
  external_model_ids?: Record<string, string>
  pricing?: {
    currency?: string
    prompt_per_1m?: number
    cached_input_per_1m?: number
    cache_write_per_1m?: number
    cache_write_1h_per_1m?: number
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
}

export interface ProviderDefaults {
  model?: string
  reasoning_effort?: string
}

export interface ReasoningFamily {
  type:
    | 'reasoning_effort'
    | 'reasoning_mode'
    | 'chat_template_kwargs'
    | 'top_level_reasoning_effort'
  parameter: string // e.g., "reasoning_effort", "enable_thinking"
  activation_parameter?: string
  effort_flags?: Record<string, string>
  levels?: string[]
  default?: string
  modes?: Array<'enabled' | 'disabled' | 'adaptive'>
  default_mode?: 'enabled' | 'disabled' | 'adaptive'
  disabled?: string
}

export interface ReasoningConfig extends Partial<ReasoningFamily> {
  family?: string
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
  description?: string
  operator: 'AND' | 'OR'
  keywords: string[]
  case_sensitive: boolean
  /** An OR rule matches on how many times its keywords occur. */
  count?: NumericPredicate
  /** An OR rule matches on its keyword occurrences per 1,000 characters. */
  density?: NumericPredicate
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

export interface HallucinationSignal {
  name: string
  description?: string
}

export interface UserFeedbackSignal {
  name: string
  description: string
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

export interface NumericPredicate {
  gt?: number
  gte?: number
  lt?: number
  lte?: number
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

export interface InputModalitySignal {
  name: string
  description?: string
  modality: 'text' | 'image' | 'audio' | 'video'
}

// TopicContinuitySignal is context-policy evidence, not a decision condition.
// Every field except name is optional; the Router applies documented defaults.
export interface TopicContinuitySignal {
  name: string
  description?: string
  include_assistant?: boolean
  thresholds?: {
    continuation?: number
    change?: number
  }
  limits?: {
    max_prior_turns?: number
    max_turn_bytes?: number
    max_input_bytes?: number
  }
}

export interface SystemOneOption {
  key: string
  description?: string
}

export interface SystemOneInput {
  history?: boolean
  previous_turns?: number
  answer?: boolean
  context?: boolean
  tool_results?: boolean
  image?: boolean
  audio?: boolean
  video?: boolean
}

export interface SystemOneSignal {
  name: string
  description?: string
  deployment: string
  question: {
    type: 'choice' | 'noul' | 'score' | 'set' | 'span'
    instructions: string
    choices?: SystemOneOption[]
    levels?: string[]
    labels?: SystemOneOption[]
    threshold?: number
    head?: 'router' | 'broad'
  }
  labels?: string[]
  ignore_labels?: string[]
  predicate?: NumericPredicate
  stage?: 'request' | 'response'
  input?: SystemOneInput
  coverage?: 'truncate' | 'full'
  on_unscanned?: 'match' | 'no_match' | 'unknown'
  timeout_ms?: number
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
  /** The deployment that answers this rule; without one, its binding or the decision model does. */
  deployment?: string
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
  description?: string
  /** The deployment that answers this rule; without one, its binding or the decision model does. */
  deployment?: string
  predicate?: NumericPredicate
  stage?: 'request' | 'response'
  timeout_ms?: number
}

export interface SafetySignal {
  name: string
  description?: string
  /** The deployment that answers this rule; without one, its binding or the decision model does. */
  deployment?: string
  /** Harm categories the rule matches; without labels it asks whether the content is harmful. */
  labels?: string[]
  predicate?: NumericPredicate
  stage?: 'request' | 'response'
  timeout_ms?: number
}

export interface PIISignal {
  name: string
  description?: string
  /** The deployment that answers this rule; without one, its binding or the decision model does. */
  deployment?: string
  predicate?: NumericPredicate
  /** PII types the rule does not match. */
  ignore_labels?: string[]
  timeout_ms?: number
}

export interface Signals {
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
  pii?: PIISignal[]
  conversation?: ConversationSignal[]
  metadata?: MetadataSignal[]
  input_modality?: InputModalitySignal[]
  systemone?: SystemOneSignal[]
  topic_continuity?: TopicContinuitySignal[]
}

// =============================================================================
// DECISIONS - Routing logic
// =============================================================================

export type DecisionConditionType =
  | 'keyword'
  | 'domain'
  | 'user_feedback'
  | 'turn_relation'
  | 'fact_check'
  | 'language'
  | 'context'
  | 'complexity'
  | 'modality'
  | 'authz'
  | 'jailbreak'
  | 'safety'
  | 'hallucination'
  | 'refusal'
  | 'pii'
  | 'conversation'
  | 'metadata'
  | 'input_modality'
  | 'systemone'
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
  reasoning_mode?: 'enabled' | 'disabled' | 'adaptive'
  reasoning_effort?: string
  lora_name?: string
  weight?: number
}

export interface PluginConfig {
  type:
    | 'response_cache'
    | 'context'
    | 'system_prompt'
    | 'header_mutation'
    | 'router_replay'
    | 'fast_response'
    | 'tools'
    | 'request_params'
    | 'response_guard'
    | 'prompt_cache'
    | 'masking'
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

export interface ListenerTLS {
  cert_file: string
  key_file: string
}

export interface ListenerIdentity {
  trust_headers?: boolean
  trusted_peers?: string[]
}

export interface Listener {
  name: string
  address: string
  port: number
  timeout?: string
  api_keys?: string[]
  models?: string[]
  tls?: ListenerTLS
  identity?: ListenerIdentity
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
    (Array.isArray(root?.complexity_rules) && root.complexity_rules.length > 0) ||
    (Array.isArray(root?.jailbreak) && root.jailbreak.length > 0) ||
    (Array.isArray(root?.hallucination) && root.hallucination.length > 0) ||
    (Array.isArray(root?.pii) && root.pii.length > 0)
  )
}

/**
 * Check if config is in Python CLI format
 */
export function isPythonCLIFormat(config: unknown): config is PythonCLIConfig {
  return detectConfigFormat(config) === 'python-cli'
}
