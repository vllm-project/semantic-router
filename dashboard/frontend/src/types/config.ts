/**
 * Configuration types for vLLM Semantic Router Dashboard
 * 
 * This file defines TypeScript interfaces for the Python CLI config format.
 * The format uses: providers, signals, decisions (not vllm_endpoints, model_config)
 */

// =============================================================================
// PROVIDERS - Model and endpoint configuration
// =============================================================================

export interface ProviderEndpoint {
  name: string
  weight: number
  endpoint: string  // e.g., "host.docker.internal:8000" or "api.openai.com"
  protocol: 'http' | 'https'
}

export interface ProviderModel {
  name: string  // e.g., "openai/gpt-oss-120b"
  reasoning_family?: string
  endpoints: ProviderEndpoint[]
  access_key?: string
  pricing?: {
    currency?: string
    prompt_per_1m?: number
    completion_per_1m?: number
  }
}

export interface ReasoningFamily {
  type: 'reasoning_effort' | 'chat_template_kwargs'
  parameter: string  // e.g., "reasoning_effort", "enable_thinking"
}

export interface Providers {
  models: ProviderModel[]
  default_model: string
  reasoning_families?: Record<string, ReasoningFamily>
  default_reasoning_effort?: string
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

export interface PreferenceSignal {
  name: string
  description: string
}

export interface LanguageSignal {
  name: string
  description?: string
}

export interface LatencySignal {
  name: string
  tpot_percentile?: number
  ttft_percentile?: number
  description?: string
}

export interface ContextSignal {
  name: string
  min_tokens: string
  max_tokens: string
  description?: string
}

export interface ComplexityCandidates {
  candidates: string[]
}

export interface RuleComposer {
  operator: 'AND' | 'OR'
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

export interface Signals {
  keywords?: KeywordSignal[]
  embeddings?: EmbeddingSignal[]
  domains?: DomainSignal[]
  fact_check?: FactCheckSignal[]
  user_feedbacks?: UserFeedbackSignal[]
  preferences?: PreferenceSignal[]
  language?: LanguageSignal[]
  latency?: LatencySignal[]
  context?: ContextSignal[]
  complexity?: ComplexitySignal[]
}

// =============================================================================
// DECISIONS - Routing logic
// =============================================================================


export type DecisionConditionType = 'keyword' | 'domain' | 'preference' | 'user_feedback' | 'embedding' | 'latency' | 'context' | 'complexity'
export interface DecisionCondition {
  type: DecisionConditionType
  name: string
}

export interface DecisionRules {
  operator: 'AND' | 'OR'
  conditions: DecisionCondition[]
}

export interface ModelRef {
  model: string
  use_reasoning: boolean
}

export interface PluginConfig {
  type: 'system_prompt' | 'semantic-cache' | 'pii' | 'hallucination' | 'jailbreak' | 'header_mutation' | 'router_replay'
  configuration: Record<string, any>
}

export interface Decision {
  name: string
  description: string
  priority: number
  rules: DecisionRules
  modelRefs: ModelRef[]
  plugins?: PluginConfig[]
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
  model_config?: Record<string, any>
  categories?: LegacyCategory[]
  classifier?: {
    category_model?: LegacyModelConfig
    pii_model?: LegacyModelConfig
  }
  prompt_guard?: LegacyModelConfig & { enabled: boolean }
  semantic_cache?: {
    enabled: boolean
    backend_type?: string
    similarity_threshold: number
    max_entries: number
    ttl_seconds: number
    eviction_policy?: string
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
export function detectConfigFormat(config: any): ConfigFormat {
  // Python CLI format has providers.models
  if (config?.providers?.models) {
    return 'python-cli'
  }
  // Legacy format has vllm_endpoints or model_config at root
  if (config?.vllm_endpoints || config?.model_config) {
    return 'legacy'
  }
  // Default to python-cli as that's the future
  return 'python-cli'
}

/**
 * Check if config is in Python CLI format
 */
export function isPythonCLIFormat(config: any): config is PythonCLIConfig {
  return detectConfigFormat(config) === 'python-cli'
}

/**
 * Check if config is in legacy format
 */
export function isLegacyFormat(config: any): config is LegacyConfig {
  return detectConfigFormat(config) === 'legacy'
}

