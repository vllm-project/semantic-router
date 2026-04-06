/**
 * Configuration types for vLLM Semantic Router Dashboard
 *
 * Public config uses the canonical v0.3 layout:
 *   version / listeners / providers / routing / global
 *
 * Some interfaces below are dashboard view-model adapters that flatten
 * routing-owned data for editing and legacy read-only fallback handling.
 */

import type { ConfigTreeObject } from './configTree'

// =============================================================================
// PROVIDERS - Model and endpoint configuration
// =============================================================================

export interface ProviderEndpoint {
  name: string
  weight: number
  endpoint: string  // e.g., "host.docker.internal:8000" or "api.openai.com"
  protocol: 'http' | 'https'
  base_url?: string
  provider?: 'openai' | 'anthropic'
  api_key?: string
  api_key_env?: string
}

export interface ProviderModel {
  name: string  // e.g., "openai/gpt-oss-120b"
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
    completion_per_1m?: number
  }
}

export interface ProviderDefaults {
  default_model?: string
  reasoning_families?: Record<string, ReasoningFamily>
  default_reasoning_effort?: string
}

export interface ReasoningFamily {
  type: 'reasoning_effort' | 'chat_template_kwargs'
  parameter: string  // e.g., "reasoning_effort", "enable_thinking"
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
  min_tokens: string
  max_tokens: string
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
}

// =============================================================================
// DECISIONS - Routing logic
// =============================================================================


export type DecisionConditionType = 'keyword' | 'domain' | 'preference' | 'user_feedback' | 'reask' | 'embedding' | 'fact_check' | 'language' | 'context' | 'structure' | 'complexity' | 'modality' | 'authz' | 'jailbreak' | 'pii' | 'projection'
export interface DecisionCondition {
  type: DecisionConditionType
  name: string
}

export interface DecisionRules {
  operator: 'AND' | 'OR' | 'NOT'
  conditions: DecisionCondition[]
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
    | 'semantic-cache'
    | 'memory'
    | 'system_prompt'
    | 'header_mutation'
    | 'hallucination'
    | 'router_replay'
    | 'rag'
    | 'image_gen'
    | 'fast_response'
    | 'tools'
    | 'request_params'
    | 'response_jailbreak'
  configuration: ConfigTreeObject
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
