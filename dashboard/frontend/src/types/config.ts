/**
 * Configuration types for vLLM Semantic Router Dashboard
 *
 * Public config uses the canonical v0.5 layout:
 *   version / listeners / providers / global / recipes / entrypoints
 */

// =============================================================================
// PROVIDERS - Model and endpoint configuration
// =============================================================================

export interface ProviderEndpoint {
  name: string
  weight: number
  endpoint: string // e.g., "host.docker.internal:8000" or "https://api.openai.com/v1"
  protocol: 'http' | 'https'
  provider?: string
  api_key?: string
  auth?: { header?: string; prefix?: string; api_key_env?: string }
  headers?: Record<string, string>
}

export interface ProviderModel {
  name: string // e.g., "openai/gpt-oss-120b"
  catalog?: string
  reasoning?: ReasoningConfig
  provider_model_id?: string
  backend_refs?: ProviderEndpoint[]
  endpoints?: ProviderEndpoint[]
  access_key?: string
  api_format?: 'openai' | 'responses' | 'anthropic' | 'images' | 'systemone'
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

// =============================================================================
// SIGNALS - Classification signals for routing
// =============================================================================

export interface NumericPredicate {
  gt?: number
  gte?: number
  lt?: number
  lte?: number
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
  // false leaves the current user turn out: the rule reads only the history or tool results it names
  turn?: boolean
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
