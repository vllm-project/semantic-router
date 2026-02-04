// topology/types.ts - Topology Page Type Definitions

import { ReactNode } from 'react'

// ============== Signal Types ==============
export type SignalType =
  | 'keyword'
  | 'embedding'
  | 'domain'
  | 'fact_check'
  | 'user_feedback'
  | 'preference'
  | 'language'
  | 'latency'
  | 'context'
  | 'complexity'

export interface SignalConfig {
  type: SignalType
  name: string
  description?: string
  latency: string
  config: KeywordSignalConfig | EmbeddingSignalConfig | DomainSignalConfig | ContextSignalConfig | ComplexitySignalConfig | GenericSignalConfig
}

export interface KeywordSignalConfig {
  operator: 'AND' | 'OR'
  keywords: string[]
  case_sensitive: boolean
}

export interface EmbeddingSignalConfig {
  threshold: number
  candidates: string[]
  aggregation_method: 'max' | 'avg' | 'min'
}

export interface DomainSignalConfig {
  mmlu_categories?: string[]
}

export interface ContextSignalConfig {
  min_tokens?: string
  max_tokens?: string
}

export interface ComplexitySignalConfig {
  threshold?: number
  hard_candidates?: string[]
  easy_candidates?: string[]
}

export interface LatencySignalConfig {
  tpot_percentile?: number
  ttft_percentile?: number
}

export interface GenericSignalConfig {
  [key: string]: unknown
}

// ============== Decision Types ==============
export interface DecisionConfig {
  name: string
  description?: string
  priority: number
  rules: RuleCombination
  modelRefs: ModelRefConfig[]
  algorithm?: AlgorithmConfig
  plugins?: PluginConfig[]
}

export interface RuleCombination {
  operator: 'AND' | 'OR'
  conditions: RuleCondition[]
}

export interface RuleCondition {
  type: SignalType
  name: string
}

// ============== Algorithm Types ==============
export type AlgorithmType =
  | 'confidence'
  | 'concurrent'
  | 'sequential'
  | 'ratings'
  | 'static'
  | 'elo'
  | 'router_dc'
  | 'automix'
  | 'hybrid'
  | 'remom'

export interface AlgorithmConfig {
  type: AlgorithmType
  confidence?: ConfidenceAlgorithmConfig
  concurrent?: ConcurrentAlgorithmConfig
  autoMix?: AutoMixConfig
}

export interface ConfidenceAlgorithmConfig {
  threshold?: number
  avg_logprob_threshold?: number
  margin_threshold?: number
  max_escalations?: number
  on_error?: 'skip' | 'fail'
}

export interface ConcurrentAlgorithmConfig {
  timeout_seconds?: number
  on_error?: 'skip' | 'fail'
}

export interface AutoMixConfig {
  // POMDP cascade config
  [key: string]: unknown
}

// ============== Plugin Types ==============
export type PluginType =
  | 'semantic-cache'
  | 'jailbreak'
  | 'pii'
  | 'system_prompt'
  | 'header_mutation'
  | 'hallucination'
  | 'router_replay'

export interface PluginConfig {
  type: PluginType
  enabled: boolean
  configuration?: Record<string, unknown>
}

// ============== Model Types ==============
export interface ModelRefConfig {
  model: string
  use_reasoning?: boolean
  reasoning_effort?: 'low' | 'medium' | 'high'
  lora_name?: string
  reasoning_family?: string
}

export interface ModelConfig {
  name: string
  reasoning_family?: string
  endpoints?: EndpointConfig[]
  pricing?: PricingConfig
}

export interface EndpointConfig {
  name: string
  weight: number
  endpoint: string
  protocol: 'http' | 'https'
}

export interface PricingConfig {
  currency?: string
  prompt_per_1m?: number
  completion_per_1m?: number
}

// ============== Global Plugin Types ==============
export interface GlobalPluginConfig {
  type: 'prompt_guard' | 'pii_detection' | 'semantic_cache'
  enabled: boolean
  modelId?: string
  threshold?: number
  config?: Record<string, unknown>
}

// ============== Topology Node Types ==============
export type TopologyNodeType =
  | 'client'
  | 'global-plugin'
  | 'signal-group'
  | 'signal'
  | 'decision'
  | 'algorithm'
  | 'plugin-chain'
  | 'plugin'
  | 'model'

export interface TopologyNodeData {
  label: string | ReactNode
  nodeType: TopologyNodeType
  config?: unknown
  status?: 'enabled' | 'disabled' | 'active'
  metadata?: Record<string, unknown>
  // Collapse support
  collapsed?: boolean
  onToggleCollapse?: () => void
  // Highlight support
  isHighlighted?: boolean
}

// ============== Collapse State Types ==============
export interface CollapseState {
  signalGroups: Record<SignalType, boolean>
  decisions: Record<string, boolean>
  pluginChains: Record<string, boolean>
}

// ============== Test Query Types ==============
export type TestQueryMode = 'simulate' | 'dry-run'

export interface TestQueryResult {
  query: string
  mode: TestQueryMode
  matchedSignals: MatchedSignal[]
  matchedDecision: string | null
  matchedModels: string[]
  highlightedPath: string[]
  isAccurate: boolean
  evaluatedRules?: EvaluatedRule[]
  routingLatency?: number
  warning?: string
  isFallbackDecision?: boolean  // True if matched decision is a system fallback
  fallbackReason?: string       // Reason for fallback (e.g., "low_confidence", "no_match")
}

export interface MatchedSignal {
  type: SignalType
  name: string
  matched: boolean
  confidence?: number
  score?: number
  reason?: string
  needsBackend?: boolean
}

export interface EvaluatedRule {
  decisionName: string
  condition: string
  result: boolean
  priority: number
  matchedConditions?: number
  totalConditions?: number
  matchedModels?: string[]
}

// ============== Parsed Topology ==============
export interface ParsedTopology {
  globalPlugins: GlobalPluginConfig[]
  signals: SignalConfig[]
  decisions: DecisionConfig[]
  models: ModelConfig[]
  strategy: 'priority' | 'confidence'
  defaultModel?: string  // Default/fallback model when no decision matches
}

// ============== View Mode ==============
export type ViewMode = 'simple' | 'full'

// ============== Filter State ==============
export interface FilterState {
  signalTypes: SignalType[]
  pluginTypes: PluginType[]
  showDisabled: boolean
  searchQuery: string
}

// ============== Config Data (from API) ==============
export interface ConfigData {
  bert_model?: {
    model_id?: string
    threshold?: number
    use_cpu?: boolean
  }
  prompt_guard?: {
    enabled: boolean
    model_id?: string
    use_modernbert?: boolean
    threshold?: number
    use_vllm?: boolean
  }
  classifier?: {
    category_model?: {
      model_id?: string
      use_modernbert?: boolean
      threshold?: number
    }
    pii_model?: {
      enabled?: boolean
      model_id?: string
      use_modernbert?: boolean
      threshold?: number
    }
  }
  semantic_cache?: {
    enabled: boolean
    backend_type?: string
    similarity_threshold?: number
    ttl_seconds?: number
  }
  // Signal definitions
  keyword_rules?: Array<{
    name: string
    operator: 'AND' | 'OR'
    keywords: string[]
    case_sensitive?: boolean
  }>
  embedding_rules?: Array<{
    name: string
    threshold: number
    candidates: string[]
    aggregation_method?: 'max' | 'avg' | 'min'
  }>
  fact_check_rules?: Array<{
    name: string
    description?: string
  }>
  user_feedback_rules?: Array<{
    name: string
    description?: string
  }>
  preference_rules?: Array<{
    name: string
    description?: string
  }>
  language_rules?: Array<{
    name: string
    languages?: string[]
  }>
  latency_rules?: Array<{
    name: string
    description?: string
    tpot_percentile?: number
    ttft_percentile?: number
  }>
  context_rules?: Array<{
    name: string
    min_tokens?: string
    max_tokens?: string
  }>
  complexity_rules?: Array<{
    name: string
    threshold?: number
    hard?: {
      candidates?: string[]
    }
    easy?: {
      candidates?: string[]
    }
    description?: string
  }>
  // Legacy format
  categories?: Array<{
    name: string
    description?: string
    system_prompt?: string
    mmlu_categories?: string[]
    model_scores?: Array<{
      model: string
      score: number
      use_reasoning?: boolean
    }> | Record<string, number>
  }>
  model_config?: {
    [key: string]: {
      reasoning_family?: string
    }
  }
  // Python CLI format - signals wrapper
  signals?: {
    keywords?: Array<{
      name: string
      operator: 'AND' | 'OR'
      keywords: string[]
      case_sensitive?: boolean
    }>
    embeddings?: Array<{
      name: string
      threshold: number
      candidates: string[]
      aggregation_method?: 'max' | 'avg' | 'min'
    }>
    domains?: Array<{
      name: string
      description?: string
      mmlu_categories?: string[]
    }>
    fact_check?: Array<{
      name: string
      description?: string
    }>
    user_feedbacks?: Array<{
      name: string
      description?: string
    }>
    preferences?: Array<{
      name: string
      description?: string
    }>
    language?: Array<{
      name: string
      description?: string
    }>
    latency?: Array<{
      name: string
      description?: string
      tpot_percentile?: number
      ttft_percentile?: number
    }>
    context?: Array<{
      name: string
      min_tokens?: string
      max_tokens?: string
    }>
    complexity?: Array<{
      name: string
      threshold?: number
      hard?: {
        candidates?: string[]
      }
      easy?: {
        candidates?: string[]
      }
      description?: string
    }>
  }
  decisions?: Array<{
    name: string
    description?: string
    priority?: number
    rules?: {
      operator?: string
      conditions?: Array<{
        type: string
        name: string
      }>
    }
    algorithm?: {
      type: string
      confidence?: {
        threshold?: number
      }
      concurrent?: {
        timeout_seconds?: number
      }
    }
    modelRefs?: Array<{
      model: string
      use_reasoning?: boolean
      reasoning_effort?: 'low' | 'medium' | 'high'
      lora_name?: string
    }>
    plugins?: Array<{
      type: string
      enabled?: boolean
      configuration?: Record<string, unknown>
    }>
  }>
  providers?: {
    models?: Array<{
      name: string
      reasoning_family?: string
    }>
    default_model?: string
  }
}
