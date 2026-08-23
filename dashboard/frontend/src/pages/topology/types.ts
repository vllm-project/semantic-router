// topology/types.ts - Topology Page Type Definitions

import { ReactNode } from 'react'

// ============== Signal Types ==============
export type SignalType =
  | 'keyword'
  | 'embedding'
  | 'domain'
  | 'fact_check'
  | 'user_feedback'
  | 'reask'
  | 'preference'
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
  | 'projection'

export interface SignalConfig {
  type: SignalType
  name: string
  description?: string
  latency: string
  config:
    | KeywordSignalConfig
    | EmbeddingSignalConfig
    | DomainSignalConfig
    | ContextSignalConfig
    | StructureSignalConfig
    | ComplexitySignalConfig
    | ModalitySignalConfig
    | AuthzSignalConfig
    | JailbreakSignalConfig
    | PIISignalConfig
    | KBSignalConfig
    | ProjectionSignalConfig
    | GenericSignalConfig
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

export interface StructureSourceConfig {
  type: string
  pattern?: string
  keywords?: string[]
  case_sensitive?: boolean
  sequences?: string[][]
}

export interface StructureFeatureConfig {
  type: string
  source?: StructureSourceConfig
}

export interface NumericPredicateConfig {
  gt?: number
  gte?: number
  lt?: number
  lte?: number
}

export interface StructureSignalConfig {
  feature?: StructureFeatureConfig
  predicate?: NumericPredicateConfig
}

export interface StructureRuleDefinition {
  name: string
  description?: string
  feature: StructureFeatureConfig
  predicate?: NumericPredicateConfig
}

export interface ComplexitySignalConfig {
  threshold?: number
  hard_candidates?: string[]
  easy_candidates?: string[]
}

export interface JailbreakSignalConfig {
  threshold?: number
  include_history?: boolean
}

// Modality is detected by the modality_detector inline model; no extra params needed.
export type ModalitySignalConfig = Record<string, never>

export interface AuthzSignalConfig {
  role?: string
}

export interface PIISignalConfig {
  threshold?: number
  pii_types_allowed?: string[]
  include_history?: boolean
}

export interface KBSignalConfig {
  kb: string
  target: {
    kind: 'label' | 'group'
    value: string
  }
  match?: 'best' | 'threshold'
}

export interface ProjectionSignalInputRef {
  type: SignalType
  name: string
}

export interface ProjectionSignalConfig {
  source: string
  method: string
  mapping: string
  upstreamSignals: ProjectionSignalInputRef[]
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
  operator: 'AND' | 'OR' | 'NOT'
  conditions: RuleNode[]
}

export interface RuleCondition {
  type: SignalType
  name: string
}

export type RuleNode = RuleCombination | RuleCondition

export interface RawRuleNode {
  type?: string
  name?: string
  operator?: string
  conditions?: RawRuleNode[]
}

export interface RawRuleCombination {
  operator?: string
  conditions?: RawRuleNode[]
}

// ============== Algorithm Types ==============
export type AlgorithmType =
  | 'confidence'
  | 'concurrent'
  | 'sequential'
  | 'ratings'
  | 'static'
  | 'router_dc'
  | 'automix'
  | 'hybrid'
  | 'remom'
  | 'fusion'
  | 'workflows'
  | 'latency_aware'
  | 'knn'
  | 'kmeans'
  | 'svm'
  | 'mlp'
  | 'multi_factor'

export interface AlgorithmConfig {
  type: AlgorithmType
  confidence?: ConfidenceAlgorithmConfig
  concurrent?: ConcurrentAlgorithmConfig
  latency_aware?: LatencyAwareAlgorithmConfig
  ratings?: GenericAlgorithmConfig
  remom?: GenericAlgorithmConfig
  fusion?: GenericAlgorithmConfig
  workflows?: GenericAlgorithmConfig
  router_dc?: GenericAlgorithmConfig
  automix?: GenericAlgorithmConfig
  autoMix?: GenericAlgorithmConfig
  hybrid?: GenericAlgorithmConfig
  knn?: GenericAlgorithmConfig
  kmeans?: GenericAlgorithmConfig
  svm?: GenericAlgorithmConfig
  mlp?: GenericAlgorithmConfig
  multi_factor?: GenericAlgorithmConfig
}

export type RawDecisionAlgorithmConfig = Omit<Partial<AlgorithmConfig>, 'type'> & {
  type: string
  autoMix?: GenericAlgorithmConfig
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

export interface LatencyAwareAlgorithmConfig {
  tpot_percentile?: number
  ttft_percentile?: number
  description?: string
}

export interface GenericAlgorithmConfig {
  [key: string]: unknown
}

// ============== Plugin Types ==============
export type PluginType =
  | 'response_cache'
  | 'memory'
  | 'system_prompt'
  | 'header_mutation'
  | 'hallucination'
  | 'router_replay'
  | 'rag'
  | 'image_gen'
  | 'fast_response'
  | 'request_params'
  | 'response_jailbreak'
  | 'tools'
  | 'tool_selection'
  | 'context_compression'

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
  cached_input_per_1m?: number
  cache_write_per_1m?: number
  completion_per_1m?: number
}

// ============== Global Plugin Types ==============
export interface GlobalPluginConfig {
  type: 'prompt_guard' | 'pii_detection' | 'response_cache'
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
  isFallbackDecision?: boolean // True if matched decision is a system fallback
  fallbackReason?: string // Reason for fallback (e.g., "low_confidence", "no_match")
}

export interface MatchedSignal {
  type: SignalType
  name: string
  matched: boolean
  value?: number
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
  defaultModel?: string // Default/fallback model when no decision matches
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

// ============== Native v0.4 Topology Projection ==============
interface NamedRecipeSignal {
  name: string
  description?: string
}

export interface TopologyRecipeSignals {
  keywords?: Array<
    NamedRecipeSignal & {
      operator: 'AND' | 'OR'
      keywords: string[]
      case_sensitive?: boolean
    }
  >
  embeddings?: Array<
    NamedRecipeSignal & {
      threshold: number
      candidates: string[]
      aggregation_method?: 'max' | 'avg' | 'min'
    }
  >
  domains?: Array<NamedRecipeSignal & { mmlu_categories?: string[] }>
  fact_check?: NamedRecipeSignal[]
  user_feedbacks?: NamedRecipeSignal[]
  reasks?: Array<NamedRecipeSignal & { threshold?: number; lookback_turns?: number }>
  preferences?: Array<NamedRecipeSignal & { examples?: string[]; threshold?: number }>
  language?: NamedRecipeSignal[]
  context?: Array<NamedRecipeSignal & { min_tokens?: string; max_tokens?: string }>
  structure?: StructureRuleDefinition[]
  complexity?: Array<
    NamedRecipeSignal & {
      threshold?: number
      hard?: { candidates?: string[] }
      easy?: { candidates?: string[] }
    }
  >
  modality?: NamedRecipeSignal[]
  role_bindings?: Array<
    NamedRecipeSignal & {
      role: string
      subjects?: Array<{ kind: 'User' | 'Group'; name: string }>
    }
  >
  jailbreak?: Array<NamedRecipeSignal & { threshold?: number; include_history?: boolean }>
  pii?: Array<
    NamedRecipeSignal & {
      threshold?: number
      pii_types_allowed?: string[]
      include_history?: boolean
    }
  >
  kb?: Array<
    NamedRecipeSignal & {
      kb: string
      target: { kind: 'label' | 'group'; value: string }
      match?: 'best' | 'threshold'
    }
  >
  conversation?: Array<
    NamedRecipeSignal & {
      feature?: Record<string, unknown>
      predicate?: Record<string, unknown>
    }
  >
  events?: Array<
    NamedRecipeSignal & {
      event_types?: string[]
      severities?: string[]
      action_codes?: string[]
      temporal?: boolean
    }
  >
}

export interface TopologyRecipeProjections {
  scores?: Array<{
    name: string
    method?: string
    inputs?: Array<{
      type: SignalType
      name: string
      weight?: number
      value_source?: string
      match?: number
      miss?: number
    }>
  }>
  mappings?: Array<{
    name: string
    source: string
    method?: string
    outputs?: Array<{
      name: string
      lt?: number
      lte?: number
      gt?: number
      gte?: number
    }>
  }>
}

export interface TopologyRecipeDecision {
  name: string
  description?: string
  priority?: number
  rules?: RawRuleCombination
  algorithm?: RawDecisionAlgorithmConfig
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
}

export interface TopologyRecipeDocument {
  strategy?: unknown
  signals?: TopologyRecipeSignals
  projections?: TopologyRecipeProjections
  decisions?: TopologyRecipeDecision[]
}

/**
 * Recipe semantics and managed Model views stay separate; Entrypoint
 * assignments are applied to the document by the caller.
 */
export interface ManagedTopologyConfig {
  models: Array<{
    name: string
    card: { reasoning?: { type?: string } }
  }>
  document: TopologyRecipeDocument
}
