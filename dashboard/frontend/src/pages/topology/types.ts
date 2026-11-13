// topology/types.ts - Topology Page Type Definitions

import { ReactNode } from 'react'
import type { NumericPredicate, SafetySignal } from '../../types/config'
import type {
  AlgorithmType as CanonicalAlgorithmType,
  PluginType as CanonicalPluginType,
  SignalType as CanonicalSignalType,
} from '../../generated/routerConfigContract'
import type { TopologyOptionalCacheConfig } from './cacheTypes'

// ============== Signal Types ==============
export type SignalType = CanonicalSignalType | 'projection'

export interface SignalConfig {
  type: SignalType
  name: string
  description?: string
  latency: string
  config:
    | KeywordSignalConfig
    | DomainSignalConfig
    | ContextSignalConfig
    | ComplexitySignalConfig
    | ModalitySignalConfig
    | AuthzSignalConfig
    | JailbreakSignalConfig
    | PIISignalConfig
    | ProjectionSignalConfig
    | GenericSignalConfig
}

export interface KeywordSignalConfig {
  operator: 'AND' | 'OR'
  keywords: string[]
  case_sensitive: boolean
  input?: { tool_results?: boolean; request?: boolean }
  on_unscanned?: 'best_effort' | 'match' | 'no_match'
  count?: NumericPredicateConfig
  density?: NumericPredicateConfig
}

export interface DomainSignalConfig {
  labels?: string[]
}

export interface ContextSignalConfig {
  min_tokens?: string
  max_tokens?: string
}

export interface NumericPredicateConfig {
  gt?: number
  gte?: number
  lt?: number
  lte?: number
}

export interface ComplexitySignalConfig {
  threshold?: number
  hard_candidates?: string[]
  easy_candidates?: string[]
}

export interface JailbreakSignalConfig {
  threshold?: number
  include_history?: boolean
  direction?: 'request' | 'response'
}

// Modality is detected by the modality_detector inline model; no extra params needed.
export type ModalitySignalConfig = Record<string, never>

export interface AuthzSignalConfig {
  role?: string
}

export interface PIISignalConfig {
  threshold?: number
  source?: 'tool_result'
  pii_types_allowed?: string[]
  include_history?: boolean
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
  type?: string
  name?: string
  operator?: string
  conditions?: RawRuleNode[]
}

// ============== Algorithm Types ==============
export type AlgorithmType = CanonicalAlgorithmType

export interface AlgorithmConfig {
  type: AlgorithmType
  minimum_candidates?: number
  budget?: AlgorithmBudget
  on_error?: string
  multi_factor?: GenericAlgorithmConfig
  systemone?: GenericAlgorithmConfig
  escalate?: EscalateAlgorithmConfig
  fusion?: GenericAlgorithmConfig
  remom?: GenericAlgorithmConfig
  ratings?: GenericAlgorithmConfig
  workflows?: GenericAlgorithmConfig
}

export type RawDecisionAlgorithmConfig = Omit<Partial<AlgorithmConfig>, 'type'> & {
  type: string
}

export interface AlgorithmBudget {
  deadline?: string
  max_calls?: number
}

export interface EscalateAlgorithmConfig {
  order?: 'size' | 'cost'
  accept?: {
    confidence?: { method?: string; gte?: number }
    systemone?: { gte?: number }
    refusal?: boolean
  }
  quality?: GenericAlgorithmConfig
  stages?: GenericAlgorithmConfig[]
}

export interface GenericAlgorithmConfig {
  [key: string]: unknown
}

// ============== Plugin Types ==============
export type PluginType = CanonicalPluginType

export interface PluginConfig {
  type: PluginType
  enabled: boolean
  configuration?: Record<string, unknown>
}

// ============== Model Types ==============
export interface ModelRefConfig {
  model: string
  use_reasoning?: boolean
  reasoning_mode?: 'enabled' | 'disabled' | 'adaptive'
  reasoning_effort?: string
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
  cache_write_1h_per_1m?: number
  completion_per_1m?: number
}

// ============== Global Plugin Types ==============
export interface GlobalPluginConfig {
  type: 'response_cache'
  enabled: boolean
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
  evalTrace?: Array<Record<string, unknown>>
  signalErrors?: Record<string, string>
  appliedUnknownPolicies?: Record<string, string>
  decisionError?: string
  selectedModel?: string
  recommendedModels?: string[]
  selectionStatus?: string
  selectionMethod?: string
  selectionReason?: string
  routingLatency?: number
  warning?: string
  decisionConfidence?: number | null
  decisionConfidenceAvailable?: boolean
  signalErrorMatches?: Record<string, boolean>
  isFallbackDecision?: boolean // True if matched decision is a system fallback
  fallbackReason?: string // Reason for fallback (e.g., "low_confidence", "no_match")
}

export interface MatchedSignal {
  type: SignalType
  name: string
  matched: boolean
  value?: number
  confidence?: number | null
  confidenceAvailable?: boolean
  score?: number | null
  reason?: string
  needsBackend?: boolean
}

export interface EvaluatedRule {
  decisionName: string
  condition: string
  state?: string
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

// ============== Config Data (from API) ==============
export interface ConfigData {
  projections?: {
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
  // The selected routing profile's signals
  signals?: {
    [collection: string]: unknown
    keywords?: Array<{
      name: string
      operator: 'AND' | 'OR'
      keywords: string[]
      case_sensitive?: boolean
    }>
    domains?: Array<{
      name: string
      description?: string
      labels?: string[]
    }>
    fact_check?: Array<{
      name: string
      description?: string
    }>
    user_feedbacks?: Array<{
      name: string
      description?: string
    }>
    language?: Array<{
      name: string
      description?: string
    }>
    context?: Array<{
      name: string
      min_tokens?: string
      max_tokens?: string
    }>
    complexity?: Array<{
      name: string
      labels?: string[]
      deployment?: string
      timeout_ms?: number
      description?: string
    }>
    modality?: Array<{
      name: string
      description?: string
    }>
    role_bindings?: Array<{
      name: string
      role: string
      subjects: Array<{
        kind: 'User' | 'Group'
        name: string
      }>
      description?: string
    }>
    safety?: SafetySignal[]
    jailbreak?: Array<{
      name: string
      predicate?: NumericPredicate
      stage?: 'request' | 'response'
      input?: { history?: boolean }
      deployment?: string
      timeout_ms?: number
      on_unknown?: 'match' | 'no_match'
      description?: string
    }>
    hallucination?: Array<{
      name: string
      description?: string
    }>
    pii?: Array<{
      name: string
      predicate?: NumericPredicate
      ignore_labels?: string[]
      input?: { history?: boolean; tool_results?: boolean; request?: boolean }
      deployment?: string
      timeout_ms?: number
      on_unknown?: 'match' | 'no_match'
      description?: string
    }>
    conversation?: Array<{
      name: string
      description?: string
      feature?: Record<string, unknown>
      predicate?: Record<string, unknown>
    }>
  }
  decisions?: Array<{
    name: string
    description?: string
    priority?: number
    rules?: RawRuleCombination
    algorithm?: RawDecisionAlgorithmConfig
    modelRefs?: Array<{
      model: string
      use_reasoning?: boolean
      reasoning_mode?: 'enabled' | 'disabled' | 'adaptive'
      reasoning_effort?: string
      lora_name?: string
    }>
    plugins?: Array<{
      type: string
      enabled?: boolean
      configuration?: Record<string, unknown>
    }>
  }>
  providers?: {
    defaults?: {
      model?: string
    }
    models?: Array<{
      name: string
      catalog?: string
      reasoning?: {
        family?: string
        type?: string
        parameter?: string
        levels?: string[]
        default?: string
      }
    }>
  }
  routing?: {
    modelCards?: Array<{
      name: string
    }>
    signals?: ConfigData['signals']
    projections?: ConfigData['projections']
    decisions?: ConfigData['decisions']
    strategy?: 'priority' | 'confidence'
  }
  entrypoints?: Array<{
    model_names: string[]
    recipe: string
  }>
  recipes?: Array<{
    name: string
    description?: string
    routing: {
      signals?: ConfigData['signals']
      projections?: ConfigData['projections']
      decisions?: ConfigData['decisions']
      strategy?: 'priority' | 'confidence'
    }
  }>
  global?: {
    router?: {
      strategy?: 'priority' | 'confidence'
    }
    stores?: {
      response_cache?: TopologyOptionalCacheConfig
    }
  }
}
