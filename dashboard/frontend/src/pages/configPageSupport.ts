import type { DecisionConditionType } from '../types/config'

export interface NumericPredicate {
  gt?: number
  gte?: number
  lt?: number
  lte?: number
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

export interface DecisionRuleSet {
  operator: 'AND' | 'OR' | 'NOT'
  conditions: DecisionCondition[]
}

export interface DecisionPluginConfiguration {
  [key: string]: unknown
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
  plugins?: DecisionPluginConfig[]
  algorithm?: Record<string, unknown>
  candidateIterations?: unknown
  tier?: number
  annotations?: Record<string, unknown>
  output_contract?: string
}

export const ROUTING_STRATEGIES = ['priority', 'confidence'] as const
export type RoutingStrategy = (typeof ROUTING_STRATEGIES)[number]
export const DEFAULT_ROUTING_STRATEGY: RoutingStrategy = 'priority'

export interface RoutingConfig {
  signals?: ConfigSignals
  projections?: ConfigProjections
  decisions?: DecisionConfig[]
  strategy?: RoutingStrategy
}

export interface ConfigSignals {
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
  kb?: KBSignal[]
  metadata?: MetadataSignal[]
  classifiers?: ClassifierSignal[]
}

export interface ConfigProjections {
  partitions?: ProjectionPartition[]
  scores?: ProjectionScore[]
  mappings?: ProjectionMapping[]
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
  kb?: string
  metric?: string
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
  aggregation_method: string
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
  type: 'local' | 'llm'
  model?: string
  model_path?: string
  labels: string[]
  instructions?: string
  use_cpu?: boolean
}

export interface DomainSignal {
  name: string
  description: string
  mmlu_categories?: string[]
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

export interface KBSignal {
  name: string
  kb: string
  target: {
    kind: 'label' | 'group'
    value: string
  }
  match?: 'best' | 'threshold'
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

export interface StructureSignal {
  name: string
  description?: string
  feature: StructureFeature
  predicate?: NumericPredicate
}

export interface ComplexitySignal {
  name: string
  threshold: number
  hard: { candidates: string[] }
  easy: { candidates: string[] }
  description?: string
  composer?: {
    operator: 'AND' | 'OR' | 'NOT'
    conditions: DecisionCondition[]
  }
}

export interface JailbreakSignal {
  name: string
  threshold?: number
  method?: string
  include_history?: boolean
  jailbreak_patterns?: string[]
  benign_patterns?: string[]
  description?: string
}

export interface PIISignal {
  name: string
  threshold?: number
  pii_types_allowed?: string[]
  include_history?: boolean
  description?: string
}

/** Model-free Recipe document edited by the Config pages. */
export type ConfigData = RoutingConfig

export type SignalType =
  | 'Keywords'
  | 'Embeddings'
  | 'Domain'
  | 'Preference'
  | 'Fact Check'
  | 'User Feedback'
  | 'Reask'
  | 'Language'
  | 'Context'
  | 'Structure'
  | 'Complexity'
  | 'Modality'
  | 'Authz'
  | 'Jailbreak'
  | 'PII'
  | 'KB'
  | 'Metadata'
  | 'Classifier'

export interface DecisionFormState {
  name: string
  description: string
  priority: number
  operator: 'AND' | 'OR' | 'NOT'
  conditions: DecisionCondition[]
  plugins: { type: string; configuration: string | DecisionPluginConfiguration }[]
}

export interface AddSignalFormState {
  type: SignalType
  name: string
  description: string
  operator: 'AND' | 'OR'
  keywords: string[]
  case_sensitive: boolean
  threshold: number
  candidates: string[]
  aggregation_method: string
  mmlu_categories: string[]
  min_tokens?: string
  max_tokens?: string
  preference_examples?: string[]
  preference_threshold?: number
  lookback_turns?: number
  complexity_threshold?: number
  structure_feature?: StructureFeature
  structure_predicate?: NumericPredicate
  role?: string
  subjects?: Subject[]
  hard_candidates?: string[]
  easy_candidates?: string[]
  composer_operator?: 'AND' | 'OR' | 'NOT'
  composer_conditions?: DecisionCondition[]
  jailbreak_threshold?: number
  jailbreak_method?: string
  include_history?: boolean
  jailbreak_patterns?: string[]
  benign_patterns?: string[]
  pii_threshold?: number
  pii_types_allowed?: string[]
  pii_include_history?: boolean
  kb_name?: string
  target_kind?: 'label' | 'group'
  target_value?: string
  kb_match?: 'best' | 'threshold'
  metadata_key?: string
  metadata_predicate_type?: 'equals' | 'in' | 'exists'
  metadata_equals?: string
  metadata_in?: string[]
  metadata_exists?: boolean
  classifier_type?: 'local' | 'llm'
  classifier_model?: string
  classifier_model_path?: string
  classifier_labels?: string[]
  classifier_instructions?: string
  classifier_use_cpu?: boolean
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

export function decisionRulesForSave(
  existing: DecisionRuleSet | undefined,
  next: DecisionRuleSet,
): DecisionRuleSet {
  if (existing?.conditions.some(conditionHasNestedRules)) {
    return JSON.parse(JSON.stringify(existing)) as DecisionRuleSet
  }
  return next
}

export function cloneDecisionConditions(
  conditions: DecisionCondition[] | undefined,
): DecisionCondition[] {
  return JSON.parse(JSON.stringify(conditions || [])) as DecisionCondition[]
}

export function conditionHasNestedRules(condition: DecisionCondition): boolean {
  return Boolean(condition.operator || condition.conditions?.length)
}

export const formatThreshold = (value: number): string => `${Math.round(value * 100)}%`

export const TABLE_COLUMN_WIDTH = {
  compact: '140px',
  medium: '160px',
} as const

export type ConfigDecisionConditionType = DecisionConditionType
