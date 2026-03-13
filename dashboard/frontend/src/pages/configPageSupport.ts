import type { Endpoint } from '../components/EndpointsEditor'
import type { DecisionConditionType } from '../types/config'

export interface VLLMEndpoint {
  name: string
  address: string
  port: number
  weight: number
  health_check_path: string
}

export interface ModelConfig {
  model_id: string
  use_modernbert?: boolean
  use_mmbert_32k?: boolean
  threshold: number
  use_cpu: boolean
  use_contrastive?: boolean
  embedding_model?: string
  category_mapping_path?: string
  pii_mapping_path?: string
  jailbreak_mapping_path?: string
}

export interface MCPCategoryModel {
  enabled: boolean
  transport_type: string
  command?: string
  args?: string[]
  env?: Record<string, string>
  url?: string
  tool_name?: string
  threshold: number
  timeout_seconds?: number
}

export interface ModelScore {
  model: string
  score: number
  use_reasoning: boolean
  reasoning_description?: string
  reasoning_effort?: string
}

export interface Category {
  name: string
  system_prompt?: string
  description?: string
  mmlu_categories?: string[]
  model_scores?: ModelScore[] | Record<string, number>
}

export interface ToolFunction {
  name: string
  description: string
  parameters: {
    type: string
    properties: Record<string, ToolParameterSchema>
    required?: string[]
  }
}

export interface ToolParameterSchema {
  type?: string
  description?: string
  [key: string]: unknown
}

export interface Tool {
  tool: {
    type: string
    function: ToolFunction
  }
  description: string
  category?: string
  tags?: string[]
}

export interface ReasoningFamily {
  type: string
  parameter: string
}

export interface ModelPricing {
  currency?: string
  prompt_per_1m?: number
  completion_per_1m?: number
}

export interface ModelConfigEntry {
  reasoning_family?: string
  preferred_endpoints?: string[]
  pricing?: ModelPricing
}

export interface NormalizedModel {
  name: string
  reasoning_family?: string
  endpoints: Endpoint[]
  access_key?: string
  pricing?: {
    currency?: string
    prompt_per_1m?: number
    completion_per_1m?: number
  }
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

export interface MemoryConfig {
  enabled?: boolean
  auto_store?: boolean
  milvus?: Record<string, unknown>
  embedding?: Record<string, unknown>
  default_retrieval_limit?: number
  default_similarity_threshold?: number
  extraction_batch_size?: number
}

export interface SemanticCacheConfig {
  enabled?: boolean
  backend_type?: string
  similarity_threshold?: number
  max_entries?: number
  ttl_seconds?: number
  eviction_policy?: string
  use_hnsw?: boolean
  hnsw_m?: number
  hnsw_ef_construction?: number
  embedding_model?: string
  max_memory_entries?: number
  backend_config_path?: string
}

export interface HallucinationMitigationConfig {
  enabled?: boolean
  fact_check_model?: Record<string, unknown>
  hallucination_model?: Record<string, unknown>
  nli_model?: Record<string, unknown>
}

export interface FeedbackDetectorConfig {
  enabled?: boolean
  model_id?: string
  threshold?: number
  use_cpu?: boolean
  use_mmbert_32k?: boolean
  use_modernbert?: boolean
}

export interface EmbeddingModelsConfig {
  qwen3_model_path?: string
  gemma_model_path?: string
  mmbert_model_path?: string
  multimodal_model_path?: string
  use_cpu?: boolean
  hnsw_config?: Record<string, unknown>
}

export interface ObservabilityConfig {
  tracing?: TracingConfig
  metrics?: { enabled?: boolean }
}

export interface LooperConfig {
  enabled?: boolean
  endpoint?: string
  timeout_seconds?: number
  headers?: Record<string, string>
}

export interface ModelSelectionConfig {
  enabled?: boolean
  default_algorithm?: string
  llm_candidates_path?: string
  models_path?: string
  training_data_path?: string
  knn?: Record<string, unknown>
  kmeans?: Record<string, unknown>
  svm?: Record<string, unknown>
  custom_training?: Record<string, unknown>
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
  examples?: string[]
  threshold?: number
}

export interface LanguageSignal {
  name: string
}

export interface ContextSignal {
  name: string
  min_tokens: string
  max_tokens: string
  description?: string
}

export interface ComplexitySignal {
  name: string
  threshold: number
  hard: { candidates: string[] }
  easy: { candidates: string[] }
  description?: string
  composer?: {
    operator: 'AND' | 'OR' | 'NOT'
    conditions: Array<{ type: string; name: string }>
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

export interface ConfigData {
  version?: string
  listeners?: Array<{
    name: string
    address: string
    port: number
    timeout?: string
  }>
  signals?: {
    keywords?: KeywordSignal[]
    embeddings?: EmbeddingSignal[]
    domains?: DomainSignal[]
    fact_check?: FactCheckSignal[]
    user_feedbacks?: UserFeedbackSignal[]
    preferences?: PreferenceSignal[]
    language?: LanguageSignal[]
    context?: ContextSignal[]
    complexity?: ComplexitySignal[]
    jailbreak?: JailbreakSignal[]
    pii?: PIISignal[]
  }
  decisions?: Array<{
    name: string
    description: string
    priority: number
    rules: {
      operator: 'AND' | 'OR' | 'NOT'
      conditions: Array<{ type: string; name: string }>
    }
    modelRefs: Array<{ model: string; use_reasoning: boolean }>
    plugins?: Array<{ type: string; configuration: Record<string, unknown> }>
  }>
  providers?: {
    models: Array<{
      name: string
      reasoning_family?: string
      endpoints: Array<{
        name: string
        weight: number
        endpoint: string
        protocol: 'http' | 'https'
      }>
      access_key?: string
      pricing?: {
        currency?: string
        prompt_per_1m?: number
        completion_per_1m?: number
      }
    }>
    default_model: string
    reasoning_families?: Record<string, ReasoningFamily>
    default_reasoning_effort?: string
  }
  bert_model?: ModelConfig
  semantic_cache?: SemanticCacheConfig
  tools?: {
    enabled: boolean
    top_k: number
    similarity_threshold: number
    tools_db_path: string
    fallback_to_empty: boolean
  }
  prompt_guard?: ModelConfig & { enabled: boolean }
  vllm_endpoints?: VLLMEndpoint[]
  classifier?: {
    category_model?: ModelConfig
    mcp_category_model?: MCPCategoryModel
    pii_model?: ModelConfig
    preference_model?: ModelConfig
  }
  categories?: (Category & { mmlu_categories?: string[] })[]
  default_reasoning_effort?: string
  default_model?: string
  model_config?: Record<string, ModelConfigEntry>
  reasoning_families?: Record<string, ReasoningFamily>
  response_api?: ResponseAPIConfig
  router_replay?: RouterReplayConfig
  memory?: MemoryConfig
  hallucination_mitigation?: HallucinationMitigationConfig
  feedback_detector?: FeedbackDetectorConfig
  external_models?: Array<Record<string, unknown>>
  embedding_models?: EmbeddingModelsConfig
  api?: APIConfig
  observability?: ObservabilityConfig
  looper?: LooperConfig
  clear_route_cache?: boolean
  model_selection?: ModelSelectionConfig
  keyword_rules?: KeywordSignal[]
  embedding_rules?: EmbeddingSignal[]
  fact_check_rules?: FactCheckSignal[]
  user_feedback_rules?: UserFeedbackSignal[]
  preference_rules?: PreferenceSignal[]
  language_rules?: LanguageSignal[]
  context_rules?: ContextSignal[]
  complexity_rules?: ComplexitySignal[]
  jailbreak?: JailbreakSignal[]
  pii?: PIISignal[]
  [key: string]: unknown
}

export type SignalType =
  | 'Keywords'
  | 'Embeddings'
  | 'Domain'
  | 'Preference'
  | 'Fact Check'
  | 'User Feedback'
  | 'Language'
  | 'Context'
  | 'Complexity'
  | 'Jailbreak'
  | 'PII'

export type DecisionConfig = NonNullable<ConfigData['decisions']>[number]

export interface DecisionFormState {
  name: string
  description: string
  priority: number
  operator: 'AND' | 'OR' | 'NOT'
  conditions: { type: string; name: string }[]
  modelRefs: { model: string; use_reasoning: boolean }[]
  plugins: { type: string; configuration: string | Record<string, unknown> }[]
}

export interface AddSignalFormState {
  type: SignalType
  name: string
  description: string
  operator: 'AND' | 'OR'
  keywords: string
  case_sensitive: boolean
  threshold: number
  candidates: string
  aggregation_method: string
  mmlu_categories: string
  min_tokens?: string
  max_tokens?: string
  preference_examples?: string
  preference_threshold?: number
  complexity_threshold?: number
  hard_candidates?: string
  easy_candidates?: string
  composer_operator?: 'AND' | 'OR' | 'NOT'
  composer_conditions?: string
  jailbreak_threshold?: number
  jailbreak_method?: string
  include_history?: boolean
  jailbreak_patterns?: string
  benign_patterns?: string
  pii_threshold?: number
  pii_types_allowed?: string
  pii_include_history?: boolean
}

export const formatThreshold = (value: number): string => {
  return `${Math.round(value * 100)}%`
}

export const normalizeModelScores = (
  modelScores: ModelScore[] | Record<string, number> | undefined
): ModelScore[] => {
  if (!modelScores) return []
  if (Array.isArray(modelScores)) return modelScores
  return Object.entries(modelScores).map(([model, score]) => ({
    model,
    score: typeof score === 'number' ? score : 0,
    use_reasoning: false,
  }))
}

export const normalizeEndpointProtocol = (protocol: unknown): Endpoint['protocol'] =>
  protocol === 'https' ? 'https' : 'http'

export const normalizeEndpoint = (
  endpoint: Partial<Endpoint> | undefined,
  index: number
): Endpoint => ({
  name: endpoint?.name?.trim() || `endpoint-${index + 1}`,
  endpoint: endpoint?.endpoint?.trim() || '',
  protocol: normalizeEndpointProtocol(endpoint?.protocol),
  weight:
    typeof endpoint?.weight === 'number' && Number.isFinite(endpoint.weight) ? endpoint.weight : 1,
})

export const normalizeEndpoints = (endpoints: Partial<Endpoint>[] | undefined): Endpoint[] =>
  Array.isArray(endpoints) ? endpoints.map((endpoint, index) => normalizeEndpoint(endpoint, index)) : []

export const TABLE_COLUMN_WIDTH = {
  compact: '140px',
  medium: '160px',
} as const

const SIGNAL_SECTION_KEYS = [
  'keywords',
  'embeddings',
  'domains',
  'fact_check',
  'user_feedbacks',
  'preferences',
  'language',
  'context',
  'complexity',
  'jailbreak',
  'pii',
] as const

type ConfigSignalSections = NonNullable<ConfigData['signals']>

export const collectConfiguredSignalNames = (signals?: ConfigData['signals']) => {
  if (!signals) {
    return new Set<string>()
  }

  return new Set(
    SIGNAL_SECTION_KEYS.flatMap((key) => ((signals as ConfigSignalSections)[key] || []).map((entry) => entry.name))
  )
}

export const clonePresetSignals = (signals?: Record<string, unknown>) => {
  if (!signals) {
    return undefined
  }

  return Object.fromEntries(
    Object.entries(signals).map(([key, value]) => [
      key,
      Array.isArray(value) ? value.map((item) => ({ ...item })) : value,
    ]),
  )
}

export const clonePresetDecisions = (decisions: DecisionConfig[]) =>
  decisions.map((decision) => ({
    ...decision,
    rules: {
      ...decision.rules,
      conditions: decision.rules.conditions.map((condition) => ({ ...condition })),
    },
    modelRefs: decision.modelRefs.map((modelRef) => ({ ...modelRef })),
    plugins: decision.plugins?.map((plugin) => ({
      ...plugin,
      configuration: { ...plugin.configuration },
    })),
  }))

export type ConfigDecisionConditionType = DecisionConditionType
