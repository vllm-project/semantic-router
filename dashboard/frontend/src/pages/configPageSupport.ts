import type { BackendEngine, ProviderLBPolicy } from '../generated/routerConfigContract'
import type { ModelBindings } from './modelBindings'
import type { CanonicalGlobalConfig, ModelPricing } from './configPageGlobalTypes'
import type { Endpoint } from '../components/EndpointsEditor'
import bundledCatalog from '../modelCatalogMetadata'
import type { Listener, SafetySignal, SystemOneSignal } from '../types/config'
import type { BuiltInModelCatalog, CatalogBenchmark, CatalogIndex } from '../types/modelCatalog'
export { BACKEND_ENGINES, PROVIDER_LB_POLICIES } from '../generated/routerConfigContract'
export type { BackendEngine, ProviderLBPolicy } from '../generated/routerConfigContract'
export type * from './configPageGlobalTypes'

export const PROVIDER_LB_POLICY_LABELS: Record<ProviderLBPolicy, string> = {
  round_robin: 'Round robin',
  least_request: 'Least requests',
  load_aware: 'Least engine load',
}

export type ListenerConfig = Listener

export interface ModelScore {
  model: string
  score: number
  use_reasoning: boolean
  reasoning_description?: string
  reasoning_mode?: 'enabled' | 'disabled' | 'adaptive'
  reasoning_effort?: string
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
  activation_parameter?: string
  effort_flags?: Record<string, string>
  levels?: string[]
  default?: string
  modes?: Array<'enabled' | 'disabled' | 'adaptive'>
  default_mode?: 'enabled' | 'disabled' | 'adaptive'
  disabled?: string
}

export interface ProviderReliability {
  lb_policy?: ProviderLBPolicy
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

export interface LoRAAdapter {
  name: string
  description?: string
}

/** A backend's credential, the connection block a deployment uses. */
export interface BackendRefAuth {
  // delegated sends the requesting user's credential
  // (global.services.authz.delegated_credential) instead of a key, caller the
  // provider login the client itself sent.
  credential?: 'delegated' | 'caller'
  header?: string
  prefix?: string
  api_key_env?: string
  api_key_file?: string
}

/** How an https backend is verified, by Envoy and by the Router's own calls. */
export interface BackendRefTLS {
  // Replaces the system roots for this backend.
  ca_file?: string
  server_name?: string
  cert_file?: string
  key_file?: string
}

export interface BackendRefEntry {
  name?: string
  /** host:port with protocol, or a full URL. */
  endpoint?: string
  protocol?: 'http' | 'https'
  weight?: number
  provider?: string
  api_version?: string
  chat_path?: string
  api_key?: string
  auth?: BackendRefAuth
  headers?: Record<string, string>
  tls?: BackendRefTLS
  engine?: BackendEngine
  metrics?: BackendMetrics
}

export interface BackendMetrics {
  port?: number
  path?: string
}

export interface ModelReasoningConfig {
  family?: string
  type?: string
  parameter?: string
  activation_parameter?: string
  effort_flags?: Record<string, string>
  levels?: string[]
  default?: string
  modes?: Array<'enabled' | 'disabled' | 'adaptive'>
  default_mode?: 'enabled' | 'disabled' | 'adaptive'
  disabled?: string
}

export interface EvaluationRecordConfig {
  model: string
  benchmark: string
  benchmark_profile?: string
  reasoning_effort?: string
  metrics: Record<string, number>
  source?: string
  measured_at?: string
  metadata?: Record<string, string | number | boolean | null>
}

export interface ProviderModelConfig {
  name: string
  catalog?: string
  reasoning?: ModelReasoningConfig
  provider_model_id?: string
  api_format?: string
  external_model_ids?: Record<string, string>
  backend_refs?: BackendRefEntry[]
  endpoints?: Array<{
    name: string
    weight: number
    endpoint: string
    protocol: 'http' | 'https'
  }>
  access_key?: string
  pricing?: ModelPricing
  reliability?: ProviderReliability
}

export interface ProviderDefaultsConfig {
  model?: string
  reasoning_effort?: string
  /** The CA bundle of every https backend whose own tls names none. */
  tls?: { ca_file: string }
}

export interface ProvidersConfig {
  defaults?: ProviderDefaultsConfig
  models: ProviderModelConfig[]
}

export interface RoutingModelCard {
  name: string
  display_name?: string
  publisher?: string
  presentation?: {
    logo: string
    monogram: string
    monochrome: boolean
  }
  distribution?: {
    type: 'proprietary_api' | 'open_weights' | 'router_recipe'
    source: string
    license?: string
  }
  family?: string
  revision?: string
  released_at?: string
  knowledge_cutoff?: string
  lifecycle?: 'experimental' | 'active' | 'deprecated' | 'removed'
  param_size?: string
  context_window_size?: number
  max_output_tokens?: number
  description?: string
  capabilities?: string[]
  modalities?: { input: string[]; output: string[] }
  loras?: LoRAAdapter[]
  tags?: string[]
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

// The root is a combination, a single leaf condition, or empty (unconditional).
export interface DecisionRuleSet extends DecisionCondition {
  on_unknown?: 'no_match' | 'match' | 'fail_request'
}

export interface DecisionModelRef {
  model: string
  use_reasoning: boolean
  reasoning_description?: string
  reasoning_mode?: '' | 'enabled' | 'disabled' | 'adaptive'
  reasoning_effort?: string
  lora_name?: string
  weight?: number
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
  modelRefs: DecisionModelRef[]
  plugins?: DecisionPluginConfig[]
  algorithm?: Record<string, unknown>
  action?: { type: string; destination: string }
  adaptations?: Record<string, unknown>
  reliability?: Record<string, unknown>
  fallback?: Record<string, unknown>
  output_contract_spec?: Record<string, unknown>
  candidateIterations?: Array<Record<string, unknown>>
  retention?: Record<string, unknown>
  output?: DecisionOutputConfig
  tier?: number
  annotations?: Record<string, unknown>
  output_contract?: string
}

// The media a decision's answers carry: one kind or several, generated or an
// edit of the media the request provides. Without it a decision answers text.
export interface DecisionOutputConfig {
  kind?: DecisionOutputKind | DecisionOutputKind[]
  action?: 'generate' | 'edit'
}

export type DecisionOutputKind = 'text' | 'image' | 'audio' | 'video'

// The decision editor edits kind as a list; a single kind is written as the
// scalar the configuration documents.
export function decisionOutputForEditor(output?: DecisionOutputConfig): Record<string, unknown> {
  if (!output) return {}
  const { kind, ...rest } = output
  return kind === undefined ? rest : { ...rest, kind: Array.isArray(kind) ? kind : [kind] }
}

export function decisionOutputFromEditor(value: unknown): DecisionOutputConfig | undefined {
  if (!value || typeof value !== 'object') return undefined
  const { kind, action } = value as { kind?: unknown; action?: unknown }
  const kinds = (Array.isArray(kind) ? kind : [kind])
    .filter((entry): entry is string => typeof entry === 'string' && entry.trim() !== '')
    .map((entry) => entry.trim() as DecisionOutputKind)
  const output: DecisionOutputConfig = {}
  if (kinds.length > 0) output.kind = kinds.length === 1 ? kinds[0] : kinds
  if (action === 'generate' || action === 'edit') output.action = action
  return Object.keys(output).length > 0 ? output : undefined
}

export const ROUTING_STRATEGIES = ['priority', 'confidence'] as const
export type RoutingStrategy = (typeof ROUTING_STRATEGIES)[number]
export const DEFAULT_ROUTING_STRATEGY: RoutingStrategy = 'priority'

export interface RoutingConfig {
  model_bindings?: ModelBindings
  modelCards?: RoutingModelCard[]
  signals?: ConfigSignals
  projections?: ConfigProjections
  decisions?: DecisionConfig[]
  strategy?: RoutingStrategy
}

export interface EntrypointConfig {
  /** Another entrypoint model used for agent approval reviews. */
  review_model?: string
  /** Another entrypoint model used for agent memory extraction and consolidation. */
  memory_model?: string
  model_names: string[]
  recipe: string
  api?: 'chat' | 'systemone'
  /** Marks the entrypoint of the default recipe when there are several recipes. */
  default?: boolean
}

export interface RecipeRoutingConfig {
  model_bindings?: ModelBindings
  signals?: ConfigSignals
  projections?: ConfigProjections
  decisions?: DecisionConfig[]
  strategy?: RoutingStrategy
}

export interface RecipeConfig {
  name: string
  description?: string
  routing: RecipeRoutingConfig
}

export interface NormalizedModel {
  name: string
  catalog?: string
  reasoning?: ModelReasoningConfig
  reasoning_family?: string
  reasoning_modes?: Array<'enabled' | 'disabled' | 'adaptive'>
  reasoning_efforts?: string[]
  provider_model_id?: string
  api_format?: string
  api_format_override?: string
  external_model_ids?: Record<string, string>
  backend_refs?: BackendRefEntry[]
  endpoints: Endpoint[]
  param_size?: string
  context_window_size?: number
  description?: string
  capabilities?: string[]
  loras?: LoRAAdapter[]
  tags?: string[]
  card_override?: RoutingModelCard
  modalities?: { input: string[]; output: string[] }
  pricing?: {
    currency?: string
    prompt_per_1m?: number
    cached_input_per_1m?: number
    cache_write_per_1m?: number
    cache_write_1h_per_1m?: number
    completion_per_1m?: number
  }
  reliability?: ProviderReliability
}

export interface ConfigSignals {
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
  refusal?: RefusalSignal[]
  turn_relation?: TurnRelationSignal[]
  pii?: PIISignal[]
  metadata?: MetadataSignal[]
  conversation?: ConversationSignal[]
  input_modality?: InputModalitySignal[]
  agent?: AgentSignal[]
  systemone?: SystemOneSignal[]
}

export interface ConfigProjections {
  partitions?: ProjectionPartition[]
  scores?: ProjectionScore[]
  mappings?: ProjectionMapping[]
}

export interface DecisionPluginConfiguration {
  [key: string]: unknown
}

export interface KeywordSignal {
  name: string
  description?: string
  operator: 'AND' | 'OR' | 'all' | 'any'
  keywords: string[]
  case_sensitive?: boolean
  method?: 'regex' | 'bm25' | 'ngram'
  fuzzy_match?: boolean
  fuzzy_threshold?: number
  bm25_threshold?: number
  ngram_threshold?: number
  ngram_arity?: number
  include_history?: boolean
  count?: NumericPredicate
  density?: NumericPredicate
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

export interface DomainSignal {
  name: string
  description: string
  labels?: string[]
  model_scores?: ModelScore[]
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
}

export interface AgentSignal {
  name: string
  description?: string
  role?: 'main' | 'subagent'
  harness?: 'claude_code' | 'codex'
  types?: string[]
  request_classes?: Array<'main' | 'subagent' | 'workflow' | 'compaction' | 'auxiliary'>
}

export interface InputModalitySignal {
  name: string
  description?: string
  modality: 'text' | 'image' | 'audio' | 'video'
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

export interface ModalitySignal {
  name: string
  description?: string
  labels?: string[]
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
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

export interface FactCheckSignal {
  name: string
  description: string
  labels?: string[]
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
}

export interface HallucinationSignal {
  name: string
  description?: string
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
}

export interface RefusalSignal {
  name: string
  description?: string
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
}

export interface UserFeedbackSignal {
  name: string
  description: string
  labels?: string[]
  predicate?: NumericPredicate
  deployment?: string
  timeout_ms?: number
}

export interface TurnRelationSignal {
  name: string
  description?: string
  /** continue, reask, shift or return; a rule without labels matches nothing itself. */
  labels?: string[]
  /** How many turns before the current one the rule reads (default 4). */
  input?: { previous_turns?: number }
  deployment?: string
  timeout_ms?: number
}

export interface LanguageSignal {
  name: string
  description?: string
  threshold?: number
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

export interface ConversationSource {
  type: string
  role?: string
}

export interface ConversationFeature {
  type: string
  source: ConversationSource
}

export interface ConversationSignal {
  name: string
  description?: string
  feature: ConversationFeature
  predicate?: NumericPredicate
}

export interface ComplexitySignal {
  name: string
  /** The deployment that answers this rule; without one, its binding or the decision model does. */
  deployment?: string
  /** easy, medium or hard; a rule without labels matches nothing itself. */
  labels?: string[]
  timeout_ms?: number
  description?: string
}

export interface JailbreakSignal {
  name: string
  predicate?: NumericPredicate
  stage?: 'request' | 'response'
  input?: { history?: boolean }
  deployment?: string
  timeout_ms?: number
  on_unknown?: 'match' | 'no_match'
  description?: string
}

export interface PIISignal {
  name: string
  predicate?: NumericPredicate
  ignore_labels?: string[]
  input?: { history?: boolean; tool_results?: boolean; request?: boolean }
  deployment?: string
  timeout_ms?: number
  on_unknown?: 'match' | 'no_match'
  description?: string
}

export interface ConfigData {
  version?: string
  listeners?: ListenerConfig[]
  signals?: ConfigSignals
  projections?: ConfigProjections
  decisions?: DecisionConfig[]
  providers?: ProvidersConfig
  evaluation?: {
    benchmarks?: CatalogBenchmark[]
    indices?: CatalogIndex[]
    records?: EvaluationRecordConfig[]
  }
  routing?: RoutingConfig
  /** The name and description of the recipe the manager edits as routing. */
  default_recipe?: { name: string; description?: string }
  entrypoints?: EntrypointConfig[]
  recipes?: RecipeConfig[]
  global?: CanonicalGlobalConfig
}

export type SignalType = string

export interface DecisionFormState {
  name: string
  description: string
  priority: number
  rules: DecisionRuleSet
  modelRefs: DecisionModelRef[]
  plugins: { type: string; configuration: string | DecisionPluginConfiguration }[]
  tier?: number
  output_contract?: string
  output_contract_spec: Record<string, unknown>
  action: Record<string, unknown>
  algorithm?: Record<string, unknown>
  adaptations: Record<string, unknown>
  reliability: Record<string, unknown>
  fallback: Record<string, unknown>
  declarative: Record<string, unknown>
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

export const formatThreshold = (value: number): string => {
  return `${Math.round(value * 100)}%`
}

export const normalizeEndpointProtocol = (protocol: unknown): Endpoint['protocol'] =>
  protocol === 'https' ? 'https' : 'http'

export const normalizeEndpoint = (
  endpoint: Partial<Endpoint> | undefined,
  index: number,
): Endpoint => ({
  name: endpoint?.name?.trim() || `endpoint-${index + 1}`,
  endpoint: endpoint?.endpoint?.trim() || '',
  protocol: normalizeEndpointProtocol(endpoint?.protocol),
  weight:
    typeof endpoint?.weight === 'number' && Number.isFinite(endpoint.weight) ? endpoint.weight : 1,
})

export const normalizeEndpoints = (endpoints: Partial<Endpoint>[] | undefined): Endpoint[] =>
  Array.isArray(endpoints)
    ? endpoints.map((endpoint, index) => normalizeEndpoint(endpoint, index))
    : []

export const normalizeProviderModelEndpoints = (model: {
  endpoints?: Partial<Endpoint>[]
  backend_refs?: BackendRefEntry[]
}): Endpoint[] => {
  if (Array.isArray(model.backend_refs) && model.backend_refs.length > 0) {
    return model.backend_refs.map((backend, index) => {
      const endpoint = typeof backend.endpoint === 'string' ? backend.endpoint.trim() : ''
      const protocol = backend.protocol || (endpoint.startsWith('https://') ? 'https' : 'http')
      return normalizeEndpoint(
        {
          name: backend.name,
          endpoint,
          protocol,
          weight: backend.weight,
        },
        index,
      )
    })
  }
  return normalizeEndpoints(model.endpoints)
}

export const TABLE_COLUMN_WIDTH = {
  compact: '140px',
  medium: '160px',
} as const

export const getDefaultModelName = (config: ConfigData | null): string =>
  config?.providers?.defaults?.model || ''

export const getReasoningFamiliesMap = (
  catalog: BuiltInModelCatalog | null = bundledCatalog as unknown as BuiltInModelCatalog,
): Record<string, ReasoningFamily> =>
  Object.fromEntries(
    (catalog?.reasoning_families ?? []).map((family) => [
      family.id,
      {
        type: family.type,
        parameter: family.parameter,
        activation_parameter: family.activation_parameter,
        effort_flags: family.effort_flags ? { ...family.effort_flags } : undefined,
        levels: [...(family.levels ?? [])],
        default: family.default,
        modes: family.modes ? [...family.modes] : undefined,
        default_mode: family.default_mode,
        disabled: family.disabled,
      },
    ]),
  )
