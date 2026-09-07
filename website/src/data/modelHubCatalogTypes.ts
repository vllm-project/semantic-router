export type SupportTier = 'native' | 'compatible' | 'runtime'
export type ModelView = 'list' | 'table'
export type ModelKind = 'physical' | 'virtual'
export type Distribution = 'proprietary_api' | 'open_weights' | 'router_recipe'
export type ModelRelationship = 'first_party' | 'managed_cloud' | 'gateway' | 'self_hosted'
export type ReasoningTransport =
  | 'chat_template_kwargs'
  | 'top_level_effort'
  | 'top_level_boolean'
  | 'top_level_effort_template_switch'
  | 'top_level_effort_boolean_switch'
  | 'reasoning_object'
  | 'thinking_object'
  | 'thinking_object_effort'
  | 'output_config_effort'
  | 'deepseek_thinking'

export interface CatalogProtocol {
  id: string
  display_name: string
  default_base_path: string
  operations: Array<{ id: string, method: string, path: string }>
}

export interface CatalogPresentation {
  logo: string
  monogram: string
  monochrome: boolean
}

export interface CatalogModelBinding {
  catalog: string
  relationship: ModelRelationship
  id: string
  protocols: string[]
  reasoning_transport?: ReasoningTransport
  lifecycle: string
}

export interface CatalogProvider {
  id: string
  display_name: string
  description: string
  category: 'start_here' | 'model_api' | 'private_runtime'
  support_tier: SupportTier
  default_base_url?: string
  protocols: string[]
  default_protocol: string
  supported_operations: string[]
  path_overrides?: Record<string, string>
  reasoning_transport?: ReasoningTransport
  auth: { strategy: string }
  presentation: CatalogPresentation
  conformance: { status: string, verified_at?: string }
  models?: CatalogModelBinding[]
}

export interface VirtualRole {
  name: string
  required: boolean
  minimum_candidates: number
  recommended_pool: string[]
  traits: string[]
}

export interface CatalogModel {
  id: string
  display_name: string
  description: string
  kind: ModelKind
  publisher: string
  presentation: CatalogPresentation
  distribution: { type: Distribution, source: string, license?: string }
  family: string
  parameter_size?: string
  released_at?: string
  lifecycle: 'experimental' | 'active' | 'deprecated' | 'removed'
  limits?: { context_window_size?: number, max_output_tokens?: number }
  capabilities: string[]
  modalities: { input: string[], output: string[] }
  reasoning_family?: string
  verification: { status: string, verified_at: string, source?: string }
  entrypoint?: string
  recipe?: string
  policy_version?: string
  roles?: VirtualRole[]
  traits?: string[]
}

export interface BenchmarkMetric {
  id: string
  direction: 'higher_is_better' | 'lower_is_better'
  range: [number, number]
  unit: string
  normalization?: MetricNormalization
}

export interface MetricNormalization {
  type: 'identity' | 'one_minus' | 'linear_clamp' | 'piecewise_linear' | 'logistic' | 'lookup'
  min?: number
  max?: number
  k?: number
  x0?: number
  points?: Array<{ input: number, output: number }>
  values?: Record<string, number>
}

export interface CatalogBenchmark {
  id: string
  display_name: string
  domain: string
  tags?: string[]
  default_profile: string
  source?: string
  profiles: Array<{ id: string, display_name: string, description?: string }>
  metrics: BenchmarkMetric[]
}

export interface CatalogEvaluation {
  id: string
  model: string
  benchmark: string
  benchmark_profile: string
  reasoning_effort: string
  status: 'available' | 'missing'
  measured_at?: string
  observed_at?: string
  metrics?: Record<string, number | null>
  subject: Record<string, unknown>
  evidence: {
    provenance: 'vendor_claimed' | 'third_party' | 'vllm_sr_reproduced' | 'operator'
    verification: string
    source?: string
  }
}

export interface CatalogReasoningFamily {
  id: string
  type: 'chat_template_kwargs' | 'reasoning_effort' | 'reasoning_mode' | 'top_level_reasoning_effort'
  parameter: string
  activation_parameter?: string
  effort_flags?: Record<string, string>
  levels?: string[]
  default?: string
  modes?: Array<'enabled' | 'disabled' | 'adaptive'>
  default_mode?: 'enabled' | 'disabled' | 'adaptive'
  disabled?: string
}

export interface CatalogSnapshot {
  protocols: CatalogProtocol[]
  providers: CatalogProvider[]
  reasoning_families: CatalogReasoningFamily[]
  models: CatalogModel[]
  benchmarks: CatalogBenchmark[]
  evaluations: CatalogEvaluation[]
}

export interface BenchmarkRow {
  evaluation: CatalogEvaluation
  model: CatalogModel
  value: number
}

export type ProviderScope = 'mapped' | 'contract_only' | 'all'
