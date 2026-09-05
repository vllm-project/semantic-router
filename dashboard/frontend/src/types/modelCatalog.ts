export type ModelCatalogChannel = 'latest' | 'release'
export type ProviderSupportTier = 'native' | 'compatible' | 'runtime'
export type ModelCatalogLifecycle = 'experimental' | 'active' | 'deprecated' | 'removed'
export type CatalogEvidenceStatus = 'claimed' | 'imported' | 'reproduced'
export type CatalogResultStatus =
  | 'available'
  | 'missing'
  | 'failed'
  | 'not_applicable'
  | 'withheld'

export interface BuiltInModelCatalogVersion {
  catalog_version: string
  channel: ModelCatalogChannel
  default_model: string
  enabled_models: string[]
  default_intelligence_index: string
}

export interface CatalogProtocolOperation {
  id: string
  method: 'GET' | 'POST' | 'DELETE'
  path: string
}

export interface CatalogProtocol {
  id: string
  display_name: string
  wire_format: string
  operations: CatalogProtocolOperation[]
  capabilities: string[]
}

export interface CatalogProvider {
  id: string
  display_name: string
  description: string
  category: 'start_here' | 'model_api' | 'private_runtime'
  support_tier: ProviderSupportTier
  default_base_url?: string
  protocols: string[]
  default_protocol: string
  supported_operations: string[]
  path_overrides?: Record<string, string>
  default_headers?: Record<string, string>
  reasoning_transport?:
    | 'chat_template_kwargs'
    | 'top_level_effort'
    | 'top_level_boolean'
    | 'reasoning_object'
    | 'thinking_object'
    | 'deepseek_thinking'
  api_version_query?: boolean
  auth: {
    strategy: 'none' | 'bearer' | 'api_key_header'
    header: string
    prefix: string
    injected_header?: string
  }
  presentation: {
    logo: string
    monogram: string
    monochrome: boolean
  }
  conformance: {
    status: 'unverified' | 'fixture_verified' | 'live_verified'
    verified_at?: string
  }
  models?: CatalogModelBinding[]
}

export interface CatalogModelBinding {
  catalog: string
  id: string
  protocols: string[]
  reasoning_transport?:
    | 'chat_template_kwargs'
    | 'top_level_effort'
    | 'top_level_boolean'
    | 'reasoning_object'
    | 'thinking_object'
    | 'deepseek_thinking'
  pricing?: Record<string, string | number | boolean>
  restrictions?: Record<string, unknown>
  lifecycle: ModelCatalogLifecycle
  verification: {
    status: CatalogEvidenceStatus
    verified_at?: string
    source?: string
  }
}

export interface CatalogReasoningFamily {
  id: string
  type: 'chat_template_kwargs' | 'reasoning_effort' | 'top_level_reasoning_effort'
  parameter: string
  levels: string[]
  default: string
  disabled?: string
}

export interface BuiltInModelRole {
  name: string
  required: boolean
  minimum_candidates: number
  traits: string[]
  recommended_pool: string[]
}

export interface BuiltInModelVerification {
  authority: string
  status: CatalogEvidenceStatus
  verified_at: string
  source?: string
  asset_sha256?: string
}

export interface CatalogPresentation {
  logo: string
  monogram: string
  monochrome: boolean
}

export interface CatalogModelDistribution {
  type: 'proprietary_api' | 'open_weights' | 'router_recipe'
  source: string
  license?: string
}

export interface BuiltInModelMetadata {
  id: string
  display_name: string
  description: string
  kind: 'physical' | 'virtual'
  publisher: string
  presentation: CatalogPresentation
  distribution: CatalogModelDistribution
  family: string
  parameter_size?: string
  revision?: string
  released_at?: string
  knowledge_cutoff?: string
  lifecycle: ModelCatalogLifecycle
  limits?: {
    context_window_size?: number
    max_output_tokens?: number
  }
  capabilities: string[]
  modalities: { input: string[]; output: string[] }
  reasoning_family?: string
  tags?: string[]
  generation?: number
  policy_version?: string
  asset?: string
  entrypoint?: string
  recipe?: string
  traits?: string[]
  roles?: BuiltInModelRole[]
  verification: BuiltInModelVerification
}

export interface CatalogBenchmarkMetric {
  id: string
  unit: string
  direction: 'higher_is_better' | 'lower_is_better'
  range: [number, number]
}

export interface CatalogBenchmark {
  id: string
  display_name: string
  domain: string
  source?: string
  default_profile: string
  profiles: Array<{ id: string; display_name: string; description: string }>
  metrics: CatalogBenchmarkMetric[]
}

export interface CatalogEvaluation {
  id: string
  model: string
  benchmark: string
  benchmark_profile: string
  reasoning_effort: string
  subject: Record<string, unknown>
  metrics: Record<string, number>
  status: CatalogResultStatus
  measured_at?: string
  evidence: {
    provenance: 'vendor_claimed' | 'third_party' | 'vllm_sr_reproduced' | 'operator'
    verification: CatalogEvidenceStatus
    source?: string
    artifact?: string
    redistributable: boolean
  }
}

export interface CatalogIndexComponent {
  benchmark?: string
  metric?: string
  benchmark_profile?: string
  index?: string
  weight: number
  normalization: {
    type:
      | 'identity'
      | 'one_minus'
      | 'linear_clamp'
      | 'piecewise_linear'
      | 'logistic'
      | 'lookup'
    min?: number
    max?: number
    k?: number
    x0?: number
    points?: Array<{ input: number; output: number }>
    values?: Record<string, number>
  }
}

export interface CatalogIndex {
  id: string
  display_name: string
  description: string
  methodology?: string
  aggregation: 'weighted_mean'
  scale: [number, number]
  missing: {
    policy: 'require_all' | 'require_coverage' | 'reported_only'
    minimum?: number
  }
  domains: Record<string, number>
  components: CatalogIndexComponent[]
}

export interface CatalogIndexResult {
  model: string
  reasoning_effort: string
  index: string
  status: CatalogResultStatus
  score: number | null
  coverage: number
  components: Array<{
    benchmark?: string
    metric?: string
    benchmark_profile?: string
    index?: string
    evaluation?: string
    weight: number
    status: CatalogResultStatus
    value?: number | null
    normalized?: number | null
  }>
  domains?: Record<string, number>
  provenance: string[]
}

export interface CatalogEvaluationCoverage {
  model: string
  reasoning_effort: string
  benchmark: string
  benchmark_profile: string
  metric: string
  status: CatalogResultStatus
  value?: number
  evaluation?: string
}

export interface BuiltInModelCatalog {
  schema_version: 'vllm-sr/model-catalog/v2'
  catalogs: BuiltInModelCatalogVersion[]
  protocols: CatalogProtocol[]
  providers: CatalogProvider[]
  reasoning_families: CatalogReasoningFamily[]
  models: BuiltInModelMetadata[]
  benchmarks: CatalogBenchmark[]
  evaluations: CatalogEvaluation[]
  evaluation_coverage: CatalogEvaluationCoverage[]
  indices: CatalogIndex[]
  index_results: CatalogIndexResult[]
}
