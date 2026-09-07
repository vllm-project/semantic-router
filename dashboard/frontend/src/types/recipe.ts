export interface RecipeMetadataAuthor {
  name: string
  email?: string
  url?: string
}

export interface RecipeMetadata {
  schema_version: string
  id: string
  name: string
  version: string
  description: string
  authors?: Array<RecipeMetadataAuthor | string>
  license?: string
  tags?: string[]
  links?: Record<string, string>
}

export interface RecipeSourceFileHealth {
  present: boolean
  digest?: string
  size_bytes?: number
}

export interface RecipeSourceHealth {
  status: 'ready' | 'unmanaged' | 'invalid' | 'recovering' | 'inconsistent'
  files: Record<'metadata' | 'config' | 'probes' | 'dsl' | 'readme', RecipeSourceFileHealth>
  issues: string[]
}

export interface RecipeDigests {
  recipe?: string
  metadata?: string
  config?: string
  dsl?: string
  probes?: string
  readme?: string
}

export interface RecipeCounts {
  unified_models: number
  recipes: number
  decisions: number
  probes: number
}

export interface RecipeDescriptor {
  managed: boolean
  metadata?: RecipeMetadata
  readme?: string
  source_health: RecipeSourceHealth
  digests: RecipeDigests
  counts: RecipeCounts
  activation_state?: RecipeActivationState
  active_recipe_digest?: string
}

export type RecipeActivationState = 'none' | 'active' | 'recovering' | 'inconsistent'

export interface RecipePackageSummary {
  id: string
  name: string
  version: string
  description: string
  recipe_digest: string
  config_digest: string
  archive_sha256: string
  archive_verified: boolean
  installed_at: string
  active: boolean
  counts: RecipeCounts
  warnings: RecipePackageWarning[]
}

export interface RecipePackageWarning {
  code: string
  path: string
  message: string
}

export interface RecipePackageList {
  items: RecipePackageSummary[]
  page: number
  page_size: number
  total: number
  total_pages: number
  active_digest?: string
  activation_state: RecipeActivationState
}

export interface RecipePackageListFilters {
  page?: number
  pageSize?: number
  query?: string
}

export type RecipeActivationMode = 'hot_switch' | 'stack_recreation'

export interface RecipeActivationListener {
  name: string
  address: string
  port: number
  host_port: number
}

export interface RecipeActivationPlan {
  recipe_digest: string
  plan_digest: string
  mode: RecipeActivationMode
  requires_confirmation: boolean
  listeners_before: RecipeActivationListener[]
  listeners_after: RecipeActivationListener[]
  storage: {
    before: string[]
    after: string[]
    add: string[]
    remove: string[]
    repair: string[]
  }
  management_before?: {
    bind_address: string
    port: number
    host_port: number
  }
  management_after?: {
    bind_address: string
    port: number
    host_port: number
  }
  management_auth: {
    mode: string
    service_role?: string
    service_permissions: string[]
  }
  effects: string[]
}

export interface RecipeActivationConfirmation {
  expected_plan_digest: string
  confirm_stack_recreation: boolean
}

export interface RecipePackageActivationResult {
  status: 'active'
  recipe_digest: string
  previous_recipe_digest?: string
  plan_digest: string
  mode: RecipeActivationMode
}

export interface RecipePackageDeactivationResult {
  status: 'inactive'
  previous_recipe_digest?: string
  plan_digest?: string
  mode?: RecipeActivationMode
}

export type RecipePackageErrorCode =
  | 'invalid_request'
  | 'invalid_recipe'
  | 'package_not_found'
  | 'deploy_in_progress'
  | 'activation_failed'
  | 'activation_rollback_failed'
  | 'activation_inconsistent'
  | 'activation_confirmation_required'
  | 'activation_incompatible'
  | 'managed_storage_isolation_required'
  | 'environment_binding_required'
  | 'managed_recipe_active'
  | 'config_coordination_failed'
  | 'warnings_not_acknowledged'
  | 'runtime_config_readonly'
  | 'readonly_mode'

export type RecipeProbeRequestShape = 'text' | 'messages' | 'tools'

export interface RecipeProbeExpectedRoute {
  decision: string
  recipe?: string
  algorithm?: string
  alias?: string
  plugins: string[]
  forbidden_plugins: string[]
  plugin_match?: string
  signals: Record<string, string[]>
  forbidden_signals: Record<string, string[]>
  signal_match?: string
}

export interface RecipeProbeSummary {
  id: string
  decision_id: string
  variant_id: string
  query_preview: string
  model?: string
  display_prompt?: string
  tags: string[]
  request_shapes: RecipeProbeRequestShape[]
  expected: RecipeProbeExpectedRoute
  playground: {
    enabled: boolean
    reason?: string
  }
  editable: boolean
}

export interface RecipeProbeMessage extends Record<string, unknown> {
  role: string
  content: unknown
}

export interface RecipeProbeDetail extends RecipeProbeSummary {
  query?: string
  messages?: RecipeProbeMessage[]
  tools?: unknown[]
  tool_choice?: unknown
  repeat: number
  padding?: {
    text: string
    repeat: number
    placement: string
  }
  generated_text?: {
    message_index: number
    content_index: number
    target_text_bytes: number
    character: string
  }
  image_fixtures?: Record<
    string,
    {
      description: string
      media_type: string
      bytes: number
      sha256: string
    }
  >
  notes?: string
  raw_payload_hidden?: boolean
}

export interface RecipeProbeFacets {
  recipes: Record<string, number>
  decisions: Record<string, number>
  tags: Record<string, number>
  models: Record<string, number>
  shapes: Record<string, number>
}

export interface RecipeProbePage {
  items: RecipeProbeSummary[]
  page: number
  page_size: number
  total: number
  total_pages: number
  facets: RecipeProbeFacets
  recipe_digest: string
}

export interface RecipeProbeListFilters {
  page?: number
  pageSize?: number
  query?: string
  recipe?: string
  decision?: string
  tag?: string
  model?: string
  requestShape?: string
}

export interface RecipeProbeValidationResult {
  probe_id: string
  recipe_digest: string
  passed: boolean
  expected: RecipeProbeExpectedRoute
  actual: {
    decision?: string
    model?: string
    requested_model?: string
    selection_status?: string
    selection_method?: string
    selection_reason?: string
    recipe?: string
    algorithm?: string
    plugins: string[]
    recommended_models: string[]
    matched_signals: Record<string, string[]>
    trace_decisions: string[]
  }
  checks: Record<
    'decision' | 'model' | 'recipe' | 'algorithm' | 'plugins' | 'signals' | 'alias' | 'trace',
    boolean
  >
  failures: string[]
  provenance?: {
    status: 'verified' | 'unverified'
    reason?: string
    package_hash: string
    before: {
      source_config_hash?: string
      generated_runtime_hash?: string
      active_runtime_hash?: string
      activation_status?: string
    }
    after: {
      source_config_hash?: string
      generated_runtime_hash?: string
      active_runtime_hash?: string
      activation_status?: string
    }
  }
  latency_ms: number
  error?: string
}

export interface RecipeProbeRunPlan {
  probe_id: string
  recipe_digest: string
  recipe: string
  model?: string
  messages: RecipeProbeMessage[]
  tools?: unknown[]
  tool_choice?: unknown
  request: Record<string, unknown>
  editable: boolean
}
