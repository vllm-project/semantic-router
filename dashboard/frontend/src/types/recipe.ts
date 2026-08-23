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
    package_config_hash: string
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
  model?: string
  messages: RecipeProbeMessage[]
  tools?: unknown[]
  request: Record<string, unknown>
  editable: boolean
}
