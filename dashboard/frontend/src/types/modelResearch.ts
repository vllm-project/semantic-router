import type { RouterModelsInfo } from '../utils/routerRuntime'

export type ModelResearchStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'stopped'
  | 'blocked'

export type ModelResearchGoalTemplate = 'improve_accuracy' | 'explore_signal'

export interface ModelResearchBudget {
  max_trials: number
}

export interface ModelResearchOverrides {
  api_base_override?: string
  request_model_override?: string
  dataset_override?: string
  hyperparameter_hints?: Record<string, unknown>
  allow_cpu_dry_run?: boolean
}

export interface ModelResearchCreateRequest {
  name: string
  goal_template: ModelResearchGoalTemplate
  target: string
  budget: ModelResearchBudget
  success_threshold_pp: number
  overrides?: ModelResearchOverrides
}

export interface ModelResearchEvent {
  timestamp: string
  kind: 'status' | 'log' | 'progress' | 'metric'
  level?: string
  message: string
  percent?: number
  trial_index?: number
}

export interface ModelResearchBaseline {
  label: string
  source: string
  runtime_name?: string
  model_path?: string
  model_id?: string
  state?: string
  description?: string
  categories?: string[]
  request_model?: string
}

export interface ModelResearchMetricSnapshot {
  source: string
  dataset: string
  accuracy: number
  f1?: number
  precision?: number
  recall?: number
  latency_avg_ms?: number
  output_path?: string
  model_id?: string
  improvement_pp?: number
}

export interface ModelResearchTrialResult {
  index: number
  name: string
  status: ModelResearchStatus
  started_at: string
  completed_at?: string
  params?: Record<string, unknown>
  model_path?: string
  use_lora?: boolean
  primary_metric: string
  eval?: ModelResearchMetricSnapshot
  runtime_eval?: ModelResearchMetricSnapshot
  error?: string
  artifacts?: Record<string, string>
}

export interface ModelResearchRecipeSummary {
  key: string
  label: string
  goal_templates: ModelResearchGoalTemplate[]
  default_dataset: string
  dataset_hint: string
  default_success_threshold_pp: number
  primary_metric: string
  supports_dataset_override: boolean
  supports_hyperparameter_hints: boolean
  baseline: ModelResearchBaseline
}

export interface ModelResearchCampaign {
  id: string
  name: string
  status: ModelResearchStatus
  goal_template: ModelResearchGoalTemplate
  target: string
  platform: string
  primary_metric: string
  success_threshold_pp: number
  budget: ModelResearchBudget
  created_at: string
  updated_at: string
  completed_at?: string
  default_api_base: string
  api_base: string
  default_request_model: string
  request_model: string
  overrides?: ModelResearchOverrides
  recipe: ModelResearchRecipeSummary
  baseline: ModelResearchBaseline
  baseline_eval?: ModelResearchMetricSnapshot
  runtime_baseline?: ModelResearchMetricSnapshot
  best_trial?: ModelResearchTrialResult
  trials?: ModelResearchTrialResult[]
  events?: ModelResearchEvent[]
  artifact_dir: string
  config_fragment_path?: string
  last_error?: string
  runtime_models?: RouterModelsInfo | null
}

export interface ModelResearchRecipesResponse {
  default_api_base: string
  default_request_model: string
  default_platform: string
  runtime_models?: RouterModelsInfo | null
  recipes: ModelResearchRecipeSummary[]
}

