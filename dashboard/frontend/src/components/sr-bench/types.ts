export interface Benchmark {
  id: string
  title: string
  kind: string
  source_url?: string
  description?: string
}

export interface Dataset {
  id: string
  path: string
  sha256: string
  case_count: number
  seed?: number
  profile?: string
  benchmarks?: string[]
  name?: string
}

export interface Target {
  id: string
  kind: 'single' | 'mom'
  base_url: string
  model: string
  api_key_env?: string
  config_hash?: string
  capture_recipe?: boolean
  preview_url?: string
  max_inference_calls?: number
  prices?: Record<
    string,
    { input: number; cached_input: number; cache_write: number; output: number }
  >
}

export interface Manifest {
  version: 'sr-bench-1.0'
  name: string
  mode: 'live' | 'preview' | 'replay'
  cost_policy?: 'require_priced' | 'capability_only'
  profile: string
  seed: number
  targets: Target[]
  dataset?: { path: string; sha256: string }
  cases?: unknown[]
  limits: {
    concurrency: number
    total_timeout_s: number
    idle_timeout_s: number
    max_output_tokens: number
    max_output_chars: number
    repetition_window: number
    repetition_limit: number
    max_cost_usd: number
    max_run_seconds: number
    max_calls_per_case: number
  }
  sampling: { temperature: number; top_p: number; max_tokens: number; seed?: number }
}

export interface Catalog {
  version: string
  benchmarks: Benchmark[]
  profiles: Array<{ id: string; purpose: string }>
}

export interface TargetMetrics {
  id: string
  total?: number
  completed?: number
  failed?: number
  scored?: number
  correct?: number
  accuracy?: number | null
  cost_usd?: number | null
  tokens?: number | null | Record<string, number | null>
  latency_p50_s?: number | null
  latency_p95_s?: number | null
  ttft_p50_s?: number | null
  selected_models?: Record<string, number>
  decisions?: Record<string, number>
  selection_statuses?: Record<string, number>
  selection_reasons?: Record<string, number>
  [key: string]: unknown
}

export interface EvidencePage {
  total: number
  limit: number
  next_cursor: number | null
}

export interface PageState {
  total: number | null
  nextCursor: number | null
  loading: boolean
  error: string
}

export interface Run {
  id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted'
  created_at: string
  updated_at: string
  manifest: Manifest
  progress: { total: number; completed: number; failed: number }
  summary?: { targets?: TargetMetrics[]; wall_time_s?: number | null }
  error?: string | null
}

export interface CaseResult {
  case_id: string
  target_id: string
  benchmark: string
  status: string
  answer?: string | null
  correct?: boolean | null
  score?: number | null
  usage?: {
    input_tokens?: number
    cached_input_tokens?: number
    cache_write_tokens?: number
    output_tokens?: number
  }
  cost_usd?: number | null
  latency_s?: number | null
  ttft_s?: number | null
  details?: Record<string, unknown>
  error?: string | null
  [key: string]: unknown
}

export interface Report {
  version: string
  run_id: string
  status: string
  summary: { targets: TargetMetrics[]; wall_time_s?: number | null }
  benchmarks: Array<Record<string, unknown>>
  limitations: string[]
  provenance: Record<string, unknown>
  failure?: {
    case_id: string
    target_id: string
    reason: string
    inferred_from_saved_results?: boolean
  } | null
  [key: string]: unknown
}

export interface Plan {
  manifest: Manifest
  plan_sha256: string
  total: number
  status: string
}

export interface RunEvent {
  seq?: number
  sequence?: number
  type?: string
  event?: string
  timestamp?: string
  at?: string
  kind?: string
  [key: string]: unknown
}

export interface CallRecord {
  id: string
  target_id: string
  case_id: string
  role: string
  selected_model?: string
  model?: string
  decision?: string
  status: string
  [key: string]: unknown
}

export interface Comparison {
  version: string
  baseline_run_id: string
  candidate_run_id: string
  baseline_selection: string
  baseline_tied_best_target_ids?: string[]
  baseline_tie_policy?: string
  baseline_cost_comparison_eligible?: boolean
  baseline_cost_comparison_reason?: string | null
  comparisons: Array<{
    baseline_target_id: string
    candidate_target_id: string
    paired_cases: number
    quality_delta: number
    quality_delta_ci95: [number, number]
    cost_saving_percent: number | null
    baseline_cost_usd: number | null
    candidate_cost_usd: number | null
    wins: number
    losses: number
    ties: number
  }>
}

export interface RecoveryCell {
  case_id: string
  target_id: string
}
export interface RecoveryPlan {
  parent_run_id: string
  mode: 'undispatched' | 'failed'
  eligible_cells: RecoveryCell[]
  excluded: Array<RecoveryCell & { reason: string }>
  counts: { eligible: number; excluded: number }
  parent: {
    status: string
    progress: Run['progress']
    known_spend_usd: number | null
    spend_complete: boolean
  }
  plan_sha256: string
  new_attempt_budget_usd?: number
  requires_new_attempt_acknowledgment?: boolean
  scope?: string
}
export interface RecoveryRequest {
  mode: RecoveryPlan['mode']
  plan_sha256: string
  cells: RecoveryCell[]
  idempotency_key: string
  acknowledge_new_attempt?: boolean
}
