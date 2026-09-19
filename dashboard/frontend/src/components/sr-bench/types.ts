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
  split?: string
  custom_subset?: boolean
}

export interface DatasetDetail {
  id: string
  name?: string
  profile?: string
  split?: string
  custom_subset?: boolean
  case_count: number
  benchmarks: Array<{
    id: string
    title: string
    count: number
    categories: Array<{ name: string; count: number }>
  }>
  categories: Array<{ benchmark: string; name: string; count: number }>
  provenance: {
    sha256?: string
    seed?: number
    selection?: unknown
    sources: Array<{
      benchmark: string
      url?: string
      revision?: string
      revision_verification?: string
      normalizer?: string
      access_note?: string
      license_url?: string
      file_count?: number
    }>
  }
}

export interface DatasetCase {
  id: string
  benchmark: string
  category?: string
  question: string
  messages: Array<{ role: string; content: string }>
  choices?: string[]
  input_status: 'available' | 'unavailable'
  input_notice?: string
}

export interface DatasetCasePage {
  dataset_id: string
  cases: DatasetCase[]
  total: number
  dataset_total: number
  next_cursor: string | null
  limit: number
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
  request_params?: Record<string, unknown>
  prices?: Record<
    string,
    { input: number; cached_input: number; cache_write: number; output: number }
  >
}

export interface Manifest {
  version: 'sr-bench-1.0'
  case_sha256?: string
  benchmark_weights?: Record<string, number>
  adapter_versions?: Record<string, string>
  benchmark_options?: Record<string, unknown>
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
    case_timeout_s?: number
  }
  sampling: { temperature: number; top_p: number; max_tokens: number; seed?: number }
  preview_context?: { session_id?: string; conversation_id?: string; sampling_seed?: number }
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
  cache_neutral_cost_usd?: number | null
  cache_neutral_cost_basis?: string
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

export interface AccountingCorrectionReceipt {
  id: string
  version: string
  created_at: string
  evidence_sha256: string
  qualified: boolean
  corrected_call_count: number
  verified_call_count: number
  unverifiable_call_count: number
  original_known_spend_usd: number
  corrected_known_spend_usd: number
  original_receipts_preserved: boolean
  model_requests: number
}

export interface Report {
  version: string
  run_id: string
  status: string
  summary: {
    targets: TargetMetrics[]
    wall_time_s?: number | null
    total_spend_usd?: number | null
  }
  benchmarks: Array<Record<string, unknown>>
  limitations: string[]
  provenance: Record<string, unknown> & {
    accounting_correction?: AccountingCorrectionReceipt | null
  }
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
    quality_delta_ci95_method?: string
    quality_delta_ci95_qualification?: string
    quality_delta_bootstrap_ci95?: [number, number]
    cost_saving_percent: number | null
    baseline_cost_usd: number | null
    candidate_cost_usd: number | null
    cache_neutral_baseline_cost_usd?: number | null
    cache_neutral_candidate_cost_usd?: number | null
    cache_neutral_cost_saving_percent?: number | null
    cache_neutral_cost_basis?: string
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
