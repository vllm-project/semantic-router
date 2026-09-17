/** The recorded routing explanation. These scores are selector units, not probabilities. */
export interface ReplayCandidateTrace {
  base_score?: number
  final_score?: number
  quality_gap?: number
  handoff_penalty?: number
  prefix_cache_penalty?: number
  tool_loop_penalty?: number
  switch_history_penalty?: number
  net_switch_advantage?: number
}

export interface ReplaySessionPolicy {
  identity?: {
    headers?: { session?: string; conversation?: string }
    session?: { source?: string; status?: string }
    conversation?: { source?: string; status?: string }
  }
  mode?: string
  scope?: string
  action?: string
  reason?: string
  phase?: string
  current_model?: string
  base_selected_model?: string
  selected_model?: string
  final_model?: string
  active_tool_loop?: boolean
  hard_locked?: boolean
  hard_lock_reason?: string
  decision_reason?: string
  idle_known?: boolean
  idle_for_seconds?: number
  idle_expired?: boolean
  missing_signals?: string[]
  candidate_models?: string[]
  base_scores?: Record<string, number>
  final_scores?: Record<string, number>
  candidate_traces?: Record<string, ReplayCandidateTrace>
}

export interface ReplayAdaptationScore {
  score?: number
  predicted_quality?: number
  cost_penalty?: number
  reliability_penalty?: number
  latency_adjustment?: number
  cache_adjustment?: number
}

export interface ReplayRouteDiagnostics {
  selection_reasoning?: string
  previous_model?: string
  proposal_model?: string
  selected_model?: string
  session_policy_applied?: boolean
  session_action?: string
  session_phase?: string
  session_reason?: string
  hard_lock_reason?: string
  decision_reason?: string
  signal_errors?: Record<string, string>
  prompt_helper_model?: string
  prompt_helper_latency_ms?: number
  request_demand_snapshots?: Array<{
    stage: string
    model?: string
    prompt_tokens: number
    reserved_output_tokens: number
    total_demand_tokens: number
    total_demand_known: boolean
    counting_source: string
    output_reserve_source: string
  }>
}

export interface InsightsTrajectoryRoute {
  conversation_id?: string
  record_id: string
  timestamp: string
  turn_index: number
  decision?: string
  selected_model?: string
  previous_model?: string
  selection_method?: string
  selection_reasoning?: string
  session_policy_applied: boolean
  session_action?: string
  session_reason?: string
  lifecycle_state: string
  response_status?: number
  duration_ms: number
}
