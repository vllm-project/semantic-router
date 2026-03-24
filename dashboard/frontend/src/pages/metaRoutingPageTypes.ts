export interface MetaRoutingTraceQuality {
  signal_dominance?: number
  avg_signal_confidence?: number
  decision_margin?: number
  projection_boundary_min_distance?: number
  fragile?: boolean
}

export interface MetaRoutingAssessment {
  needs_refine?: boolean
  triggers?: string[]
  root_causes?: string[]
  trace_quality?: MetaRoutingTraceQuality
}

export interface MetaRoutingActionPlan {
  type: string
  signal_families?: string[]
}

export interface MetaRoutingPlan {
  max_passes?: number
  trigger_names?: string[]
  root_causes?: string[]
  actions?: MetaRoutingActionPlan[]
}

export interface MetaRoutingPassTrace {
  index: number
  kind?: string
  latency_ms?: number
  input_compressed?: boolean
  decision_name?: string
  decision_confidence?: number
  decision_margin?: number
  decision_candidate_count?: number
  decision_winner_basis?: string
  runner_up_decision_name?: string
  runner_up_confidence?: number
  category_name?: string
  selected_model?: string
  selection_method?: string
  matched_signal_counts?: Record<string, number>
  partition_conflicts?: string[]
  trace_quality?: MetaRoutingTraceQuality
  assessment?: MetaRoutingAssessment
}

export interface MetaRoutingTrace {
  mode?: string
  max_passes?: number
  pass_count?: number
  trigger_names?: string[]
  refined_signal_families?: string[]
  overturned_decision?: boolean
  latency_delta_ms?: number
  decision_margin_delta?: number
  projection_boundary_delta?: number
  final_decision_name?: string
  final_decision_confidence?: number
  final_model?: string
  final_assessment?: MetaRoutingAssessment
  final_plan?: MetaRoutingPlan
  passes?: MetaRoutingPassTrace[]
}

export interface MetaRoutingFeedbackObservation {
  request_id?: string
  request_model?: string
  request_query?: string
  trace?: MetaRoutingTrace
}

export interface MetaRoutingFeedbackAction {
  planned?: boolean
  executed?: boolean
  executed_pass_count?: number
  executed_action_types?: string[]
  executed_signal_families?: string[]
  plan?: MetaRoutingPlan
}

export interface MetaRoutingFeedbackOutcome {
  final_decision_name?: string
  final_decision_confidence?: number
  final_model?: string
  response_status?: number
  streaming?: boolean
  cache_hit?: boolean
  pii_blocked?: boolean
  hallucination_detected?: boolean
  unverified_factual_response?: boolean
  response_jailbreak_detected?: boolean
  rag_backend?: string
  router_replay_id?: string
  user_feedback_signals?: string[]
}

export interface MetaRoutingFeedbackRecord {
  mode?: string
  observation: MetaRoutingFeedbackObservation
  action: MetaRoutingFeedbackAction
  outcome: MetaRoutingFeedbackOutcome
}

export interface MetaRoutingFeedbackSummary {
  id: string
  timestamp: string
  mode?: string
  request_id?: string
  request_model?: string
  request_query_preview?: string
  pass_count?: number
  planned?: boolean
  executed?: boolean
  executed_pass_count?: number
  trigger_names?: string[]
  root_causes?: string[]
  action_types?: string[]
  refined_signal_families?: string[]
  overturned_decision?: boolean
  latency_delta_ms?: number
  decision_margin_delta?: number
  projection_boundary_delta?: number
  final_decision_name?: string
  final_decision_confidence?: number
  final_model?: string
  response_status?: number
  streaming?: boolean
  cache_hit?: boolean
  pii_blocked?: boolean
  hallucination_detected?: boolean
  response_jailbreak_detected?: boolean
  router_replay_id?: string
  user_feedback_signals?: string[]
}

export interface MetaRoutingFeedbackListResponse {
  object: string
  count: number
  total: number
  limit: number
  offset: number
  has_more: boolean
  next_offset?: number
  data: MetaRoutingFeedbackSummary[]
}

export interface MetaRoutingFeedbackDetailResponse {
  object: string
  id: string
  timestamp: string
  record: MetaRoutingFeedbackRecord
}

export interface MetaRoutingAggregateValue {
  name: string
  value: number
}

export interface MetaRoutingAggregateSummary {
  planned_refinement_rate: number
  executed_refinement_rate: number
  overturn_rate: number
  average_latency_delta_ms: number
  p95_latency_delta_ms: number
  top_trigger?: string
  top_root_cause?: string
}

export interface MetaRoutingAggregateResponse {
  object: string
  record_count: number
  summary: MetaRoutingAggregateSummary
  mode_distribution: MetaRoutingAggregateValue[]
  trigger_distribution: MetaRoutingAggregateValue[]
  root_cause_distribution: MetaRoutingAggregateValue[]
  action_type_distribution: MetaRoutingAggregateValue[]
  signal_family_distribution: MetaRoutingAggregateValue[]
  decision_change_distribution: MetaRoutingAggregateValue[]
  decision_distribution: MetaRoutingAggregateValue[]
  model_distribution: MetaRoutingAggregateValue[]
  response_status_distribution: MetaRoutingAggregateValue[]
  available_modes: string[]
  available_triggers: string[]
  available_root_causes: string[]
  available_action_types: string[]
  available_signal_families: string[]
  available_decisions: string[]
  available_models: string[]
  available_response_statuses: number[]
}

export type MetaRoutingBooleanFilter = 'all' | 'true' | 'false'

export interface MetaRoutingQueryFilters {
  searchTerm: string
  mode: string
  trigger: string
  rootCause: string
  actionType: string
  signalFamily: string
  overturned: MetaRoutingBooleanFilter
  decision: string
  model: string
  responseStatus: string
}
