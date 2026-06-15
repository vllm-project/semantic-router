export interface Signal {
  keyword?: string[]
  embedding?: string[]
  domain?: string[]
  fact_check?: string[]
  user_feedback?: string[]
  reask?: string[]
  preference?: string[]
  language?: string[]
  context?: string[]
  structure?: string[]
  complexity?: string[]
  modality?: string[]
  authz?: string[]
  jailbreak?: string[]
  pii?: string[]
  kb?: string[]
}

export interface ToolTraceStep {
  type: string
  source?: string
  role?: string
  text?: string
  tool_name?: string
  tool_call_id?: string
  arguments?: string
  raw_arguments?: string
  raw_output?: string
  status?: string
  content_redacted?: boolean
}

export interface ToolTrace {
  flow?: string
  stage?: string
  tool_names?: string[]
  steps?: ToolTraceStep[]
}

/** Router replay projection explainability payload (schema_version "1"). */
export interface ProjectionTracePartitionContender {
  name: string
  raw_score: number
  normalized_score?: number
}

export interface ProjectionTracePartition {
  group_name: string
  signal_type: string
  semantics?: string
  temperature?: number
  contenders?: ProjectionTracePartitionContender[]
  winner?: string
  winner_score?: number
  raw_winner_score?: number
  margin?: number
  default_used?: boolean
}

export interface ProjectionTraceOutputEval {
  name: string
  matched: boolean
  boundary_distance: number
}

export interface ProjectionTraceMapping {
  mapping_name: string
  source_score: string
  score_value: number
  selected_output?: string
  confidence?: number
  boundary_distance?: number
  outputs?: ProjectionTraceOutputEval[]
}

export interface ProjectionTraceScoreInput {
  type: string
  name?: string
  kb?: string
  metric?: string
  weight: number
  value: number
  contribution: number
}

export interface ProjectionTraceScore {
  name: string
  total: number
  inputs?: ProjectionTraceScoreInput[]
}

export interface ProjectionTrace {
  schema_version: string
  partitions?: ProjectionTracePartition[]
  scores?: ProjectionTraceScore[]
  mappings?: ProjectionTraceMapping[]
}

export interface InsightsRecord {
  id: string
  timestamp: string
  request_id?: string
  decision?: string
  decision_tier: number
  decision_priority: number
  category?: string
  original_model?: string
  selected_model?: string
  reasoning_mode?: string
  confidence_score?: number
  selection_method?: string
  signals: Signal
  projections?: string[]
  projection_scores?: Record<string, number>
  projection_trace?: ProjectionTrace
  signal_confidences?: Record<string, number>
  signal_values?: Record<string, number>
  tool_trace?: ToolTrace
  request_body?: string
  response_body?: string
  response_status?: number
  from_cache?: boolean
  streaming?: boolean
  request_body_truncated?: boolean
  response_body_truncated?: boolean
  guardrails_enabled?: boolean
  jailbreak_enabled?: boolean
  pii_enabled?: boolean
  jailbreak_detected?: boolean
  jailbreak_type?: string
  jailbreak_confidence?: number
  response_jailbreak_detected?: boolean
  response_jailbreak_type?: string
  response_jailbreak_confidence?: number
  pii_detected?: boolean
  pii_entities?: string[]
  pii_blocked?: boolean
  rag_enabled?: boolean
  rag_backend?: string
  rag_context_length?: number
  rag_similarity_score?: number
  hallucination_enabled?: boolean
  hallucination_detected?: boolean
  hallucination_confidence?: number
  hallucination_spans?: string[]
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  actual_cost?: number
  baseline_cost?: number
  cost_savings?: number
  currency?: string
  baseline_model?: string
}

export interface InsightsListResponse {
  object: string
  count: number
  total?: number
  limit?: number
  offset?: number
  has_more?: boolean
  next_offset?: number
  data: InsightsRecord[]
}

export type InsightsFilterType = 'all' | 'cached' | 'streamed'

export interface InsightsCostSummary {
  totalSaved: number
  baselineSpend: number
  actualSpend: number
  currency?: string
  costRecordCount: number
  excludedRecordCount: number
}

export interface InsightsAggregateValue {
  name: string
  value: number
}

export interface InsightsAggregateSummary {
  total_saved: number
  baseline_spend: number
  actual_spend: number
  currency?: string
  cost_record_count: number
  excluded_record_count: number
}

export interface InsightsAggregateTokenVolume {
  input_tokens: number
  output_tokens: number
  total_tokens: number
  excluded_record_count: number
}

export interface InsightsAggregateTokenEntry {
  name: string
  input_tokens: number
  output_tokens: number
  total_tokens: number
}

export interface InsightsAggregateTokenBreakdown {
  by_decision: InsightsAggregateTokenEntry[]
  by_selected_model: InsightsAggregateTokenEntry[]
}

export interface InsightsAggregateResponse {
  object: string
  record_count: number
  summary: InsightsAggregateSummary
  model_selection: InsightsAggregateValue[]
  decision_distribution: InsightsAggregateValue[]
  signal_distribution: InsightsAggregateValue[]
  token_volume: InsightsAggregateTokenVolume
  token_breakdown: InsightsAggregateTokenBreakdown
  available_decisions: string[]
  available_models: string[]
}
