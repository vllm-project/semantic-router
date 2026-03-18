export interface Signal {
  keyword?: string[]
  embedding?: string[]
  domain?: string[]
  fact_check?: string[]
  user_feedback?: string[]
  preference?: string[]
  language?: string[]
  context?: string[]
  complexity?: string[]
}

export interface InsightsRecord {
  id: string
  timestamp: string
  request_id?: string
  decision?: string
  category?: string
  original_model?: string
  selected_model?: string
  reasoning_mode?: string
  confidence_score?: number
  selection_method?: string
  signals: Signal
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
