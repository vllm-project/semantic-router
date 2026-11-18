import type {
  InsightsTrajectoryRoute,
  ReplayAdaptationScore,
  ReplayRouteDiagnostics,
  ReplaySessionPolicy,
} from './insightsPageRoutingTypes'
import type { SignalType } from '../generated/routerConfigContract'
import type { InsightsReview, InsightsReviewAggregate } from './insightsReviewTypes'

export type Signal = Partial<Record<SignalType, string[]>> & Record<string, string[] | undefined>

export interface InsightsOutcome {
  timestamp?: string
  source: string
  target: string
  target_ref?: string
  verdict: string
  reason?: string
  score?: number
  metadata?: Record<string, string>
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
  output?: string
  status?: string
  content_redacted?: boolean
  truncated?: boolean
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

/**
 * The agent of its run that sent a request (routerreplay/store/agent.go). An
 * absent id is the run's main agent and a present one a subagent; redaction
 * keeps an id or parent_id as an empty string. kind is the subagent kind Codex
 * names, such as collab_spawn or review.
 */
export interface InsightsAgentIdentity {
  harness?: string
  id?: string
  parent_id?: string
  kind?: string
}

export interface InsightsRecord {
  provider_cache?: ProviderCacheAccounting
  conversation_id?: string
  id: string
  timestamp: string
  request_id?: string
  /**
   * codex_approval_review for an agent approval review, session_title for the
   * call a harness makes to title its session; absent otherwise.
   */
  request_kind?: string
  review?: InsightsReview
  session_id?: string
  /** The name the harness gave the session, on the call that named it. */
  session_name?: string
  agent?: InsightsAgentIdentity | null
  /** The client request a Looper hop was made for. */
  parent_request_id?: string
  previous_response_id?: string
  reasoning_effort?: string
  prompt?: string
  prompt_truncated?: boolean
  cached_prompt_tokens?: number
  /**
   * The Codex thread and turn that sent the request. parent_turn_id is the
   * turn that started the thread, such as the turn a review judges or a
   * subagent was spawned from.
   */
  thread_id?: string
  turn_id?: string
  parent_turn_id?: string
  turn_index: number
  recipe?: string
  decision?: string
  decision_tier: number
  decision_priority: number
  category?: string
  original_model?: string
  selected_model?: string
  reasoning_mode?: string
  confidence_score?: number | null
  confidence_score_available?: boolean
  selection_method?: string
  route_diagnostics?: ReplayRouteDiagnostics
  session_policy?: ReplaySessionPolicy
  learning?: {
    protection?: ReplaySessionPolicy
    protection_preflight?: ReplaySessionPolicy
    adaptation?: {
      mode?: string
      action?: string
      reason?: string
      scores?: Record<string, ReplayAdaptationScore>
    }
  }
  signals: Signal
  outcomes?: InsightsOutcome[]
  projections?: string[]
  projection_scores?: Record<string, number>
  projection_trace?: ProjectionTrace
  signal_confidences?: Record<string, number>
  signal_values?: Record<string, number>
  signal_error_matches?: Record<string, boolean>
  tool_trace?: ToolTrace
  request_body?: string
  response_body?: string
  response_status?: number
  /** A response header alone is not terminal; only completed records are successful. */
  lifecycle_state?: 'unknown' | 'in_progress' | 'completed' | 'aborted' | 'failed'
  ended_at?: string
  duration_ms?: number
  terminal_reason?: string
  from_cache?: boolean
  streaming?: boolean
  request_body_truncated?: boolean
  response_body_truncated?: boolean
  guardrails_enabled?: boolean
  jailbreak_enabled?: boolean
  pii_enabled?: boolean
  jailbreak_detected?: boolean
  jailbreak_type?: string
  jailbreak_confidence?: number | null
  jailbreak_score_available?: boolean
  pii_detected?: boolean
  pii_entities?: string[]
  pii_blocked?: boolean
  rag_enabled?: boolean
  rag_backend?: string
  rag_context_length?: number
  rag_similarity_score?: number
  // v0.4 demoted-header replay fields (#2200, #2254): recoverable here even when
  // x-vsr-cache-similarity / x-vsr-context-token-count are not emitted as headers.
  cache_similarity?: number
  context_token_count?: number
  hallucination_enabled?: boolean
  hallucination_detected?: boolean
  hallucination_score_available?: boolean
  hallucination_score_kind?: string
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
  cost_accounting?: InsightsCostAccounting
}

/** Model calls the Router made itself to serve a request. */
export interface InsightsRouterCallCost {
  model?: string
  kind: string
  calls: number
  prompt_tokens?: number
  completion_tokens?: number
  cost?: number
  currency?: string
  unpriced_calls?: number
  unpriced_reasons?: Record<string, number>
}

/** Where a record's cost figures came from and what they leave out. */
export interface InsightsCostAccounting {
  pricing_source?: 'configured' | 'catalog_list_price'
  caller_login?: boolean
  unpriced_reason?: string
  baseline_source?: 'configured_model' | 'recipe_most_expensive' | 'selected_model'
  baseline_unavailable?: string
  router_calls?: InsightsRouterCallCost[]
  budget?: InsightsBudgetAccounting
}

/** The tightest spend budget a request belongs to, as it left it. */
export interface InsightsBudgetAccounting {
  scope: string
  currency: string
  limit: number
  remaining: number
  estimate: number
  charged?: number
  settlement?: 'usage' | 'estimate' | 'released'
  unpriced_calls?: number
}

export interface TokenCostComponent {
  tokens?: number
  provenance: 'unknown' | 'estimated' | 'derived' | 'authoritative'
  cost?: number
}

export interface ProviderCacheAccounting {
  state: 'complete' | 'partial' | 'unavailable' | 'invalid'
  uncached_input: TokenCostComponent
  cache_read: TokenCostComponent
  cache_write: TokenCostComponent
  output: TokenCostComponent
  cache_write_5m: TokenCostComponent
  cache_write_1h: TokenCostComponent
  no_cache_cost?: number
  cache_savings?: number
}

export interface InsightsTrajectoryToolCall {
  id: string
  type: string
  function: {
    name: string
    arguments: string
  }
}

export interface InsightsTrajectoryMessage {
  conversation_id?: string
  role: 'user' | 'assistant' | 'tool'
  content?: string
  tool_calls?: InsightsTrajectoryToolCall[]
  tool_call_id?: string
  tool_name?: string
  status?: 'succeeded' | 'failed'
  content_redacted?: boolean
  content_truncated?: boolean
  turn_index: number
}

export interface InsightsTrajectory {
  object: 'router_replay.trajectory'
  session_id: string
  recipe?: string
  routes?: InsightsTrajectoryRoute[]
  record_count: number
  turn_count: number
  messages: InsightsTrajectoryMessage[]
}

export interface InsightsListResponse {
  object: string
  count: number
  total?: number
  limit?: number
  offset?: number
  has_more?: boolean
  next_offset?: number
  /** Records by harness under every filter but harness: a harness id, or api_client for none. */
  harness_counts?: Record<string, number> | null
  data: InsightsRecord[]
}

export type InsightsFilterType = 'all' | 'cached' | 'streamed'

export interface InsightsCurrencyCostSummary {
  totalSaved: number
  baselineSpend: number
  actualSpend: number
  currency: string
  costRecordCount: number
}

export interface InsightsCostSummary {
  totalSaved: number
  baselineSpend: number
  actualSpend: number
  currency?: string
  costRecordCount: number
  excludedRecordCount: number
  byCurrency?: InsightsCurrencyCostSummary[]
}

export interface InsightsAggregateValue {
  name: string
  value: number
}

export interface InsightsAggregateCurrencySummary {
  /** Savings net of the Router calls of the compared requests. */
  total_saved: number
  /** baseline_spend minus actual_spend, before those Router calls. */
  gross_saved?: number
  netted_router_call_spend?: number
  baseline_spend: number
  actual_spend: number
  router_call_spend?: number
  total_spend?: number
  currency: string
  cost_record_count: number
}

/** How much of the filtered traffic the cost figures leave out, and why. */
export interface InsightsAggregateCostCoverage {
  priced_records: number
  partly_priced_records: number
  unpriced_records: number
  unpriced_reasons?: Record<string, number>
  looper_hop_records: number
  list_price_records: number
  caller_login_records: number
  router_calls: number
  unpriced_router_calls: number
  baseline_sources?: Record<string, number>
  baseline_unavailable?: Record<string, number>
  /** Router calls netted out of savings, by kind. */
  netted_router_calls?: Record<string, number>
  /** Requests left out of savings for a Router call without a price, by why. */
  net_unavailable?: Record<string, number>
}

export interface InsightsAggregateSummary {
  /** Savings net of the Router calls of the compared requests. */
  total_saved: number
  /** baseline_spend minus actual_spend, before those Router calls. */
  gross_saved?: number
  netted_router_call_spend?: number
  baseline_spend: number
  actual_spend: number
  router_call_spend?: number
  total_spend?: number
  currency?: string
  cost_record_count: number
  excluded_record_count: number
  by_currency?: InsightsAggregateCurrencySummary[]
  coverage?: InsightsAggregateCostCoverage
}

export interface InsightsProviderCacheCurrencyTotal {
  currency: string
  no_cache_spend: number
  actual_spend: number
  net_savings: number
  priced_records: number
}

export interface InsightsProviderCacheSpendEntry extends InsightsProviderCacheCurrencyTotal {
  name: string
}

export interface InsightsAggregateProviderCache {
  no_cache_spend: number
  actual_spend: number
  net_savings: number
  currency?: string
  priced_records: number
  excluded_records: number
  by_currency?: InsightsProviderCacheCurrencyTotal[]
  by_decision: InsightsProviderCacheSpendEntry[]
  by_selected_model: InsightsProviderCacheSpendEntry[]
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
  lifecycle: {
    completed: number
    failed: number
    aborted: number
    in_progress: number
    unknown: number
  }
  summary: InsightsAggregateSummary
  provider_cache?: InsightsAggregateProviderCache
  model_selection: InsightsAggregateValue[]
  decision_distribution: InsightsAggregateValue[]
  signal_distribution: InsightsAggregateValue[]
  token_volume: InsightsAggregateTokenVolume
  token_breakdown: InsightsAggregateTokenBreakdown
  available_recipes: string[]
  available_decisions: string[]
  available_models: string[]
  reviews?: InsightsReviewAggregate
  available_review_models?: string[]
  /** Present when the aggregate was asked for an interval along with since. */
  traffic?: InsightsTrafficSeries
}

/** The aggregate's records in contiguous intervals from since to the query, oldest first. */
export interface InsightsTrafficSeries {
  since: string
  interval_seconds: number
  latency_p50_ms?: number
  latency_p95_ms?: number
  buckets: InsightsTrafficBucket[]
}

export interface InsightsTrafficBucket {
  start: string
  requests: number
  cached: number
  failed: number
  latency_p50_ms?: number
  latency_p95_ms?: number
  cost: InsightsAggregateCurrencySummary[]
}
