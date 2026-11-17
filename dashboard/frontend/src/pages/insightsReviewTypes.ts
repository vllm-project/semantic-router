/** Router Replay's record of one agent approval review (schema_version 1). */
export type InsightsReviewStatus =
  | 'pending'
  | 'completed'
  | 'tool_call'
  | 'invalid'
  | 'failed'
  | 'cancelled'

export interface InsightsReviewAssessment {
  outcome: 'allow' | 'deny'
  risk_level?: 'low' | 'medium' | 'high' | 'critical'
  user_authorization?: 'unknown' | 'low' | 'medium' | 'high'
  rationale?: string
  rationale_truncated?: boolean
}

export interface InsightsReviewAction {
  tool?: string
  summary?: string
  cwd?: string
  arguments?: string
  arguments_truncated?: boolean
}

export interface InsightsReview {
  schema_version: number
  identification_source: 'entrypoint' | 'content'
  reviewed_session_id?: string
  thread_id?: string
  turn_id?: string
  parent_turn_id?: string
  status: InsightsReviewStatus
  assessment?: InsightsReviewAssessment
  action?: InsightsReviewAction
  latency_ms?: number
  /** Set by the replay API when this viewer may not read the review's text. */
  content_redacted?: boolean
}

export interface InsightsReviewCost {
  currency: string
  amount: number
  requests: number
}

export interface InsightsReviewTimeBucket {
  start: string
  allowed: number
  denied: number
  invalid: number
  failed: number
  cancelled: number
}

export interface InsightsReviewModelEntry {
  model: string
  requests: number
  cost: InsightsReviewCost[]
}

export interface InsightsReviewAggregate {
  reviews: number
  allowed: number
  denied: number
  invalid: number
  failed: number
  cancelled: number
  pending: number
  tool_calls: number
  approval_rate?: number
  latency_p50_ms?: number
  latency_p95_ms?: number
  cost: InsightsReviewCost[]
  unpriced_requests: number
  bucket?: 'minute' | 'hour' | 'day'
  over_time: InsightsReviewTimeBucket[]
  by_model: InsightsReviewModelEntry[]
}
