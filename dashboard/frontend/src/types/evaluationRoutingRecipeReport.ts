export interface EvaluationRoutingRecipeLatencyReport {
  available: boolean
  reason?: string
  sample_count: number
  p50_ms?: number
  p95_ms?: number
}

export interface EvaluationRoutingRecipeInputAvailabilityReport {
  id: string
  expected: number
  present: number
  missing: number
  error: number
  timeout: number
  latency: EvaluationRoutingRecipeLatencyReport
}

export interface EvaluationRoutingRecipeMetricAvailability {
  available: boolean
  reason?: string
  value?: number
  sample_count: number
}

export interface EvaluationRoutingRecipeReliabilityBin {
  lower: number
  upper: number
  count: number
  mean_prediction?: number
  observed_frequency?: number
}

export interface EvaluationRoutingRecipeProjectionOutcomeReport {
  projection_id: string
  spearman: EvaluationRoutingRecipeMetricAvailability
  brier: EvaluationRoutingRecipeMetricAvailability
  ece_10: EvaluationRoutingRecipeMetricAvailability
  reliability_bins: EvaluationRoutingRecipeReliabilityBin[]
}

export interface EvaluationRoutingRecipeTopKReport {
  k: number
  feasible_oracle_recall: EvaluationRoutingRecipeMetricAvailability
}

export interface EvaluationRoutingRecipeReport {
  contract_version: 'routing-recipe-eval.v1'
  plan_digest: string
  e1: {
    expected_decisions: number
    observed_decisions: number
    signals: EvaluationRoutingRecipeInputAvailabilityReport[]
    projections: EvaluationRoutingRecipeInputAvailabilityReport[]
    eligibility_complete: number
    selected_feasible: number
  }
  e2: {
    projection_outcomes: EvaluationRoutingRecipeProjectionOutcomeReport[]
    top_k: EvaluationRoutingRecipeTopKReport[]
    oracle_regret: EvaluationRoutingRecipeMetricAvailability
  }
}
