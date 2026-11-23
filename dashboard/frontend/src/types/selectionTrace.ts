/** JSON contract from pkg/selectiontrace; candidate refs retain config.ModelRef casing. */
export interface SelectionCandidateRef {
  Model: string
  LoRAName: string
  Weight: number
  UseReasoning: boolean | null
  ReasoningDescription: string
  ReasoningMode: string
  ReasoningEffort: string
}

export interface SelectionCandidateValue {
  candidate: SelectionCandidateRef
  value?: number
  metric?: string
  elimination_reason?: 'missing_measurement' | 'outside_tolerance'
}

export interface SelectionObjectiveStage {
  factor: string
  metric?: string
  percentile?: number
  action: 'applied' | 'skipped'
  reason: 'tolerance_band' | 'no_available_values' | 'incomplete_latency_coverage'
  tolerance: number
  available: number
  total: number
  candidates: SelectionCandidateValue[]
}

export type SelectionEstimateStatus = 'available' | 'unavailable' | 'deadline_exceeded'

/** One candidate's estimate; its values are present only when available. */
export interface SelectionCandidateEstimate {
  candidate: SelectionCandidateRef
  status: SelectionEstimateStatus
  reason?: string
  success_probability?: number
  calibrated: boolean
  expected_output_tokens?: number
  estimated_cost?: number
  currency?: string
  source: string
  deployment?: string
}

/** Per-request candidate estimates; observe mode records them only. */
export interface SelectionCandidateEstimates {
  mode: 'observe'
  candidates: SelectionCandidateEstimate[]
}

/**
 * Base objective evidence, before Learning or session protection. A weighted
 * objective records no stages or survivors, only candidate estimates.
 */
export interface SelectionTrace {
  stages?: SelectionObjectiveStage[]
  final_survivors?: SelectionCandidateRef[]
  estimates?: SelectionCandidateEstimates
}
