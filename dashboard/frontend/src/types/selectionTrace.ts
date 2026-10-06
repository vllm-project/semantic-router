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

/** Base objective evidence, before Learning or session protection. */
export interface SelectionTrace {
  stages: SelectionObjectiveStage[]
  final_survivors: SelectionCandidateRef[]
}
