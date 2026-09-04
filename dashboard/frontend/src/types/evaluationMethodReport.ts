import type { EvidenceLevel } from './evaluationPlane'

export interface EvaluationMethodCurvePoint {
  action: { id: string }
  budget: number
  mean_score: number
  case_count: number
}

export type EvaluationMethodReadiness =
  | 'native-qualified'
  | 'exploratory-import'
  | 'data-required'
  | 'blocked'

export interface EvaluationMethodSlice {
  schema_version: 'evaluation-method.v2'
  id: string
}

export interface EvaluationMethodAnalysisPlan {
  schema_version: 'evaluation-method.v2'
  id: string
  analysis_unit: string
  cluster_unit: string
  slices: EvaluationMethodSlice[]
  curve_domain: 'shared_budget' | 'not_applicable'
  missingness: 'fail_closed'
}

export interface EvaluationMethodDescriptor {
  schema_version: 'evaluation-method.v2'
  id: string
  version: 'evaluation-method.v2'
  status: EvaluationMethodReadiness
  execution_owner: 'server' | 'worker' | 'provider' | 'benchmark_native'
  input_schema: string
  export_schema: string
  live_input_complete: boolean
  live_grader: boolean
  applicable_tracks: string[]
  live_tracks: string[]
  produced_metric_ids: string[]
  evidence_ceiling: EvidenceLevel
  native_parity: 'native' | 'source_qualified' | 'none'
  required_artifact_ids: string[]
  analysis_plan: EvaluationMethodAnalysisPlan
}

export interface EvaluationMethodReport {
  method: EvaluationMethodDescriptor
  analysis_plan: EvaluationMethodAnalysisPlan
  action_refs: EvaluationMethodSlice[]
  slice_refs: EvaluationMethodSlice[]
  raw_shared_domain_curve: EvaluationMethodCurvePoint[]
  audc: number
  nauc: number
  peak: number
  qnc: number
  missing_case_action_budget_cells: number
}
