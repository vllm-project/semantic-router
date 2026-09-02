import type {
  EvaluationAttestationRevision,
  EvaluationSchemaVersion,
  EvaluationSummaryVerdict,
  EvaluationTrackId,
} from './evaluationPlane'
import type { EvaluationGate, EvaluationMetric } from './evaluationReport'

export interface EvaluationComparison {
  schema_version: EvaluationSchemaVersion
  attestation_revision: EvaluationAttestationRevision
  baseline_run_id: string
  candidate_run_id: string
  verdict: EvaluationSummaryVerdict
  summary: string
  metrics: EvaluationMetric[]
  statistics: EvaluationComparisonStatistic[]
  gates: EvaluationGate[]
  recommendations: string[]
  created_at: string
}

export interface EvaluationComparisonStatistic {
  id: string
  track_id: EvaluationTrackId
  /** Server-owned paired delta estimator; distinct from Metric point-estimate provenance. */
  estimator_id: 'paired-bootstrap-case-clustered-delta'
  estimator_version: 'v1'
  analysis_unit: 'case_mean' | 'case_max' | 'case_oracle_regret' | 'case_normalized_regret'
  direction: 'higher_is_better' | 'lower_is_better'
  non_inferiority_margin: number
  baseline_value: number
  candidate_value: number
  delta: number
  confidence_level: number
  delta_confidence_interval: number[]
  candidate_confidence_interval: number[]
  sample_count: number
  verdict: EvaluationSummaryVerdict
}
