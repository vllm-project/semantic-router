import type {
  EvaluationAttestationRevision,
  EvaluationChangeProfileId,
  EvaluationGateContractVersion,
  EvaluationRun,
  EvaluationSchemaVersion,
  EvaluationTrackId,
  EvaluationTrackStatus,
  EvaluationSummaryVerdict,
  EvidenceLevel,
  GateDisposition,
  GateVerdict,
} from './evaluationPlane'
import type { EvaluationMethodReport } from './evaluationMethodReport'
import type { EvaluationRoutingRecipeReport } from './evaluationRoutingRecipeReport'

export interface EvaluationCoverage {
  evaluated: number
  total: number
  fraction: number
  unavailable?: number
  confidence_level?: number
  confidence_interval?: [number, number]
}

export interface EvaluationMetric {
  id: string
  name: string
  track_id?: EvaluationTrackId
  value: number | null
  unit: string
  direction?: 'higher_is_better' | 'lower_is_better' | 'target'
  baseline_value?: number | null
  delta?: number | null
  confidence_interval?: [number, number]
  sample_count?: number
  analysis_provenance: EvaluationMetricAnalysisProvenance
}

export interface EvaluationMetricAnalysisProvenance {
  contract_version: 'metric-analysis.v1'
  estimator_id: string
  estimator_version: string
  analysis_unit: string
  cluster_unit: string
  weighting:
    | 'inverse_propensity'
    | 'uniform_arm'
    | 'uniform_arm_pair'
    | 'uniform_assignment'
    | 'uniform_attempt'
    | 'uniform_case'
    | 'uniform_level'
    | 'uniform_observation'
    | 'uniform_pair'
    | 'uniform_repetition'
    | 'uniform_request'
    | 'uniform_task'
    | 'uniform_tool_call'
    | 'unweighted'
  missingness: 'fail_closed'
  exclusion_policy: 'exclude_unavailable_evidence'
  observed_exclusions: number
}

export interface EvaluationGate {
  id: string
  name: string
  description?: string
  track_id?: EvaluationTrackId
  disposition: GateDisposition
  verdict: GateVerdict
  change_profile: EvaluationChangeProfileId
  contract_version: EvaluationGateContractVersion
  evidence_refs: string[]
  evidence_level?: EvidenceLevel
  observed?: number | null
  threshold?: {
    operator: string
    value: number
    unit?: string
  }
  sample_count?: number
  coverage?: EvaluationCoverage
  owner?: string
  evaluated_at?: string
  rationale?: string
}

export interface EvaluationArtifact {
  id: string
  name: string
  kind: string
  uri?: string
  digest?: string
  media_type?: string
  size_bytes?: number
}

export interface EvaluationProvenance {
  schema_version: EvaluationSchemaVersion
  generated_at: string
  code_revision?: string
  benchmark_revisions?: Record<string, string>
  workload_snapshot_digest?: string
  policy_snapshot_digest?: string
  binding_snapshot_digest?: string
  pool_snapshot_digest?: string
  environment_snapshot_digest?: string
  target_id: string
  seed: number
  redaction_policy?: string
}

export interface EvaluationCostAmount {
  amount: number | null
  currency: string
  input_tokens?: number
  output_tokens?: number
  gpu_seconds?: number
  energy_kwh?: number
}

export interface EvaluationCostLedgers {
  runtime: EvaluationCostAmount
  evaluation_overhead: EvaluationCostAmount
  capacity_tco: EvaluationCostAmount
}

export interface EvaluationTrackReport {
  track_id: EvaluationTrackId
  status: EvaluationTrackStatus
  evidence_level: EvidenceLevel
  summary: string
  coverage: EvaluationCoverage
  metrics: EvaluationMetric[]
  gates: EvaluationGate[]
  artifacts?: EvaluationArtifact[]
  error?: string
}

export interface EvaluationReportSummary {
  verdict: EvaluationSummaryVerdict
  quality_score: number | null
  latency_p95_ms: number | null
  runtime_cost: number | null
  capacity_tco: number | null
  coverage: EvaluationCoverage
  passed_gates: number
  failed_gates: number
  unavailable_gates: number
}

export interface EvaluationReport {
  schema_version: EvaluationSchemaVersion
  attestation_revision: EvaluationAttestationRevision
  run: EvaluationRun
  summary: EvaluationReportSummary
  tracks: EvaluationTrackReport[]
  metrics: EvaluationMetric[]
  gates: EvaluationGate[]
  costs: EvaluationCostLedgers
  recommendations: string[]
  provenance: EvaluationProvenance
  artifacts: EvaluationArtifact[]
  method_reports: EvaluationMethodReport[]
  routing_recipe_report: EvaluationRoutingRecipeReport | null
}

export interface EvaluationFailureSummaryRow {
  track_id: EvaluationTrackId
  succeeded: number
  failed: number
  unavailable: number
}

export interface EvaluationFailureSummary {
  schema_version: EvaluationSchemaVersion
  total_records: number
  failed: number
  unavailable: number
  by_track: EvaluationFailureSummaryRow[]
}
