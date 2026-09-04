import type {
  EvaluationCapacityLoadProtocol,
  EvaluationCapacitySLO,
  EvaluationSchemaVersion,
} from './evaluationPlane'

export interface EvaluationCapacityRepetition {
  concurrency: number
  requests: number
  successes: number
  errors: number
  elapsed_seconds: number
  throughput_rps: number
  latency_p95_ms: number
  error_rate: number
  error_rate_upper_bound: number
  repetition: number
}

export interface EvaluationCapacityLevel {
  concurrency: number
  warmup_requests: number
  warmup_errors: number
  warmup_elapsed_seconds: number
  measurement_requests: number
  successes: number
  errors: number
  elapsed_seconds: number
  throughput_rps: number
  throughput_cv: number
  latency_p50_ms: number
  latency_p95_ms: number
  latency_p99_ms: number
  latency_p95_cv: number
  error_rate: number
  error_rate_upper_bound: number
  measurement_cluster_count: number
  error_rate_cluster_range: number
  input_tokens: number
  output_tokens: number
  runtime_cost_usd: number
  repetitions: EvaluationCapacityRepetition[]
  throughput_scaling_efficiency: number | null
  warmup_passed: boolean
  latency_slo_passed: boolean
  cluster_coverage_passed: boolean
  error_rate_stability_passed: boolean
  error_slo_passed: boolean
  throughput_slo_passed: boolean
  scaling_slo_passed: boolean
  throughput_stability_passed: boolean
  latency_stability_passed: boolean
  qualified: boolean
}

export type EvaluationCapacityFailureReason =
  | 'required_concurrency'
  | 'warmup_errors'
  | 'latency_p95'
  | 'measurement_cluster_coverage'
  | 'error_rate_cluster_stability'
  | 'error_rate_upper_bound'
  | 'throughput'
  | 'throughput_scaling'
  | 'throughput_stability'
  | 'latency_stability'

export interface EvaluationCapacitySLOAssessment {
  qualified_concurrency: number | null
  saturation_concurrency: number | null
  slo_headroom: number
  verdict: 'pass' | 'fail'
  failure_reasons: EvaluationCapacityFailureReason[]
}

export interface EvaluationCapacityProfile {
  schema_version: EvaluationSchemaVersion
  kind: 'repeated-closed-loop-capacity'
  protocol: EvaluationCapacityLoadProtocol
  levels: EvaluationCapacityLevel[]
  slo: EvaluationCapacitySLO
  assessment: EvaluationCapacitySLOAssessment
}
