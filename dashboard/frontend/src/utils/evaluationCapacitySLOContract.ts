import type {
  EvaluationCapacityLoadProtocol,
  EvaluationCapacitySLO,
  EvaluationMode,
  EvaluationTrackId,
} from '../types/evaluationPlane'
import { EVALUATION_SCHEMA_VERSION } from '../types/evaluationPlane'
import {
  hasOnlyEvaluationFields,
  isEvaluationRecord,
  isFiniteNumber,
} from './evaluationContractValidation'

const CAPACITY_SLO_FIELDS = [
  'schema_version',
  'required_concurrency',
  'max_latency_p95_ms',
  'max_error_rate',
  'min_throughput_rps',
  'min_throughput_scaling_efficiency',
] as const

const CAPACITY_LOAD_PROTOCOL_FIELDS = [
  'schema_version',
  'kind',
  'concurrency_levels',
  'warmup_request_multiplier',
  'measurement_requests_per_repetition',
  'repetitions_per_level',
  'minimum_measurement_clusters_per_level',
  'confidence_level',
  'max_error_rate_cluster_range',
  'max_throughput_cv',
  'max_latency_p95_cv',
] as const

export function capacityConcurrencyLevels(maximum: number): number[] {
  if (!Number.isSafeInteger(maximum) || maximum < 2 || maximum > 128) {
    throw new Error('Capacity concurrency must be an integer between 2 and 128.')
  }
  const levels = [1]
  for (let level = 2; level < maximum; level *= 2) levels.push(level)
  levels.push(maximum)
  return levels
}

export function defaultEvaluationCapacityLoadProtocol(
  maximum: number,
): EvaluationCapacityLoadProtocol {
  return {
    schema_version: EVALUATION_SCHEMA_VERSION,
    kind: 'closed-loop',
    concurrency_levels: capacityConcurrencyLevels(maximum),
    warmup_request_multiplier: 2,
    measurement_requests_per_repetition: 100,
    repetitions_per_level: 3,
    minimum_measurement_clusters_per_level: 3,
    confidence_level: 0.95,
    max_error_rate_cluster_range: 0.05,
    max_throughput_cv: 0.2,
    max_latency_p95_cv: 0.2,
  }
}

export function requiresCapacitySLO(
  mode: EvaluationMode,
  trackIDs: readonly EvaluationTrackId[],
): boolean {
  return mode === 'live' && trackIDs.includes('capacity')
}

export function decodeEvaluationCapacitySLO(
  value: unknown,
  label = 'Capacity SLO',
): EvaluationCapacitySLO {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, CAPACITY_SLO_FIELDS) ||
    value.schema_version !== EVALUATION_SCHEMA_VERSION ||
    !Number.isSafeInteger(value.required_concurrency) ||
    (value.required_concurrency as number) < 1 ||
    (value.required_concurrency as number) > 128 ||
    !isFiniteNumber(value.max_latency_p95_ms) ||
    value.max_latency_p95_ms <= 0 ||
    !isFiniteNumber(value.max_error_rate) ||
    value.max_error_rate < 0 ||
    value.max_error_rate >= 1 ||
    !isFiniteNumber(value.min_throughput_rps) ||
    value.min_throughput_rps <= 0 ||
    !isFiniteNumber(value.min_throughput_scaling_efficiency) ||
    value.min_throughput_scaling_efficiency <= 0 ||
    value.min_throughput_scaling_efficiency > 1
  ) {
    throw new Error(
      `${label} must define bounded concurrency, p95 latency, error rate, throughput, and scaling efficiency.`,
    )
  }
  return {
    schema_version: EVALUATION_SCHEMA_VERSION,
    required_concurrency: value.required_concurrency as number,
    max_latency_p95_ms: value.max_latency_p95_ms,
    max_error_rate: value.max_error_rate,
    min_throughput_rps: value.min_throughput_rps,
    min_throughput_scaling_efficiency: value.min_throughput_scaling_efficiency,
  }
}

export function equalEvaluationCapacitySLO(
  left: EvaluationCapacitySLO | undefined,
  right: EvaluationCapacitySLO | undefined,
): boolean {
  if (!left || !right) return left === right
  return CAPACITY_SLO_FIELDS.every((field) => left[field] === right[field])
}

export function decodeEvaluationCapacityLoadProtocol(
  value: unknown,
  maximum: number,
  label = 'Capacity load protocol',
): EvaluationCapacityLoadProtocol {
  const expectedLevels = capacityConcurrencyLevels(maximum)
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, CAPACITY_LOAD_PROTOCOL_FIELDS) ||
    value.schema_version !== EVALUATION_SCHEMA_VERSION ||
    value.kind !== 'closed-loop' ||
    !Array.isArray(value.concurrency_levels) ||
    value.concurrency_levels.length !== expectedLevels.length ||
    value.concurrency_levels.some(
      (level, index) => level !== expectedLevels[index] || !Number.isSafeInteger(level),
    ) ||
    typeof value.warmup_request_multiplier !== 'number' ||
    !Number.isSafeInteger(value.warmup_request_multiplier) ||
    value.warmup_request_multiplier < 2 ||
    value.warmup_request_multiplier > 4 ||
    typeof value.measurement_requests_per_repetition !== 'number' ||
    !Number.isSafeInteger(value.measurement_requests_per_repetition) ||
    value.measurement_requests_per_repetition < 100 ||
    value.measurement_requests_per_repetition > 500 ||
    typeof value.repetitions_per_level !== 'number' ||
    !Number.isSafeInteger(value.repetitions_per_level) ||
    value.repetitions_per_level < 3 ||
    value.repetitions_per_level > 5 ||
    value.minimum_measurement_clusters_per_level !== 3 ||
    value.minimum_measurement_clusters_per_level > value.repetitions_per_level ||
    value.confidence_level !== 0.95 ||
    value.max_error_rate_cluster_range !== 0.05 ||
    !isFiniteNumber(value.max_throughput_cv) ||
    value.max_throughput_cv <= 0 ||
    value.max_throughput_cv > 0.2 ||
    !isFiniteNumber(value.max_latency_p95_cv) ||
    value.max_latency_p95_cv <= 0 ||
    value.max_latency_p95_cv > 0.2
  ) {
    throw new Error(
      `${label} must use the platform geometric ladder, repeated measurement window, 95% confidence, and bounded stability CV.`,
    )
  }
  return {
    schema_version: EVALUATION_SCHEMA_VERSION,
    kind: 'closed-loop',
    concurrency_levels: [...expectedLevels],
    warmup_request_multiplier: value.warmup_request_multiplier as number,
    measurement_requests_per_repetition: value.measurement_requests_per_repetition as number,
    repetitions_per_level: value.repetitions_per_level as number,
    minimum_measurement_clusters_per_level: 3,
    confidence_level: 0.95,
    max_error_rate_cluster_range: 0.05,
    max_throughput_cv: value.max_throughput_cv,
    max_latency_p95_cv: value.max_latency_p95_cv,
  }
}

export function equalEvaluationCapacityLoadProtocol(
  left: EvaluationCapacityLoadProtocol | undefined,
  right: EvaluationCapacityLoadProtocol | undefined,
): boolean {
  if (!left || !right) return left === right
  return CAPACITY_LOAD_PROTOCOL_FIELDS.every((field) =>
    field === 'concurrency_levels'
      ? left.concurrency_levels.length === right.concurrency_levels.length &&
        left.concurrency_levels.every((level, index) => level === right.concurrency_levels[index])
      : left[field] === right[field],
  )
}
