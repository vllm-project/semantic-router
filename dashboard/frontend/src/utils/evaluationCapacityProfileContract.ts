import type {
  EvaluationCapacityFailureReason,
  EvaluationCapacityLevel,
  EvaluationCapacityProfile,
  EvaluationCapacityRepetition,
  EvaluationCapacitySLOAssessment,
} from '../types/evaluationCapacityReport'
import type {
  EvaluationCapacityLoadProtocol,
  EvaluationCapacitySLO,
} from '../types/evaluationPlane'
import {
  decodeEvaluationCapacityLoadProtocol,
  decodeEvaluationCapacitySLO,
  equalEvaluationCapacityLoadProtocol,
  equalEvaluationCapacitySLO,
} from './evaluationCapacitySLOContract'
import {
  approximatelyEqual,
  booleanValue,
  boundedInteger,
  invalid,
  nonNegativeFiniteNumber,
  positiveFiniteNumber,
  recordWithExactKeys,
} from './evaluationDiagnosticArtifactValidation'

const ARTIFACT_NAME = 'capacity-profile.json'
const FAILURE_REASONS: EvaluationCapacityFailureReason[] = [
  'required_concurrency',
  'warmup_errors',
  'latency_p95',
  'measurement_cluster_coverage',
  'error_rate_cluster_stability',
  'error_rate_upper_bound',
  'throughput',
  'throughput_scaling',
  'throughput_stability',
  'latency_stability',
]

function arithmeticMean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function sampleCV(values: number[]): number {
  const mean = arithmeticMean(values)
  if (mean === 0) return 0
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1)
  return Math.sqrt(Math.max(variance, 0)) / mean
}

function oneSidedWilsonUpper(events: number, total: number): number {
  const z = 1.6448536269514722
  const estimate = events / total
  const z2 = z * z
  const denominator = 1 + z2 / total
  const center = estimate + z2 / (2 * total)
  const margin = z * Math.sqrt((estimate * (1 - estimate)) / total + z2 / (4 * total ** 2))
  return Math.min(1, (center + margin) / denominator)
}

function decodeRepetition(value: unknown, path: string): EvaluationCapacityRepetition {
  const row = recordWithExactKeys(
    value,
    [
      'concurrency',
      'repetition',
      'requests',
      'successes',
      'errors',
      'elapsed_seconds',
      'throughput_rps',
      'latency_p95_ms',
      'error_rate',
      'error_rate_upper_bound',
    ],
    ARTIFACT_NAME,
    path,
  )
  const requests = boundedInteger(row.requests, ARTIFACT_NAME, `${path}.requests`, 1)
  const successes = boundedInteger(row.successes, ARTIFACT_NAME, `${path}.successes`)
  const errors = boundedInteger(row.errors, ARTIFACT_NAME, `${path}.errors`)
  const elapsed = positiveFiniteNumber(
    row.elapsed_seconds,
    ARTIFACT_NAME,
    `${path}.elapsed_seconds`,
  )
  const throughput = positiveFiniteNumber(
    row.throughput_rps,
    ARTIFACT_NAME,
    `${path}.throughput_rps`,
  )
  const errorRate = ratio(row.error_rate, `${path}.error_rate`)
  const errorUpper = ratio(row.error_rate_upper_bound, `${path}.error_rate_upper_bound`)
  if (successes + errors !== requests || !approximatelyEqual(throughput, requests / elapsed)) {
    invalid(ARTIFACT_NAME, `${path} counts or throughput do not match its request window`)
  }
  if (
    !approximatelyEqual(errorRate, errors / requests) ||
    !approximatelyEqual(errorUpper, oneSidedWilsonUpper(errors, requests))
  ) {
    invalid(ARTIFACT_NAME, `${path} error statistics do not match its independent cluster`)
  }
  return {
    concurrency: boundedInteger(row.concurrency, ARTIFACT_NAME, `${path}.concurrency`, 1, 128),
    repetition: boundedInteger(row.repetition, ARTIFACT_NAME, `${path}.repetition`, 1, 5),
    requests,
    successes,
    errors,
    elapsed_seconds: elapsed,
    throughput_rps: throughput,
    latency_p95_ms: nonNegativeFiniteNumber(
      row.latency_p95_ms,
      ARTIFACT_NAME,
      `${path}.latency_p95_ms`,
    ),
    error_rate: errorRate,
    error_rate_upper_bound: errorUpper,
  }
}

function ratio(value: unknown, path: string): number {
  const result = nonNegativeFiniteNumber(value, ARTIFACT_NAME, path)
  if (result > 1) invalid(ARTIFACT_NAME, `${path} must be between zero and one`)
  return result
}

const CAPACITY_LEVEL_FIELDS = [
  'concurrency',
  'warmup_requests',
  'warmup_errors',
  'warmup_elapsed_seconds',
  'measurement_requests',
  'successes',
  'errors',
  'elapsed_seconds',
  'throughput_rps',
  'throughput_cv',
  'latency_p50_ms',
  'latency_p95_ms',
  'latency_p99_ms',
  'latency_p95_cv',
  'error_rate',
  'error_rate_upper_bound',
  'measurement_cluster_count',
  'error_rate_cluster_range',
  'input_tokens',
  'output_tokens',
  'runtime_cost_usd',
  'repetitions',
  'throughput_scaling_efficiency',
  'warmup_passed',
  'latency_slo_passed',
  'cluster_coverage_passed',
  'error_rate_stability_passed',
  'error_slo_passed',
  'throughput_slo_passed',
  'scaling_slo_passed',
  'throughput_stability_passed',
  'latency_stability_passed',
  'qualified',
] as const

interface DecodedLevelFields {
  concurrency: number
  warmupRequests: number
  warmupErrors: number
  measurementRequests: number
  successes: number
  errors: number
  measurementClusterCount: number
  repetitions: EvaluationCapacityRepetition[]
}

interface DecodedLevelStatistics {
  elapsed: number
  throughput: number
  throughputCV: number
  latencyP50: number
  latencyP95: number
  latencyP99: number
  latencyP95CV: number
  errorRate: number
  errorUpper: number
  errorClusterRange: number
  scaling: number | null
}

function decodeLevelFields(level: Record<string, unknown>, path: string): DecodedLevelFields {
  const concurrency = boundedInteger(
    level.concurrency,
    ARTIFACT_NAME,
    `${path}.concurrency`,
    1,
    128,
  )
  const warmupRequests = boundedInteger(
    level.warmup_requests,
    ARTIFACT_NAME,
    `${path}.warmup_requests`,
    1,
  )
  const warmupErrors = boundedInteger(level.warmup_errors, ARTIFACT_NAME, `${path}.warmup_errors`)
  const measurementRequests = boundedInteger(
    level.measurement_requests,
    ARTIFACT_NAME,
    `${path}.measurement_requests`,
    1,
  )
  const successes = boundedInteger(level.successes, ARTIFACT_NAME, `${path}.successes`)
  const errors = boundedInteger(level.errors, ARTIFACT_NAME, `${path}.errors`)
  const measurementClusterCount = boundedInteger(
    level.measurement_cluster_count,
    ARTIFACT_NAME,
    `${path}.measurement_cluster_count`,
    3,
    5,
  )
  if (warmupErrors > warmupRequests || successes + errors !== measurementRequests) {
    invalid(ARTIFACT_NAME, `${path} request accounting is inconsistent`)
  }
  if (
    !Array.isArray(level.repetitions) ||
    level.repetitions.length < 3 ||
    level.repetitions.length > 5
  ) {
    invalid(ARTIFACT_NAME, `${path}.repetitions must contain three to five independent windows`)
  }
  const repetitions = level.repetitions.map((row, repetition) =>
    decodeRepetition(row, `${path}.repetitions[${repetition}]`),
  )
  if (
    repetitions.some(
      (row, repetition) => row.concurrency !== concurrency || row.repetition !== repetition + 1,
    ) ||
    repetitions.reduce((sum, row) => sum + row.requests, 0) !== measurementRequests ||
    repetitions.reduce((sum, row) => sum + row.successes, 0) !== successes ||
    repetitions.reduce((sum, row) => sum + row.errors, 0) !== errors ||
    measurementClusterCount !== repetitions.length
  ) {
    invalid(ARTIFACT_NAME, `${path}.repetitions do not exactly cover the level`)
  }
  return {
    concurrency,
    warmupRequests,
    warmupErrors,
    measurementRequests,
    successes,
    errors,
    measurementClusterCount,
    repetitions,
  }
}

function decodeLevelStatistics(
  level: Record<string, unknown>,
  path: string,
  fields: DecodedLevelFields,
): DecodedLevelStatistics {
  const elapsed = positiveFiniteNumber(
    level.elapsed_seconds,
    ARTIFACT_NAME,
    `${path}.elapsed_seconds`,
  )
  const throughput = positiveFiniteNumber(
    level.throughput_rps,
    ARTIFACT_NAME,
    `${path}.throughput_rps`,
  )
  const throughputCV = nonNegativeFiniteNumber(
    level.throughput_cv,
    ARTIFACT_NAME,
    `${path}.throughput_cv`,
  )
  const latencyP50 = nonNegativeFiniteNumber(
    level.latency_p50_ms,
    ARTIFACT_NAME,
    `${path}.latency_p50_ms`,
  )
  const latencyP95 = nonNegativeFiniteNumber(
    level.latency_p95_ms,
    ARTIFACT_NAME,
    `${path}.latency_p95_ms`,
  )
  const latencyP99 = nonNegativeFiniteNumber(
    level.latency_p99_ms,
    ARTIFACT_NAME,
    `${path}.latency_p99_ms`,
  )
  const latencyP95CV = nonNegativeFiniteNumber(
    level.latency_p95_cv,
    ARTIFACT_NAME,
    `${path}.latency_p95_cv`,
  )
  const errorRate = ratio(level.error_rate, `${path}.error_rate`)
  const errorUpper = ratio(level.error_rate_upper_bound, `${path}.error_rate_upper_bound`)
  const errorClusterRange = ratio(
    level.error_rate_cluster_range,
    `${path}.error_rate_cluster_range`,
  )
  const throughputValues = fields.repetitions.map((row) => row.throughput_rps)
  const latencyP95Values = fields.repetitions.map((row) => row.latency_p95_ms)
  const errorRates = fields.repetitions.map((row) => row.error_rate)
  if (
    latencyP50 > latencyP95 ||
    latencyP95 > latencyP99 ||
    !approximatelyEqual(
      elapsed,
      fields.repetitions.reduce((sum, row) => sum + row.elapsed_seconds, 0),
    ) ||
    !approximatelyEqual(throughput, arithmeticMean(throughputValues)) ||
    !approximatelyEqual(throughputCV, sampleCV(throughputValues)) ||
    !approximatelyEqual(latencyP95CV, sampleCV(latencyP95Values)) ||
    !approximatelyEqual(errorRate, arithmeticMean(errorRates)) ||
    !approximatelyEqual(
      errorUpper,
      Math.max(...fields.repetitions.map((row) => row.error_rate_upper_bound)),
    ) ||
    !approximatelyEqual(errorClusterRange, Math.max(...errorRates) - Math.min(...errorRates))
  ) {
    invalid(ARTIFACT_NAME, `${path} statistics do not match its independent repetitions`)
  }
  const scaling =
    level.throughput_scaling_efficiency === null
      ? null
      : nonNegativeFiniteNumber(
          level.throughput_scaling_efficiency,
          ARTIFACT_NAME,
          `${path}.throughput_scaling_efficiency`,
        )
  return {
    elapsed,
    throughput,
    throughputCV,
    latencyP50,
    latencyP95,
    latencyP99,
    latencyP95CV,
    errorRate,
    errorUpper,
    errorClusterRange,
    scaling,
  }
}

function decodeLevelFlags(
  level: Record<string, unknown>,
  path: string,
): Pick<
  EvaluationCapacityLevel,
  | 'warmup_passed'
  | 'latency_slo_passed'
  | 'cluster_coverage_passed'
  | 'error_rate_stability_passed'
  | 'error_slo_passed'
  | 'throughput_slo_passed'
  | 'scaling_slo_passed'
  | 'throughput_stability_passed'
  | 'latency_stability_passed'
  | 'qualified'
> {
  return {
    warmup_passed: booleanValue(level.warmup_passed, ARTIFACT_NAME, `${path}.warmup_passed`),
    latency_slo_passed: booleanValue(
      level.latency_slo_passed,
      ARTIFACT_NAME,
      `${path}.latency_slo_passed`,
    ),
    cluster_coverage_passed: booleanValue(
      level.cluster_coverage_passed,
      ARTIFACT_NAME,
      `${path}.cluster_coverage_passed`,
    ),
    error_rate_stability_passed: booleanValue(
      level.error_rate_stability_passed,
      ARTIFACT_NAME,
      `${path}.error_rate_stability_passed`,
    ),
    error_slo_passed: booleanValue(
      level.error_slo_passed,
      ARTIFACT_NAME,
      `${path}.error_slo_passed`,
    ),
    throughput_slo_passed: booleanValue(
      level.throughput_slo_passed,
      ARTIFACT_NAME,
      `${path}.throughput_slo_passed`,
    ),
    scaling_slo_passed: booleanValue(
      level.scaling_slo_passed,
      ARTIFACT_NAME,
      `${path}.scaling_slo_passed`,
    ),
    throughput_stability_passed: booleanValue(
      level.throughput_stability_passed,
      ARTIFACT_NAME,
      `${path}.throughput_stability_passed`,
    ),
    latency_stability_passed: booleanValue(
      level.latency_stability_passed,
      ARTIFACT_NAME,
      `${path}.latency_stability_passed`,
    ),
    qualified: booleanValue(level.qualified, ARTIFACT_NAME, `${path}.qualified`),
  }
}

function decodeLevel(value: unknown, index: number): EvaluationCapacityLevel {
  const path = `levels[${index}]`
  const level = recordWithExactKeys(value, CAPACITY_LEVEL_FIELDS, ARTIFACT_NAME, path)
  const fields = decodeLevelFields(level, path)
  const statistics = decodeLevelStatistics(level, path, fields)
  const warmupElapsed = positiveFiniteNumber(
    level.warmup_elapsed_seconds,
    ARTIFACT_NAME,
    `${path}.warmup_elapsed_seconds`,
  )
  const inputTokens = boundedInteger(level.input_tokens, ARTIFACT_NAME, `${path}.input_tokens`)
  const outputTokens = boundedInteger(level.output_tokens, ARTIFACT_NAME, `${path}.output_tokens`)
  const runtimeCost = nonNegativeFiniteNumber(
    level.runtime_cost_usd,
    ARTIFACT_NAME,
    `${path}.runtime_cost_usd`,
  )
  const flags = decodeLevelFlags(level, path)
  return {
    concurrency: fields.concurrency,
    warmup_requests: fields.warmupRequests,
    warmup_errors: fields.warmupErrors,
    warmup_elapsed_seconds: warmupElapsed,
    measurement_requests: fields.measurementRequests,
    successes: fields.successes,
    errors: fields.errors,
    elapsed_seconds: statistics.elapsed,
    throughput_rps: statistics.throughput,
    throughput_cv: statistics.throughputCV,
    latency_p50_ms: statistics.latencyP50,
    latency_p95_ms: statistics.latencyP95,
    latency_p99_ms: statistics.latencyP99,
    latency_p95_cv: statistics.latencyP95CV,
    error_rate: statistics.errorRate,
    error_rate_upper_bound: statistics.errorUpper,
    measurement_cluster_count: fields.measurementClusterCount,
    error_rate_cluster_range: statistics.errorClusterRange,
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    runtime_cost_usd: runtimeCost,
    repetitions: fields.repetitions,
    throughput_scaling_efficiency: statistics.scaling,
    ...flags,
  }
}

function nullableConcurrency(value: unknown, path: string): number | null {
  return value === null ? null : boundedInteger(value, ARTIFACT_NAME, path, 1, 128)
}

function decodeAssessment(value: unknown): EvaluationCapacitySLOAssessment {
  const assessment = recordWithExactKeys(
    value,
    [
      'qualified_concurrency',
      'saturation_concurrency',
      'slo_headroom',
      'verdict',
      'failure_reasons',
    ],
    ARTIFACT_NAME,
    'assessment',
  )
  if (assessment.verdict !== 'pass' && assessment.verdict !== 'fail') {
    invalid(ARTIFACT_NAME, 'assessment.verdict must be pass or fail')
  }
  if (
    !Array.isArray(assessment.failure_reasons) ||
    assessment.failure_reasons.some(
      (reason) =>
        typeof reason !== 'string' ||
        !FAILURE_REASONS.includes(reason as EvaluationCapacityFailureReason),
    )
  ) {
    invalid(ARTIFACT_NAME, 'assessment.failure_reasons are invalid')
  }
  const failureReasons = assessment.failure_reasons as EvaluationCapacityFailureReason[]
  const canonical = FAILURE_REASONS.filter((reason) => failureReasons.includes(reason))
  if (
    failureReasons.length !== new Set(failureReasons).size ||
    failureReasons.some((reason, index) => canonical[index] !== reason)
  ) {
    invalid(ARTIFACT_NAME, 'assessment.failure_reasons must be unique and canonical')
  }
  const headroom = boundedInteger(
    assessment.slo_headroom,
    ARTIFACT_NAME,
    'assessment.slo_headroom',
    -128,
    128,
  )
  return {
    qualified_concurrency: nullableConcurrency(
      assessment.qualified_concurrency,
      'assessment.qualified_concurrency',
    ),
    saturation_concurrency: nullableConcurrency(
      assessment.saturation_concurrency,
      'assessment.saturation_concurrency',
    ),
    slo_headroom: headroom,
    verdict: assessment.verdict,
    failure_reasons: failureReasons,
  }
}

function expectedFailureReasons(
  levels: EvaluationCapacityLevel[],
  slo: EvaluationCapacitySLO,
  qualifiedConcurrency: number | null,
): EvaluationCapacityFailureReason[] {
  if (qualifiedConcurrency !== null && qualifiedConcurrency >= slo.required_concurrency) return []
  const target = levels.find((level) => level.concurrency >= slo.required_concurrency)
  if (!target) return ['required_concurrency']
  const checks: Array<[boolean, EvaluationCapacityFailureReason]> = [
    [target.warmup_passed, 'warmup_errors'],
    [target.latency_slo_passed, 'latency_p95'],
    [target.cluster_coverage_passed, 'measurement_cluster_coverage'],
    [target.error_rate_stability_passed, 'error_rate_cluster_stability'],
    [target.error_slo_passed, 'error_rate_upper_bound'],
    [target.throughput_slo_passed, 'throughput'],
    [target.scaling_slo_passed, 'throughput_scaling'],
    [target.throughput_stability_passed, 'throughput_stability'],
    [target.latency_stability_passed, 'latency_stability'],
  ]
  const reasons = checks.filter(([passed]) => !passed).map(([, reason]) => reason)
  return reasons.length ? reasons : ['required_concurrency']
}

function validateReduction(profile: EvaluationCapacityProfile): void {
  let envelopeOpen = true
  let previous: EvaluationCapacityLevel | null = null
  for (const [index, level] of profile.levels.entries()) {
    if (
      level.concurrency !== profile.protocol.concurrency_levels[index] ||
      level.repetitions.length !== profile.protocol.repetitions_per_level ||
      level.repetitions.some(
        (repetition) =>
          repetition.requests !== profile.protocol.measurement_requests_per_repetition,
      ) ||
      level.warmup_requests !== level.concurrency * profile.protocol.warmup_request_multiplier
    ) {
      invalid(ARTIFACT_NAME, 'levels do not match the frozen load protocol')
    }
    const scaling = previous
      ? level.throughput_rps / previous.throughput_rps / (level.concurrency / previous.concurrency)
      : null
    const expected = {
      warmup: level.warmup_errors === 0,
      latency: level.latency_p95_ms <= profile.slo.max_latency_p95_ms,
      clusterCoverage:
        level.measurement_cluster_count >=
        profile.protocol.minimum_measurement_clusters_per_level,
      errorRateStable:
        level.error_rate_cluster_range <= profile.protocol.max_error_rate_cluster_range,
      errors: level.error_rate_upper_bound <= profile.slo.max_error_rate,
      throughput:
        level.concurrency < profile.slo.required_concurrency ||
        level.throughput_rps >= profile.slo.min_throughput_rps,
      scaling: scaling === null || scaling >= profile.slo.min_throughput_scaling_efficiency,
      throughputStable: level.throughput_cv <= profile.protocol.max_throughput_cv,
      latencyStable: level.latency_p95_cv <= profile.protocol.max_latency_p95_cv,
    }
    const qualified =
      envelopeOpen &&
      expected.warmup &&
      expected.latency &&
      expected.clusterCoverage &&
      expected.errorRateStable &&
      expected.errors &&
      expected.throughput &&
      expected.scaling &&
      expected.throughputStable &&
      expected.latencyStable
    if (
      (scaling === null
        ? level.throughput_scaling_efficiency !== null
        : level.throughput_scaling_efficiency === null ||
          !approximatelyEqual(level.throughput_scaling_efficiency, scaling)) ||
      level.warmup_passed !== expected.warmup ||
      level.latency_slo_passed !== expected.latency ||
      level.cluster_coverage_passed !== expected.clusterCoverage ||
      level.error_rate_stability_passed !== expected.errorRateStable ||
      level.error_slo_passed !== expected.errors ||
      level.throughput_slo_passed !== expected.throughput ||
      level.scaling_slo_passed !== expected.scaling ||
      level.throughput_stability_passed !== expected.throughputStable ||
      level.latency_stability_passed !== expected.latencyStable ||
      level.qualified !== qualified
    ) {
      invalid(ARTIFACT_NAME, 'level decisions do not match measured observations')
    }
    if (!qualified) envelopeOpen = false
    previous = level
  }
  const qualifiedConcurrency =
    [...profile.levels].reverse().find((level) => level.qualified)?.concurrency ?? null
  const saturationConcurrency =
    profile.levels.find((level) => !level.qualified)?.concurrency ?? null
  const headroom = (qualifiedConcurrency ?? 0) - profile.slo.required_concurrency
  const reasons = expectedFailureReasons(profile.levels, profile.slo, qualifiedConcurrency)
  const verdict = headroom >= 0 ? 'pass' : 'fail'
  if (
    profile.assessment.qualified_concurrency !== qualifiedConcurrency ||
    profile.assessment.saturation_concurrency !== saturationConcurrency ||
    profile.assessment.slo_headroom !== headroom ||
    profile.assessment.verdict !== verdict ||
    profile.assessment.failure_reasons.length !== reasons.length ||
    profile.assessment.failure_reasons.some((reason, index) => reason !== reasons[index])
  ) {
    invalid(ARTIFACT_NAME, 'assessment does not match the measured SLO envelope')
  }
}

export function decodeEvaluationCapacityProfile(
  value: unknown,
  expectedSLO: EvaluationCapacitySLO | undefined,
  expectedProtocol: EvaluationCapacityLoadProtocol | undefined,
): EvaluationCapacityProfile {
  const root = recordWithExactKeys(
    value,
    ['schema_version', 'kind', 'protocol', 'levels', 'slo', 'assessment'],
    ARTIFACT_NAME,
    'artifact',
  )
  if (root.schema_version !== 'evaluation.v1' || root.kind !== 'repeated-closed-loop-capacity') {
    invalid(ARTIFACT_NAME, 'artifact must use the current repeated closed-loop capacity contract')
  }
  if (!expectedProtocol || !expectedSLO) {
    invalid(ARTIFACT_NAME, 'frozen report capacity contracts are unavailable')
  }
  let protocol: EvaluationCapacityLoadProtocol
  let slo: EvaluationCapacitySLO
  try {
    protocol = decodeEvaluationCapacityLoadProtocol(
      root.protocol,
      expectedProtocol.concurrency_levels[expectedProtocol.concurrency_levels.length - 1],
      'Capacity profile load protocol',
    )
    slo = decodeEvaluationCapacitySLO(root.slo, 'Capacity profile SLO')
  } catch (error) {
    invalid(
      ARTIFACT_NAME,
      error instanceof Error ? error.message : 'capacity contracts are invalid',
    )
  }
  if (
    !equalEvaluationCapacityLoadProtocol(protocol, expectedProtocol) ||
    !equalEvaluationCapacitySLO(slo, expectedSLO)
  ) {
    invalid(ARTIFACT_NAME, 'protocol or SLO differs from the frozen report run contract')
  }
  if (!Array.isArray(root.levels) || root.levels.length < 2 || root.levels.length > 8) {
    invalid(ARTIFACT_NAME, 'levels must contain two to eight protocol observations')
  }
  const profile: EvaluationCapacityProfile = {
    schema_version: 'evaluation.v1',
    kind: 'repeated-closed-loop-capacity',
    protocol,
    levels: root.levels.map(decodeLevel),
    slo,
    assessment: decodeAssessment(root.assessment),
  }
  validateReduction(profile)
  return profile
}
