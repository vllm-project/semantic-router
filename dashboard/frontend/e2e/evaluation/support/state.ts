import type { Route } from '@playwright/test'

import type {
  CreateEvaluationRunPayload,
  EvaluationCapacityLoadProtocol,
  EvaluationCapacitySLO,
  EvaluationCatalog,
  EvaluationRun,
  EvaluationRunLedgerWarning,
} from '../../../src/types/evaluationPlane'
import type {
  CreateEvaluationCampaignPayload,
  EvaluationCampaign,
} from '../../../src/types/evaluationCampaign'
import type {
  CreateEvaluationControlledPairPayload,
  EvaluationControlledPairState,
} from '../../../src/types/evaluationControlledPair'

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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasExactFields(value: Record<string, unknown>, fields: readonly string[]): boolean {
  const keys = Object.keys(value)
  return (
    keys.length === fields.length && keys.every((key) => fields.some((field) => field === key))
  )
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function capacityConcurrencyLevels(maximum: number): number[] {
  const levels = [1]
  for (let level = 2; level < maximum; level *= 2) levels.push(level)
  levels.push(maximum)
  return levels
}

export interface MockEvaluationPlaneOptions {
  catalog?: EvaluationCatalog
  mutationDelayMs?: number
  campaignGetDelayMs?: number
  controlledPairGetDelayMs?: number
  catalogDelayMs?: number
  ledgerDelayMs?: number
  runDelayMs?: number
  runPageSize?: number
  reportDelayMs?: number
  reportMetricCount?: number
  ledgerWarnings?: EvaluationRunLedgerWarning[]
  ledgerWarningCount?: number
  failFirstLoadMore?: boolean
  failFirstCancel?: boolean
  failFirstControlledPair?: boolean
  failFirstControlledPairCancel?: boolean
  failFirstControlledPairGet?: boolean
  failControlledPairGetAt?: number
  abortControlledPairCreateResponseAfterAccept?: boolean
  eventStreamCloseOnce?: boolean
  eventStreamEventCount?: number
  completeRunOnEventStream?: string
  reportFailureIDs?: string[]
  reportFailureStatus?: number
  diagnosticArtifactBodies?: {
    failureSummary?: string
    capacityProfile?: string
  }
}

export function createEvaluationMockState(
  initialRuns: EvaluationRun[],
  options: MockEvaluationPlaneOptions,
) {
  const ledgerWarnings = options.ledgerWarnings || []
  const ledgerWarningCount = options.ledgerWarningCount ?? ledgerWarnings.length

  return {
    options,
    runs: [...initialRuns],
    createAttempts: [] as CreateEvaluationRunPayload[],
    createdRequests: [] as CreateEvaluationRunPayload[],
    reportRequests: [] as string[],
    comparisonRequests: [] as Array<{ baselineRunID: string; candidateRunID: string }>,
    campaignRequests: [] as CreateEvaluationCampaignPayload[],
    controlledPairRequests: [] as CreateEvaluationControlledPairPayload[],
    campaignGetRequests: [] as string[],
    campaigns: new Map<string, EvaluationCampaign>(),
    runRequests: [] as string[],
    cancelCount: 0,
    deleteCount: 0,
    startCount: 0,
    eventStreamCount: 0,
    rejectCampaignGets: false,
    ledgerWarnings,
    ledgerWarningCount,
    firstLoadMorePending: options.failFirstLoadMore === true,
    firstCancelPending: options.failFirstCancel === true,
    firstControlledPairPending: options.failFirstControlledPair === true,
    firstControlledPairCancelPending: options.failFirstControlledPairCancel === true,
    firstControlledPairGetPending: options.failFirstControlledPairGet === true,
    abortControlledPairCreateResponsePending:
      options.abortControlledPairCreateResponseAfterAccept === true,
    controlledPairRunIDs: new Set<string>(),
    controlledPairStates: new Map<string, EvaluationControlledPairState>(),
    controlledPairAggregatePolls: new Map<string, number>(),
    controlledPairSources: new Map<
      string,
      { baselineSourceRunID: string; candidateSourceRunID: string }
    >(),
    controlledPairCancelRequests: [] as string[],
    controlledPairDeleteRequests: [] as string[],
    controlledPairGetRequests: [] as string[],
    ledgerRequestCount: 0,
    mutationDelay: () =>
      new Promise<void>((resolve) => setTimeout(resolve, options.mutationDelayMs || 0)),
  }
}

export type EvaluationMockState = ReturnType<typeof createEvaluationMockState>

export function sameOrderedMembers<T>(left: readonly T[], right: readonly T[]): boolean {
  if (left.length !== right.length) return false
  return left.every((value, index) => value === right[index])
}

export function sameMixtureIdentity(left: EvaluationRun, right: EvaluationRun): boolean {
  if (!left.mixture || !right.mixture) return left.mixture === right.mixture
  return (
    left.mixture.id === right.mixture.id && left.mixture.recipe_name === right.mixture.recipe_name
  )
}

export function validCapacityLoadProtocol(value: unknown, concurrency: number): boolean {
  if (
    !Number.isSafeInteger(concurrency) ||
    concurrency < 2 ||
    concurrency > 128 ||
    !isRecord(value) ||
    !hasExactFields(value, CAPACITY_LOAD_PROTOCOL_FIELDS)
  ) {
    return false
  }
  const expectedLevels = capacityConcurrencyLevels(concurrency)
  return (
    value.schema_version === 'evaluation.v1' &&
    value.kind === 'closed-loop' &&
    Array.isArray(value.concurrency_levels) &&
    value.concurrency_levels.length === expectedLevels.length &&
    value.concurrency_levels.every(
      (level, index) => Number.isSafeInteger(level) && level === expectedLevels[index],
    ) &&
    typeof value.warmup_request_multiplier === 'number' &&
    Number.isSafeInteger(value.warmup_request_multiplier) &&
    value.warmup_request_multiplier >= 2 &&
    value.warmup_request_multiplier <= 4 &&
    typeof value.measurement_requests_per_repetition === 'number' &&
    Number.isSafeInteger(value.measurement_requests_per_repetition) &&
    value.measurement_requests_per_repetition >= 100 &&
    value.measurement_requests_per_repetition <= 500 &&
    typeof value.repetitions_per_level === 'number' &&
    Number.isSafeInteger(value.repetitions_per_level) &&
    value.repetitions_per_level >= 3 &&
    value.repetitions_per_level <= 5 &&
    value.minimum_measurement_clusters_per_level === 3 &&
    value.confidence_level === 0.95 &&
    value.max_error_rate_cluster_range === 0.05 &&
    isFiniteNumber(value.max_throughput_cv) &&
    value.max_throughput_cv > 0 &&
    value.max_throughput_cv <= 0.2 &&
    isFiniteNumber(value.max_latency_p95_cv) &&
    value.max_latency_p95_cv > 0 &&
    value.max_latency_p95_cv <= 0.2
  )
}

export function validCapacitySLO(value: unknown, concurrency: number): boolean {
  return (
    isRecord(value) &&
    hasExactFields(value, CAPACITY_SLO_FIELDS) &&
    value.schema_version === 'evaluation.v1' &&
    typeof value.required_concurrency === 'number' &&
    Number.isSafeInteger(value.required_concurrency) &&
    value.required_concurrency >= 1 &&
    value.required_concurrency <= concurrency &&
    isFiniteNumber(value.max_latency_p95_ms) &&
    value.max_latency_p95_ms > 0 &&
    isFiniteNumber(value.max_error_rate) &&
    value.max_error_rate >= 0 &&
    value.max_error_rate < 1 &&
    isFiniteNumber(value.min_throughput_rps) &&
    value.min_throughput_rps > 0 &&
    isFiniteNumber(value.min_throughput_scaling_efficiency) &&
    value.min_throughput_scaling_efficiency > 0 &&
    value.min_throughput_scaling_efficiency <= 1
  )
}

function sameCapacitySLO(
  left: EvaluationCapacitySLO | undefined,
  right: EvaluationCapacitySLO | undefined,
): boolean {
  if (!left || !right) return left === right
  return CAPACITY_SLO_FIELDS.every((field) => left[field] === right[field])
}

function sameCapacityLoadProtocol(
  left: EvaluationCapacityLoadProtocol | undefined,
  right: EvaluationCapacityLoadProtocol | undefined,
): boolean {
  if (!left || !right) return left === right
  return CAPACITY_LOAD_PROTOCOL_FIELDS.every((field) =>
    field === 'concurrency_levels'
      ? sameOrderedMembers(left.concurrency_levels, right.concurrency_levels)
      : left[field] === right[field],
  )
}

export function exactCohortMatches(left: EvaluationRun, right: EvaluationRun): boolean {
  return (
    left.mode === right.mode &&
    left.target_id === right.target_id &&
    sameMixtureIdentity(left, right) &&
    left.change_profile === right.change_profile &&
    left.sample_limit === right.sample_limit &&
    left.concurrency === right.concurrency &&
    sameCapacitySLO(left.capacity_slo, right.capacity_slo) &&
    sameCapacityLoadProtocol(left.capacity_load_protocol, right.capacity_load_protocol) &&
    left.seed === right.seed &&
    sameOrderedMembers(left.suite_ids, right.suite_ids) &&
    sameOrderedMembers(left.track_ids, right.track_ids)
  )
}

export function controlledPairCohortMatches(left: EvaluationRun, right: EvaluationRun): boolean {
  return (
    left.target_id !== right.target_id &&
    Boolean(left.mixture && right.mixture) &&
    sameMixtureIdentity(left, right) &&
    left.mode === right.mode &&
    left.change_profile === right.change_profile &&
    left.sample_limit === right.sample_limit &&
    left.concurrency === right.concurrency &&
    sameCapacitySLO(left.capacity_slo, right.capacity_slo) &&
    sameCapacityLoadProtocol(left.capacity_load_protocol, right.capacity_load_protocol) &&
    left.seed === right.seed &&
    sameOrderedMembers(left.suite_ids, right.suite_ids) &&
    sameOrderedMembers(left.track_ids, right.track_ids)
  )
}

export function createRequestMatchesRun(
  request: CreateEvaluationRunPayload,
  run: EvaluationRun,
): boolean {
  return (
    request.name.trim() === run.name &&
    request.description.trim() === run.description &&
    request.mode === run.mode &&
    request.target_id === run.target_id &&
    request.change_profile === run.change_profile &&
    request.sample_limit === run.sample_limit &&
    request.concurrency === run.concurrency &&
    sameCapacitySLO(request.capacity_slo, run.capacity_slo) &&
    sameCapacityLoadProtocol(request.capacity_load_protocol, run.capacity_load_protocol) &&
    request.seed === run.seed &&
    (request.baseline_run_id || '') === (run.baseline_run_id || '') &&
    sameOrderedMembers(request.suite_ids, run.suite_ids) &&
    sameOrderedMembers(request.track_ids, run.track_ids)
  )
}

export async function fulfillJSON(route: Route, status: number, payload: unknown): Promise<void> {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(payload),
  })
}

export async function fulfillError(route: Route, status: number, message: string): Promise<void> {
  await fulfillJSON(route, status, { error: { message } })
}
