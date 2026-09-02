import type {
  EvidenceLevel,
  EvaluationRun,
  EvaluationRunEvent,
  EvaluationRunLedger,
  EvaluationRunProgress,
} from '../types/evaluationPlane'
import { EVALUATION_SCHEMA_VERSION } from '../types/evaluationPlane'
import {
  decodeEvaluationCapacityLoadProtocol,
  decodeEvaluationCapacitySLO,
  requiresCapacitySLO,
} from './evaluationCapacitySLOContract'
import {
  assertCurrentEvaluationContract,
  EVALUATION_EVIDENCE_LEVEL_SET,
  EVALUATION_MODE_SET,
  EVALUATION_TRACK_ID_SET,
  type EvaluationRecord,
  hasOnlyEvaluationFields,
  isEvaluationRecord,
  isFiniteNumber,
  isKnownValue,
  isKnownValueArray,
  isNonEmptyText,
  isNonNegativeInteger,
  isOptionalText,
  isPortableEvaluationID,
  isTextArray,
} from './evaluationContractValidation'
import { isEvaluationMixture } from './evaluationMixtureContract'

const RUN_STATUS_SET = new Set([
  'pending',
  'running',
  'sealing',
  'completed',
  'failed',
  'cancelled',
])
const EVENT_TYPE_SET = new Set([
  'snapshot',
  'progress',
  'track',
  'gate',
  'artifact',
  'completed',
  'failed',
  'cancelled',
])
const EVALUATION_RUN_FIELDS = [
  'schema_version',
  'id',
  'client_request_id',
  'name',
  'description',
  'status',
  'mode',
  'evidence_level',
  'track_evidence_levels',
  'target_id',
  'mixture',
  'change_profile',
  'suite_ids',
  'track_ids',
  'sample_limit',
  'concurrency',
  'capacity_slo',
  'capacity_load_protocol',
  'seed',
  'baseline_run_id',
  'controlled_pair',
  'progress',
  'created_at',
  'started_at',
  'completed_at',
  'error',
] as const

function isProtocolText(value: unknown, allowEmpty: boolean): value is string {
  return (
    typeof value === 'string' &&
    (allowEmpty || value.length > 0) &&
    value.trim() === value &&
    new TextEncoder().encode(value).length <= 512
  )
}

function isDurableEventID(value: unknown): value is string {
  if (typeof value !== 'string' || value.length > 20 || !/^[1-9][0-9]*$/.test(value)) {
    return false
  }
  return BigInt(value) <= 18_446_744_073_709_551_615n
}

function isEvaluationRunProgress(value: unknown): value is EvaluationRunProgress {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, [
      'percent',
      'completed',
      'total',
      'current_track_id',
      'message',
    ]) &&
    isFiniteNumber(value.percent) &&
    value.percent >= 0 &&
    value.percent <= 100 &&
    isNonNegativeInteger(value.completed) &&
    isNonNegativeInteger(value.total) &&
    value.completed <= value.total &&
    (value.current_track_id === undefined ||
      isKnownValue(value.current_track_id, EVALUATION_TRACK_ID_SET)) &&
    (value.message === undefined || isProtocolText(value.message, true))
  )
}

function isTrackEvidenceLevels(value: unknown, trackIDs: unknown, headline: unknown): boolean {
  if (
    !isEvaluationRecord(value) ||
    !Array.isArray(trackIDs) ||
    !isKnownValue(headline, EVALUATION_EVIDENCE_LEVEL_SET)
  ) {
    return false
  }
  const keys = Object.keys(value)
  if (
    keys.length !== trackIDs.length ||
    keys.some(
      (trackID) =>
        !trackIDs.includes(trackID) || !isKnownValue(value[trackID], EVALUATION_EVIDENCE_LEVEL_SET),
    )
  ) {
    return false
  }
  const rank = new Map(
    [...EVALUATION_EVIDENCE_LEVEL_SET].map((level, index) => [level, index] as const),
  )
  const weakest = keys.reduce(
    (current, trackID) =>
      (rank.get(value[trackID] as EvidenceLevel) || 0) < (rank.get(current) || 0)
        ? (value[trackID] as EvidenceLevel)
        : current,
    'E5' as EvidenceLevel,
  )
  return weakest === headline
}

export function isCanonicalEvaluationRunID(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value)
  )
}

export function requireCanonicalEvaluationRunID(value: string): void {
  if (!isCanonicalEvaluationRunID(value)) {
    throw new Error('Evaluation run identity must be a canonical UUID.')
  }
}

function isEvaluationRunShape(payload: EvaluationRecord): boolean {
  return (
    hasOnlyEvaluationFields(payload, EVALUATION_RUN_FIELDS) &&
    isCanonicalEvaluationRunID(payload.id) &&
    isNonEmptyText(payload.name) &&
    typeof payload.description === 'string' &&
    isKnownValue(payload.status, RUN_STATUS_SET) &&
    isKnownValue(payload.mode, EVALUATION_MODE_SET) &&
    isKnownValue(payload.evidence_level, EVALUATION_EVIDENCE_LEVEL_SET) &&
    isTrackEvidenceLevels(
      payload.track_evidence_levels,
      payload.track_ids,
      payload.evidence_level,
    ) &&
    isNonEmptyText(payload.target_id) &&
    (payload.mixture === undefined || isEvaluationMixture(payload.mixture)) &&
    isPortableEvaluationID(payload.change_profile) &&
    isTextArray(payload.suite_ids, false) &&
    new Set(payload.suite_ids).size === payload.suite_ids.length &&
    isKnownValueArray(payload.track_ids, EVALUATION_TRACK_ID_SET, false) &&
    new Set(payload.track_ids as string[]).size === (payload.track_ids as string[]).length &&
    isNonNegativeInteger(payload.sample_limit) &&
    payload.sample_limit >= 1 &&
    isNonNegativeInteger(payload.concurrency) &&
    payload.concurrency >= 1 &&
    isNonNegativeInteger(payload.seed) &&
    isEvaluationRunProgress(payload.progress) &&
    isNonEmptyText(payload.created_at) &&
    isOptionalText(payload.started_at) &&
    isOptionalText(payload.completed_at) &&
    (payload.baseline_run_id === undefined ||
      isCanonicalEvaluationRunID(payload.baseline_run_id)) &&
    (payload.controlled_pair === undefined ||
      (isEvaluationRecord(payload.controlled_pair) &&
        hasOnlyEvaluationFields(payload.controlled_pair, ['pair_id', 'role']) &&
        isCanonicalEvaluationRunID(payload.controlled_pair.pair_id) &&
        (payload.controlled_pair.role === 'baseline' ||
          payload.controlled_pair.role === 'candidate'))) &&
    isOptionalText(payload.error)
  )
}

function validateEvaluationRunMembership(run: EvaluationRun): void {
  if (run.mode === 'live' && run.mixture === undefined) {
    throw new Error(
      'Evaluation live run does not bind its Mixture snapshot.',
    )
  }
  if (run.client_request_id !== run.id) {
    throw new Error('Evaluation run identity does not match the current contract.')
  }
  if (run.controlled_pair !== undefined) {
    if (run.mode !== 'live') {
      throw new Error('Controlled-pair run membership is only valid for live execution.')
    }
    if (run.controlled_pair.role === 'baseline' && run.baseline_run_id !== undefined) {
      throw new Error('Controlled-pair baseline member cannot declare a baseline run.')
    }
    if (
      run.controlled_pair.role === 'candidate' &&
      (run.baseline_run_id === undefined || run.baseline_run_id === run.id)
    ) {
      throw new Error(
        'Controlled-pair candidate member must reference a distinct canonical baseline run.',
      )
    }
  }
}

function validateEvaluationRunCapacity(run: EvaluationRun): void {
  const capacityRequired = requiresCapacitySLO(run.mode, run.track_ids)
  if (capacityRequired && run.concurrency < 2) {
    throw new Error('Live capacity run requires concurrency of at least 2.')
  }
  if (
    capacityRequired &&
    (run.capacity_slo === undefined || run.capacity_load_protocol === undefined)
  ) {
    throw new Error('Live capacity run is missing its frozen SLO or load protocol.')
  }
  if (
    !capacityRequired &&
    (run.capacity_slo !== undefined || run.capacity_load_protocol !== undefined)
  ) {
    throw new Error('Evaluation run contains a capacity contract outside live capacity mode.')
  }
  if (run.capacity_slo !== undefined) {
    const capacitySLO = decodeEvaluationCapacitySLO(
      run.capacity_slo,
      'Evaluation run Capacity SLO',
    )
    if (capacitySLO.required_concurrency > run.concurrency) {
      throw new Error('Evaluation run Capacity SLO exceeds its run concurrency.')
    }
  }
  if (run.capacity_load_protocol !== undefined) {
    decodeEvaluationCapacityLoadProtocol(
      run.capacity_load_protocol,
      run.concurrency,
      'Evaluation run Capacity load protocol',
    )
  }
}

export function decodeEvaluationRun(payload: unknown, expectedID?: string): EvaluationRun {
  assertCurrentEvaluationContract(payload, 'Evaluation run response')
  if (!isEvaluationRunShape(payload)) {
    throw new Error('Evaluation run response is incomplete.')
  }
  const run = payload as unknown as EvaluationRun
  validateEvaluationRunMembership(run)
  validateEvaluationRunCapacity(run)
  if (expectedID && run.id !== expectedID) {
    throw new Error('Evaluation run response did not match the requested run.')
  }
  return run
}

export function decodeEvaluationRunLedger(
  payload: unknown,
  requestedCursor?: string,
): EvaluationRunLedger {
  if (
    !isEvaluationRecord(payload) ||
    !hasOnlyEvaluationFields(payload, [
      'schema_version',
      'runs',
      'next_cursor',
      'total_runs',
      'ledger_complete',
      'warning_count',
      'warnings',
    ]) ||
    payload.schema_version !== EVALUATION_SCHEMA_VERSION ||
    !Array.isArray(payload.runs) ||
    payload.runs.some((item) => {
      try {
        decodeEvaluationRun(item)
        return false
      } catch {
        return true
      }
    }) ||
    (payload.next_cursor !== undefined && !isNonEmptyText(payload.next_cursor)) ||
    (typeof payload.next_cursor === 'string' && payload.next_cursor === requestedCursor) ||
    !isNonNegativeInteger(payload.total_runs) ||
    payload.total_runs < payload.runs.length ||
    typeof payload.ledger_complete !== 'boolean' ||
    !isNonNegativeInteger(payload.warning_count) ||
    !Array.isArray(payload.warnings) ||
    payload.warnings.some(
      (item) =>
        !isEvaluationRecord(item) ||
        !hasOnlyEvaluationFields(item, ['code', 'evidence_id', 'evidence_file', 'message']) ||
        typeof item.code !== 'string' ||
        !isNonEmptyText(item.evidence_id) ||
        typeof item.evidence_file !== 'string' ||
        typeof item.message !== 'string',
    ) ||
    payload.warnings.length > payload.warning_count ||
    payload.ledger_complete !== (payload.warning_count === 0)
  ) {
    throw new Error('Evaluation run ledger response is invalid or incomplete.')
  }
  return payload as unknown as EvaluationRunLedger
}

export function decodeEvaluationRunEvent(payload: unknown, run: EvaluationRun): EvaluationRunEvent {
  const invalidBase =
    !isEvaluationRecord(payload) ||
    !hasOnlyEvaluationFields(payload, [
      'id',
      'run_id',
      'type',
      'timestamp',
      'message',
      'track_id',
      'progress',
      'payload',
    ]) ||
    !isCanonicalEvaluationRunID(run.id) ||
    payload.run_id !== run.id ||
    !isDurableEventID(payload.id) ||
    !isKnownValue(payload.type, EVENT_TYPE_SET) ||
    !isNonEmptyText(payload.timestamp) ||
    !isProtocolText(payload.message, false) ||
    (payload.track_id !== undefined && !isKnownValue(payload.track_id, EVALUATION_TRACK_ID_SET)) ||
    (payload.progress !== undefined &&
      (!isEvaluationRunProgress(payload.progress) ||
        payload.progress.total !== run.track_ids.length))

  if (invalidBase) {
    throw new Error('Evaluation event stream returned an invalid event.')
  }

  const event = payload as EvaluationRecord
  const validTrackPayload =
    event.type === 'track' &&
    isKnownValue(event.track_id, EVALUATION_TRACK_ID_SET) &&
    isEvaluationRunProgress(event.progress) &&
    event.progress.current_track_id === event.track_id &&
    isEvaluationRecord(event.payload) &&
    hasOnlyEvaluationFields(event.payload, ['record_count']) &&
    isNonNegativeInteger(event.payload.record_count) &&
    event.payload.record_count <= 100_000_000
  const validPayloadlessEvent =
    event.type !== 'track' && event.track_id === undefined && event.payload === undefined
  const terminalEvent =
    event.type === 'completed' || event.type === 'failed' || event.type === 'cancelled'
  const validTerminalProgress =
    !terminalEvent ||
    (isEvaluationRunProgress(event.progress) &&
      event.progress.total === run.track_ids.length &&
      (event.type !== 'completed' ||
        (event.progress.percent === 100 &&
          event.progress.completed === event.progress.total &&
          event.progress.current_track_id === undefined)))

  if ((!validTrackPayload && !validPayloadlessEvent) || !validTerminalProgress) {
    throw new Error('Evaluation event stream returned an invalid event.')
  }
  return event as unknown as EvaluationRunEvent
}
