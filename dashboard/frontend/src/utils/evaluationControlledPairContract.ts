import type {
  CreateEvaluationControlledPairPayload,
  EvaluationControlledPairExecution,
} from '../types/evaluationControlledPair'
import {
  EVALUATION_CONTROLLED_PAIR_CONTRACT_VERSION,
  EVALUATION_CONTROLLED_PAIR_PROTOCOL,
} from '../types/evaluationControlledPair'
import { newEvaluationClientRequestID } from './evaluationIdentity'
import {
  assertCurrentEvaluationContract,
  hasOnlyEvaluationFields,
  isEvaluationRecord,
} from './evaluationContractValidation'
import {
  decodeEvaluationRun,
  isCanonicalEvaluationRunID,
  requireCanonicalEvaluationRunID,
} from './evaluationRunContract'

const CONTROLLED_PAIR_FIELDS = [
  'schema_version',
  'contract_version',
  'id',
  'protocol',
  'baseline_source_run_id',
  'candidate_source_run_id',
  'baseline_run',
  'candidate_run',
  'state',
  'capabilities',
] as const

const CONTROLLED_PAIR_STATES = new Set(['pending', 'running', 'terminal'])
const RUNNING_MEMBER_STATES = new Set(['running', 'sealing', 'completed', 'failed', 'cancelled'])
const TERMINAL_MEMBER_STATES = new Set(['completed', 'failed', 'cancelled'])

function capabilitiesFitState(
  state: unknown,
  capabilities: { can_cancel: boolean; can_delete: boolean },
): boolean {
  if (!CONTROLLED_PAIR_STATES.has(state as string)) return false
  if (capabilities.can_cancel && state !== 'running') return false
  if (capabilities.can_delete && state !== 'pending' && state !== 'terminal') return false
  return true
}

export function buildCreateEvaluationControlledPairPayload(
  baselineSourceRunID: string,
  candidateSourceRunID: string,
): CreateEvaluationControlledPairPayload {
  requireCanonicalEvaluationRunID(baselineSourceRunID)
  requireCanonicalEvaluationRunID(candidateSourceRunID)
  const payload = {
    client_request_id: newEvaluationClientRequestID(),
    baseline_source_run_id: baselineSourceRunID,
    candidate_source_run_id: candidateSourceRunID,
    baseline_run_id: newEvaluationClientRequestID(),
    candidate_run_id: newEvaluationClientRequestID(),
  }
  if (new Set(Object.values(payload)).size !== 5) {
    throw new Error('Controlled pair identities must be distinct canonical UUIDs.')
  }
  return payload
}

export function decodeEvaluationControlledPairExecution(
  payload: unknown,
  expectedPairID: string,
  request?: CreateEvaluationControlledPairPayload,
): EvaluationControlledPairExecution {
  requireCanonicalEvaluationRunID(expectedPairID)
  assertCurrentEvaluationContract(payload, 'Controlled pair response')
  const capabilities = isEvaluationRecord(payload.capabilities) ? payload.capabilities : null
  if (
    !hasOnlyEvaluationFields(payload, CONTROLLED_PAIR_FIELDS) ||
    payload.contract_version !== EVALUATION_CONTROLLED_PAIR_CONTRACT_VERSION ||
    payload.protocol !== EVALUATION_CONTROLLED_PAIR_PROTOCOL ||
    !isCanonicalEvaluationRunID(payload.id) ||
    !isCanonicalEvaluationRunID(payload.baseline_source_run_id) ||
    !isCanonicalEvaluationRunID(payload.candidate_source_run_id) ||
    !CONTROLLED_PAIR_STATES.has(payload.state as string) ||
    !capabilities ||
    !hasOnlyEvaluationFields(capabilities, ['can_cancel', 'can_delete']) ||
    typeof capabilities.can_cancel !== 'boolean' ||
    typeof capabilities.can_delete !== 'boolean' ||
    !capabilitiesFitState(
      payload.state,
      capabilities as { can_cancel: boolean; can_delete: boolean },
    )
  ) {
    throw new Error('Controlled pair response is incomplete.')
  }

  const baselineRun = decodeEvaluationRun(payload.baseline_run, request?.baseline_run_id)
  const candidateRun = decodeEvaluationRun(payload.candidate_run, request?.candidate_run_id)
  const identities = [
    payload.id,
    payload.baseline_source_run_id,
    payload.candidate_source_run_id,
    baselineRun.id,
    candidateRun.id,
  ]
  if (
    payload.id !== expectedPairID ||
    new Set(identities).size !== identities.length ||
    (request !== undefined &&
      (request.client_request_id !== expectedPairID ||
        payload.baseline_source_run_id !== request.baseline_source_run_id ||
        payload.candidate_source_run_id !== request.candidate_source_run_id)) ||
    baselineRun.mode !== 'live' ||
    candidateRun.mode !== 'live' ||
    baselineRun.controlled_pair?.pair_id !== expectedPairID ||
    baselineRun.controlled_pair.role !== 'baseline' ||
    candidateRun.controlled_pair?.pair_id !== expectedPairID ||
    candidateRun.controlled_pair.role !== 'candidate' ||
    baselineRun.baseline_run_id !== undefined ||
    candidateRun.baseline_run_id !== baselineRun.id ||
    (payload.state === 'pending' &&
      (baselineRun.status !== 'pending' || candidateRun.status !== 'pending')) ||
    (payload.state === 'running' &&
      ![baselineRun.status, candidateRun.status].every((status) =>
        RUNNING_MEMBER_STATES.has(status),
      )) ||
    (payload.state === 'terminal' &&
      ![baselineRun.status, candidateRun.status].every((status) =>
        TERMINAL_MEMBER_STATES.has(status),
      ))
  ) {
    throw new Error('Controlled pair response does not match the requested AB/BA execution.')
  }
  return {
    ...(payload as unknown as EvaluationControlledPairExecution),
    baseline_run: baselineRun,
    candidate_run: candidateRun,
  }
}
