import type {
  EvaluationCampaignEvidenceAnchor,
  EvaluationCampaignFidelityBinding,
  EvaluationCampaignFidelityEvidence,
} from '../types/evaluationCampaign'
import type { EvaluationChangeProfileId } from '../types/evaluationPlane'
import {
  EVALUATION_CAMPAIGN_FIDELITY_CONTRACT_VERSION,
  EVALUATION_SCHEMA_VERSION,
} from '../types/evaluationPlane'
import {
  hasOnlyEvaluationFields as exact,
  isEvaluationRecord as record,
  isFiniteNumber as finite,
  isNonNegativeInteger as integer,
} from './evaluationContractValidation'

const DIGEST = /^sha256:[0-9a-f]{64}$/
const PORTABLE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const CONFIDENCE_LEVEL = 0.95
const MINIMUM_CASES = 59
const MINIMUM_LOWER_BOUND = 0.95

function digest(value: unknown): value is string {
  return typeof value === 'string' && DIGEST.test(value)
}

function approximatelyEqual(left: number, right: number): boolean {
  if (left === right) return true
  return Math.abs(left - right) <= Number.EPSILON * 8 * Math.max(1, Math.abs(left), Math.abs(right))
}

function revisionMap(value: unknown, suiteIDs: string[]): value is Record<string, string> {
  return (
    record(value) &&
    Object.keys(value).length > 0 &&
    Object.entries(value).every(
      ([suiteID, revision]) =>
        suiteIDs.includes(suiteID) &&
        PORTABLE_ID.test(suiteID) &&
        typeof revision === 'string' &&
        revision.length > 0 &&
        revision.trim() === revision,
    )
  )
}

function expectedVerdict(
  sampleCount: number,
  unavailableCases: number,
  lowerBound: number,
): EvaluationCampaignFidelityEvidence['verdict'] {
  if (sampleCount < MINIMUM_CASES) return 'unavailable'
  return lowerBound >= MINIMUM_LOWER_BOUND && unavailableCases === 0 ? 'pass' : 'fail'
}

export function decodeEvaluationCampaignFidelityEvidence(
  value: unknown,
  changeProfile: EvaluationChangeProfileId,
  binding: EvaluationCampaignFidelityBinding,
  anchors: EvaluationCampaignEvidenceAnchor[],
): EvaluationCampaignFidelityEvidence {
  const expectedTrackID = changeProfile === 'agent_multimodal' ? 'multimodal' : 'joint'
  const reference = anchors.find(
    (anchor) => anchor.slot_id === 'g5' && anchor.binding_role === 'reference',
  )
  const live = anchors.find((anchor) => anchor.slot_id === 'g5' && anchor.binding_role === 'live')
  if (
    !record(value) ||
    !exact(value, [
      'schema_version',
      'contract_version',
      'reference_run_id',
      'live_run_id',
      'candidate_subject_digest',
      'reference_manifest_digest',
      'live_manifest_digest',
      'live_execution_attestation_digest',
      'track_id',
      'suite_ids',
      'workload_snapshot_digest',
      'benchmark_revisions',
      'matched_cases',
      'decision_mismatches',
      'outcome_mismatches',
      'unavailable_cases',
      'sample_count',
      'point_estimate',
      'lower_bound',
      'confidence_level',
      'verdict',
      'digest',
    ]) ||
    !reference ||
    !live ||
    value.schema_version !== EVALUATION_SCHEMA_VERSION ||
    value.contract_version !== EVALUATION_CAMPAIGN_FIDELITY_CONTRACT_VERSION ||
    value.reference_run_id !== binding.reference_run_id ||
    value.live_run_id !== binding.live_run_id ||
    !digest(value.candidate_subject_digest) ||
    value.candidate_subject_digest !== reference.candidate_subject_digest ||
    value.candidate_subject_digest !== live.candidate_subject_digest ||
    value.reference_manifest_digest !== reference.manifest_semantic_digest ||
    value.live_manifest_digest !== live.manifest_semantic_digest ||
    value.live_execution_attestation_digest !== live.execution_attestation_digest ||
    value.track_id !== expectedTrackID ||
    !Array.isArray(value.suite_ids) ||
    value.suite_ids.length === 0 ||
    value.suite_ids.some((suiteID) =>
      typeof suiteID === 'string' ? !PORTABLE_ID.test(suiteID) : true,
    ) ||
    new Set(value.suite_ids).size !== value.suite_ids.length ||
    !digest(value.workload_snapshot_digest) ||
    !revisionMap(value.benchmark_revisions, value.suite_ids as string[]) ||
    !integer(value.matched_cases) ||
    !integer(value.decision_mismatches) ||
    !integer(value.outcome_mismatches) ||
    !integer(value.unavailable_cases) ||
    !integer(value.sample_count) ||
    value.sample_count === 0 ||
    !finite(value.point_estimate) ||
    !finite(value.lower_bound) ||
    value.point_estimate < 0 ||
    value.point_estimate > 1 ||
    value.lower_bound < 0 ||
    value.lower_bound > value.point_estimate ||
    value.confidence_level !== CONFIDENCE_LEVEL ||
    (value.verdict !== 'pass' && value.verdict !== 'fail' && value.verdict !== 'unavailable') ||
    !digest(value.digest)
  ) {
    throw new Error('Evaluation campaign fidelity evidence is invalid.')
  }
  const total =
    value.matched_cases +
    value.decision_mismatches +
    value.outcome_mismatches +
    value.unavailable_cases
  if (
    total !== value.sample_count ||
    !approximatelyEqual(value.point_estimate, value.matched_cases / value.sample_count) ||
    value.verdict !==
      expectedVerdict(value.sample_count, value.unavailable_cases, value.lower_bound)
  ) {
    throw new Error('Evaluation campaign fidelity decision is invalid.')
  }
  return value as unknown as EvaluationCampaignFidelityEvidence
}
