import type {
  EvaluationCampaignArmReliabilityStatistic,
  EvaluationCampaignEvidenceAnchor,
  EvaluationCampaignControlledPairBinding,
  EvaluationCampaignPairedLiveEvidence,
  EvaluationCampaignPairedStatistic,
} from '../types/evaluationCampaign'
import {
  EVALUATION_CAMPAIGN_PAIRED_LIVE_CONTRACT_VERSION,
  EVALUATION_SCHEMA_VERSION,
} from '../types/evaluationPlane'
import { isCanonicalEvaluationRunID } from './evaluationRunContract'
import {
  decodeEvaluationCampaignG3Promotion,
  evaluationCampaignPromotionSampleCount,
} from './evaluationCampaignPromotionContract'
import {
  hasOnlyEvaluationFields as exact,
  isEvaluationRecord as record,
  isFiniteNumber as finite,
  isNonNegativeInteger as integer,
  type EvaluationRecord as RecordValue,
} from './evaluationContractValidation'

const DIGEST = /^sha256:[0-9a-f]{64}$/
const PORTABLE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const PAIRED_BOOTSTRAP_SAMPLES = 1000
const PAIRED_CONFIDENCE_LEVEL = 0.95
const PAIRED_MINIMUM_CASES = 20
const PAIRED_TRACKS = ['routing', 'model_pool', 'joint', 'multimodal', 'capacity'] as const
const PAIRED_VERDICTS = new Set(['pass', 'fail', 'unavailable'])
const PAIRED_LIVE_EVIDENCE_FIELDS = [
  'schema_version',
  'contract_version',
  'controlled_pair_session_id',
  'controlled_pair_protocol',
  'baseline_run_id',
  'candidate_run_id',
  'candidate_subject_digest',
  'baseline_target_id',
  'candidate_target_id',
  'mixture_id',
  'recipe_name',
  'track_ids',
  'workload_snapshot_digest',
  'benchmark_revisions',
  'seed',
  'bootstrap_samples',
  'confidence_level',
  'promotion_policy',
  'promotion_statistics',
  'baseline_manifest_digest',
  'candidate_manifest_digest',
  'baseline_execution_attestation_digest',
  'candidate_execution_attestation_digest',
  'baseline_policy_snapshot_digest',
  'candidate_policy_snapshot_digest',
  'baseline_binding_snapshot_digest',
  'candidate_binding_snapshot_digest',
  'baseline_pool_snapshot_digest',
  'candidate_pool_snapshot_digest',
  'baseline_environment_snapshot_digest',
  'candidate_environment_snapshot_digest',
  'baseline_backend_topology_digest',
  'candidate_backend_topology_digest',
  'baseline_code_revision',
  'candidate_code_revision',
  'statistics',
  'model_pool_arm_reliability',
  'digest',
] as const

type PairedTrack = (typeof PAIRED_TRACKS)[number]

function text(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.trim() === value
}

function digest(value: unknown): value is string {
  return typeof value === 'string' && DIGEST.test(value)
}

function pairedTrackIDs(value: unknown): value is PairedTrack[] {
  if (!Array.isArray(value) || value.length === 0) return false
  const positions = value.map((trackID) => PAIRED_TRACKS.indexOf(trackID as PairedTrack))
  return positions.every(
    (position, index) => position >= 0 && (index === 0 || position > positions[index - 1]),
  )
}

function approximatelyEqual(left: number, right: number): boolean {
  if (left === right) return true
  return Math.abs(left - right) <= Number.EPSILON * 8 * Math.max(1, Math.abs(left), Math.abs(right))
}

function interval(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    value.every(finite) &&
    (value.length === 0 || (value.length === 2 && value[0] <= value[1]))
  )
}

function pairedStatisticVerdict(
  statistic: EvaluationCampaignPairedStatistic,
  minimumCandidateArmReliability: number,
): 'pass' | 'fail' | 'unavailable' | null {
  if (
    statistic.missing_pairs !== 0 ||
    statistic.sample_count < PAIRED_MINIMUM_CASES ||
    statistic.confidence_interval.length !== 2
  ) {
    return 'unavailable'
  }
  const [lower, upper] = statistic.confidence_interval
  if (statistic.analysis_unit === 'pool_worst_arm_reliability') {
    if (
      statistic.direction !== 'higher_is_better' ||
      statistic.candidate_confidence_interval?.length !== 2
    ) {
      return 'unavailable'
    }
    const [candidateLower, candidateUpper] = statistic.candidate_confidence_interval
    if (lower >= -statistic.margin && candidateLower >= minimumCandidateArmReliability) {
      return 'pass'
    }
    if (upper < -statistic.margin || candidateUpper < minimumCandidateArmReliability) {
      return 'fail'
    }
    return 'unavailable'
  }
  if (statistic.candidate_confidence_interval !== undefined) return null
  if (statistic.direction === 'higher_is_better') {
    if (lower >= -statistic.margin) return 'pass'
    if (upper < -statistic.margin) return 'fail'
    return 'unavailable'
  }
  if (upper <= statistic.margin) return 'pass'
  if (lower > statistic.margin) return 'fail'
  return 'unavailable'
}

function decodePairedStatistic(
  value: unknown,
  trackIDs: PairedTrack[],
  minimumCandidateArmReliability: number,
): EvaluationCampaignPairedStatistic {
  const statisticID = record(value) && typeof value.id === 'string' ? value.id : 'unknown'
  if (
    !record(value) ||
    !exact(value, [
      'id',
      'gate_id',
      'track_id',
      'analysis_unit',
      'direction',
      'margin',
      'baseline_value',
      'candidate_value',
      'delta',
      'confidence_level',
      'confidence_interval',
      'candidate_confidence_interval',
      'sample_count',
      'missing_pairs',
      'verdict',
    ]) ||
    !PORTABLE_ID.test(typeof value.id === 'string' ? value.id : '') ||
    (value.gate_id !== 'G3' && value.gate_id !== 'G8') ||
    !trackIDs.includes(value.track_id as PairedTrack) ||
    !PORTABLE_ID.test(typeof value.analysis_unit === 'string' ? value.analysis_unit : '') ||
    (value.direction !== 'higher_is_better' && value.direction !== 'lower_is_better') ||
    !finite(value.margin) ||
    (value.margin as number) < 0 ||
    value.confidence_level !== PAIRED_CONFIDENCE_LEVEL ||
    !integer(value.sample_count) ||
    !integer(value.missing_pairs) ||
    !interval(value.confidence_interval) ||
    (value.candidate_confidence_interval !== undefined &&
      !interval(value.candidate_confidence_interval)) ||
    typeof value.verdict !== 'string' ||
    !PAIRED_VERDICTS.has(value.verdict)
  ) {
    throw new Error(`Evaluation campaign paired statistic ${statisticID} is invalid.`)
  }
  const statistic = value as unknown as EvaluationCampaignPairedStatistic
  const conclusive = statistic.sample_count >= PAIRED_MINIMUM_CASES && statistic.missing_pairs === 0
  const candidateInterval = statistic.candidate_confidence_interval
  const worstArmReliability = statistic.analysis_unit === 'pool_worst_arm_reliability'
  if (
    (conclusive && statistic.confidence_interval.length !== 2) ||
    (!conclusive && statistic.confidence_interval.length !== 0) ||
    (worstArmReliability && candidateInterval?.length !== (conclusive ? 2 : 0))
  ) {
    throw new Error(`Evaluation campaign paired statistic ${statisticID} has an invalid interval.`)
  }
  const values = [statistic.baseline_value, statistic.candidate_value, statistic.delta]
  const valuesPresent = values.every(finite)
  const valuesAbsent = values.every((item) => item === undefined)
  const expectsValues = statistic.sample_count > 0 && statistic.missing_pairs === 0
  if ((expectsValues && !valuesPresent) || (!expectsValues && !valuesAbsent)) {
    throw new Error(`Evaluation campaign paired statistic ${statisticID} has incomplete values.`)
  }
  if (valuesPresent) {
    const baseline = statistic.baseline_value as number
    const candidate = statistic.candidate_value as number
    const delta = statistic.delta as number
    if (!approximatelyEqual(candidate - baseline, delta)) {
      throw new Error(`Evaluation campaign paired statistic ${statisticID} has invalid values.`)
    }
  }
  const expectedVerdict = pairedStatisticVerdict(statistic, minimumCandidateArmReliability)
  if (expectedVerdict !== null && statistic.verdict !== expectedVerdict) {
    throw new Error(`Evaluation campaign paired statistic ${statisticID} has an invalid verdict.`)
  }
  return statistic
}

function armReliabilityVerdict(
  statistic: EvaluationCampaignArmReliabilityStatistic,
  maximumCandidateFailureRate: number,
): 'pass' | 'fail' | 'unavailable' {
  if (
    statistic.candidate_sample_count < PAIRED_MINIMUM_CASES ||
    statistic.candidate_confidence_interval?.length !== 2
  ) {
    return 'unavailable'
  }
  const [candidateLower, candidateUpper] = statistic.candidate_confidence_interval
  if (statistic.cohort === 'candidate_only') {
    if (candidateUpper <= statistic.margin) return 'pass'
    if (candidateLower > statistic.margin) return 'fail'
    return 'unavailable'
  }
  if (
    statistic.cohort !== 'paired' ||
    statistic.baseline_sample_count !== statistic.candidate_sample_count ||
    statistic.confidence_interval.length !== 2
  ) {
    return 'unavailable'
  }
  const [lower, upper] = statistic.confidence_interval
  if (upper <= statistic.margin && candidateUpper <= maximumCandidateFailureRate) return 'pass'
  if (lower > statistic.margin || candidateLower > maximumCandidateFailureRate) return 'fail'
  return 'unavailable'
}

function decodeArmReliability(
  value: unknown,
  poolSampleCount: number,
  minimumCandidateArmReliability: number,
): EvaluationCampaignArmReliabilityStatistic {
  if (
    !record(value) ||
    !exact(value, [
      'arm_id',
      'cohort',
      'direction',
      'margin',
      'baseline_failure_rate',
      'candidate_failure_rate',
      'delta',
      'confidence_level',
      'confidence_interval',
      'candidate_confidence_interval',
      'baseline_sample_count',
      'candidate_sample_count',
      'verdict',
    ]) ||
    !PORTABLE_ID.test(typeof value.arm_id === 'string' ? value.arm_id : '') ||
    !['paired', 'baseline_only', 'candidate_only'].includes(
      typeof value.cohort === 'string' ? value.cohort : '',
    ) ||
    value.direction !== 'lower_is_better' ||
    value.margin !== 0.02 ||
    value.confidence_level !== PAIRED_CONFIDENCE_LEVEL ||
    !integer(value.baseline_sample_count) ||
    !integer(value.candidate_sample_count) ||
    !interval(value.confidence_interval) ||
    (value.candidate_confidence_interval !== undefined &&
      !interval(value.candidate_confidence_interval)) ||
    !PAIRED_VERDICTS.has(typeof value.verdict === 'string' ? value.verdict : '')
  ) {
    throw new Error('Evaluation campaign frozen-arm reliability statistic is invalid.')
  }
  const statistic = value as unknown as EvaluationCampaignArmReliabilityStatistic
  const candidateInterval = statistic.candidate_confidence_interval || []
  const maximumCandidateFailureRate = 1 - minimumCandidateArmReliability
  const baselinePresent = finite(statistic.baseline_failure_rate)
  const candidatePresent = finite(statistic.candidate_failure_rate)
  if (
    (baselinePresent &&
      ((statistic.baseline_failure_rate as number) < 0 ||
        (statistic.baseline_failure_rate as number) > 1)) ||
    (candidatePresent &&
      ((statistic.candidate_failure_rate as number) < 0 ||
        (statistic.candidate_failure_rate as number) > 1)) ||
    (statistic.confidence_interval.length === 2 &&
      statistic.confidence_interval[0] > statistic.confidence_interval[1])
  ) {
    throw new Error('Evaluation campaign frozen-arm reliability values are invalid.')
  }
  if (statistic.cohort === 'paired') {
    if (
      !baselinePresent ||
      !candidatePresent ||
      !finite(statistic.delta) ||
      statistic.baseline_sample_count !== poolSampleCount ||
      statistic.candidate_sample_count !== poolSampleCount ||
      !approximatelyEqual(
        (statistic.candidate_failure_rate as number) - (statistic.baseline_failure_rate as number),
        statistic.delta as number,
      ) ||
      statistic.confidence_interval.length !== 2 ||
      candidateInterval.length !== 2 ||
      statistic.verdict !== armReliabilityVerdict(statistic, maximumCandidateFailureRate)
    ) {
      throw new Error('Evaluation campaign paired frozen-arm reliability is invalid.')
    }
    return statistic
  }
  const baselineOnly = statistic.cohort === 'baseline_only'
  if (
    baselinePresent !== baselineOnly ||
    candidatePresent === baselineOnly ||
    statistic.delta !== undefined ||
    statistic.baseline_sample_count !== (baselineOnly ? poolSampleCount : 0) ||
    statistic.candidate_sample_count !== (baselineOnly ? 0 : poolSampleCount) ||
    statistic.confidence_interval.length !== 0 ||
    (baselineOnly && candidateInterval.length !== 0) ||
    (!baselineOnly &&
      (statistic.margin !== maximumCandidateFailureRate ||
        candidateInterval.length !== 2 ||
        statistic.verdict !== armReliabilityVerdict(statistic, maximumCandidateFailureRate))) ||
    (baselineOnly && statistic.verdict !== 'unavailable')
  ) {
    throw new Error('Evaluation campaign one-sided frozen-arm reliability is invalid.')
  }
  return statistic
}

function revisionMap(value: unknown): value is Record<string, string> {
  return (
    record(value) &&
    Object.keys(value).length > 0 &&
    Object.entries(value).every(
      ([suiteID, revision]) => PORTABLE_ID.test(suiteID) && text(revision),
    )
  )
}

function decodePairedLiveEvidenceShape(value: unknown): RecordValue {
  if (
    !record(value) ||
    !exact(value, PAIRED_LIVE_EVIDENCE_FIELDS) ||
    value.schema_version !== EVALUATION_SCHEMA_VERSION ||
    value.contract_version !== EVALUATION_CAMPAIGN_PAIRED_LIVE_CONTRACT_VERSION ||
    !isCanonicalEvaluationRunID(value.controlled_pair_session_id) ||
    value.controlled_pair_protocol !== 'abba-interleaved.v1' ||
    !digest(value.candidate_subject_digest) ||
    !PORTABLE_ID.test(
      typeof value.baseline_target_id === 'string' ? value.baseline_target_id : '',
    ) ||
    !PORTABLE_ID.test(
      typeof value.candidate_target_id === 'string' ? value.candidate_target_id : '',
    ) ||
    value.baseline_target_id === value.candidate_target_id ||
    !PORTABLE_ID.test(typeof value.mixture_id === 'string' ? value.mixture_id : '') ||
    !text(value.recipe_name) ||
    !pairedTrackIDs(value.track_ids) ||
    !digest(value.workload_snapshot_digest) ||
    !revisionMap(value.benchmark_revisions) ||
    !integer(value.seed) ||
    value.seed > 4_294_967_295 ||
    value.bootstrap_samples !== PAIRED_BOOTSTRAP_SAMPLES ||
    value.confidence_level !== PAIRED_CONFIDENCE_LEVEL ||
    !digest(value.baseline_policy_snapshot_digest) ||
    !digest(value.candidate_policy_snapshot_digest) ||
    !digest(value.baseline_binding_snapshot_digest) ||
    !digest(value.candidate_binding_snapshot_digest) ||
    !digest(value.baseline_pool_snapshot_digest) ||
    !digest(value.candidate_pool_snapshot_digest) ||
    !digest(value.baseline_environment_snapshot_digest) ||
    !digest(value.candidate_environment_snapshot_digest) ||
    !digest(value.baseline_backend_topology_digest) ||
    !digest(value.candidate_backend_topology_digest) ||
    !text(value.baseline_code_revision) ||
    !text(value.candidate_code_revision) ||
    !Array.isArray(value.statistics) ||
    !Array.isArray(value.model_pool_arm_reliability) ||
    !digest(value.digest)
  ) {
    throw new Error('Evaluation campaign paired-live evidence is invalid.')
  }
  return value
}

function validatePairedLiveEvidenceIdentity(
  value: RecordValue,
  binding: EvaluationCampaignControlledPairBinding,
  anchors: EvaluationCampaignEvidenceAnchor[],
): void {
  const baseline = anchors.find(
    (anchor) => anchor.slot_id === 'g3' && anchor.binding_role === 'baseline',
  )
  const candidate = anchors.find(
    (anchor) => anchor.slot_id === 'g3' && anchor.binding_role === 'candidate',
  )
  if (
    !baseline ||
    !candidate ||
    value.baseline_run_id !== binding.baseline_run_id ||
    value.candidate_run_id !== binding.candidate_run_id ||
    value.candidate_subject_digest !== candidate.candidate_subject_digest ||
    value.baseline_manifest_digest !== baseline.manifest_semantic_digest ||
    value.candidate_manifest_digest !== candidate.manifest_semantic_digest ||
    value.baseline_execution_attestation_digest !== baseline.execution_attestation_digest ||
    value.candidate_execution_attestation_digest !== candidate.execution_attestation_digest
  ) {
    throw new Error('Evaluation campaign paired-live evidence is invalid.')
  }
}

interface DecodedPairedLiveStatistics {
  promotion: ReturnType<typeof decodeEvaluationCampaignG3Promotion>
  statistics: EvaluationCampaignPairedStatistic[]
}

function decodePairedLiveEvidenceStatistics(
  value: RecordValue,
): DecodedPairedLiveStatistics {
  const promotion = decodeEvaluationCampaignG3Promotion(
    value.promotion_policy,
    value.promotion_statistics,
    value.confidence_level as number,
  )
  const statisticValues = value.statistics as unknown[]
  if (statisticValues.length === 0) {
    throw new Error('Evaluation campaign paired-live statistic vector is incomplete.')
  }
  const trackIDs = value.track_ids as PairedTrack[]
  const statistics = statisticValues.map((statistic) =>
    decodePairedStatistic(statistic, trackIDs, promotion.policy.minimum_candidate_arm_reliability),
  )
  if (new Set(statistics.map((statistic) => statistic.id)).size !== statistics.length) {
    throw new Error('Evaluation campaign paired-live statistic identities are duplicated.')
  }
  if (!trackIDs.includes('model_pool')) {
    throw new Error('Evaluation campaign frozen-arm reliability coverage is invalid.')
  }
  return { promotion, statistics }
}

function decodePairedLiveEvidenceArmMembership(
  value: RecordValue,
  promotion: ReturnType<typeof decodeEvaluationCampaignG3Promotion>,
): EvaluationCampaignArmReliabilityStatistic[] {
  const poolSampleCount = evaluationCampaignPromotionSampleCount(promotion.statistics)
  const armReliability = (value.model_pool_arm_reliability as unknown[]).map((statistic) =>
    decodeArmReliability(
      statistic,
      poolSampleCount,
      promotion.policy.minimum_candidate_arm_reliability,
    ),
  )
  if (
    armReliability.length < 2 ||
    armReliability.some(
      (statistic, index) => index > 0 && statistic.arm_id <= armReliability[index - 1].arm_id,
    )
  ) {
    throw new Error('Evaluation campaign frozen-arm reliability vector is incomplete.')
  }
  const baselineArms = armReliability
    .filter((statistic) => statistic.cohort !== 'candidate_only')
    .map((statistic) => statistic.arm_id)
  const candidateArms = armReliability
    .filter((statistic) => statistic.cohort !== 'baseline_only')
    .map((statistic) => statistic.arm_id)
  const membershipChanged =
    baselineArms.length !== candidateArms.length ||
    baselineArms.some((armID, index) => armID !== candidateArms[index])
  if (
    baselineArms.length < 2 ||
    candidateArms.length < 2 ||
    (membershipChanged &&
      value.baseline_pool_snapshot_digest === value.candidate_pool_snapshot_digest)
  ) {
    throw new Error('Evaluation campaign frozen-arm membership is invalid.')
  }
  return armReliability
}

export function decodeEvaluationCampaignPairedLiveEvidence(
  value: unknown,
  binding: EvaluationCampaignControlledPairBinding,
  anchors: EvaluationCampaignEvidenceAnchor[],
): EvaluationCampaignPairedLiveEvidence {
  const evidence = decodePairedLiveEvidenceShape(value)
  validatePairedLiveEvidenceIdentity(evidence, binding, anchors)
  const { promotion, statistics } = decodePairedLiveEvidenceStatistics(evidence)
  const armReliability = decodePairedLiveEvidenceArmMembership(evidence, promotion)
  return {
    ...(evidence as unknown as EvaluationCampaignPairedLiveEvidence),
    promotion_policy: promotion.policy,
    promotion_statistics: promotion.statistics,
    statistics,
    model_pool_arm_reliability: armReliability,
  }
}
