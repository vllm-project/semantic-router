import type {
  EvaluationCampaignG3PromotionPolicy,
  EvaluationCampaignG3PromotionStatistic,
} from '../types/evaluationCampaign'
import {
  hasOnlyEvaluationFields as exact,
  isEvaluationRecord as record,
  isFiniteNumber as finite,
  isNonNegativeInteger as integer,
} from './evaluationContractValidation'

const PORTABLE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const PROMOTION_POLICY_FIELDS = [
  'candidate_normalized_regret_maximum',
  'paired_normalized_regret_margin',
  'minimum_no_information_frontier_lift',
  'minimum_joint_reliability',
  'maximum_all_arm_failure_rate',
  'minimum_candidate_arm_reliability',
] as const satisfies ReadonlyArray<keyof EvaluationCampaignG3PromotionPolicy>

function unitInterval(value: unknown): value is number {
  return finite(value) && value >= 0 && value <= 1
}

function confidenceLevel(value: unknown): value is number {
  return finite(value) && value > 0 && value < 1
}

function interval(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    value.every(finite) &&
    (value.length === 0 || (value.length === 2 && value[0] <= value[1]))
  )
}

function decodePolicy(value: unknown): EvaluationCampaignG3PromotionPolicy {
  if (
    !record(value) ||
    !exact(value, PROMOTION_POLICY_FIELDS) ||
    PROMOTION_POLICY_FIELDS.some((field) => !unitInterval(value[field]))
  ) {
    throw new Error('Evaluation campaign G3 promotion policy is invalid.')
  }
  return value as unknown as EvaluationCampaignG3PromotionPolicy
}

function promotionVerdict(
  statistic: EvaluationCampaignG3PromotionStatistic,
): EvaluationCampaignG3PromotionStatistic['verdict'] {
  if (statistic.missing_cases !== 0 || statistic.confidence_interval.length !== 2) {
    return 'unavailable'
  }
  const [lower, upper] = statistic.confidence_interval
  if (statistic.direction === 'higher_is_better') {
    if (lower >= statistic.threshold.value) return 'pass'
    if (upper < statistic.threshold.value) return 'fail'
    return 'unavailable'
  }
  if (upper <= statistic.threshold.value) return 'pass'
  if (lower > statistic.threshold.value) return 'fail'
  return 'unavailable'
}

function decodeStatistic(value: unknown): EvaluationCampaignG3PromotionStatistic {
  const statisticID = record(value) && typeof value.id === 'string' ? value.id : 'unknown'
  if (
    !record(value) ||
    !exact(value, [
      'id',
      'direction',
      'estimate',
      'confidence_level',
      'confidence_interval',
      'threshold',
      'sample_count',
      'missing_cases',
      'verdict',
    ]) ||
    !PORTABLE_ID.test(statisticID) ||
    (value.direction !== 'higher_is_better' && value.direction !== 'lower_is_better') ||
    !finite(value.estimate) ||
    !confidenceLevel(value.confidence_level) ||
    !interval(value.confidence_interval) ||
    !record(value.threshold) ||
    !exact(value.threshold, ['operator', 'value', 'unit']) ||
    !finite(value.threshold.value) ||
    typeof value.threshold.unit !== 'string' ||
    value.threshold.unit.length === 0 ||
    value.threshold.unit.trim() !== value.threshold.unit ||
    !integer(value.sample_count) ||
    !integer(value.missing_cases) ||
    (value.verdict !== 'pass' && value.verdict !== 'fail' && value.verdict !== 'unavailable') ||
    (value.direction === 'higher_is_better'
      ? value.threshold.operator !== '>='
      : value.threshold.operator !== '<=')
  ) {
    throw new Error(`Evaluation campaign G3 promotion statistic ${statisticID} is invalid.`)
  }
  const statistic = value as unknown as EvaluationCampaignG3PromotionStatistic
  if (
    (statistic.sample_count === 0 && statistic.confidence_interval.length !== 0) ||
    statistic.verdict !== promotionVerdict(statistic)
  ) {
    throw new Error(`Evaluation campaign G3 promotion statistic ${statisticID} is invalid.`)
  }
  return statistic
}

export function decodeEvaluationCampaignG3Promotion(
  policy: unknown,
  statistics: unknown,
  expectedConfidenceLevel: number,
): {
  policy: EvaluationCampaignG3PromotionPolicy
  statistics: EvaluationCampaignG3PromotionStatistic[]
} {
  const decodedPolicy = decodePolicy(policy)
  if (!Array.isArray(statistics) || statistics.length === 0) {
    throw new Error('Evaluation campaign G3 promotion statistic vector is incomplete.')
  }
  const decodedStatistics = statistics.map(decodeStatistic)
  const first = decodedStatistics[0]
  if (
    new Set(decodedStatistics.map((statistic) => statistic.id)).size !== decodedStatistics.length ||
    decodedStatistics.some(
      (statistic) =>
        statistic.confidence_level !== expectedConfidenceLevel ||
        statistic.confidence_level !== first.confidence_level ||
        statistic.sample_count !== first.sample_count ||
        statistic.missing_cases !== first.missing_cases,
    )
  ) {
    throw new Error('Evaluation campaign G3 promotion statistic cohort is invalid.')
  }
  return { policy: decodedPolicy, statistics: decodedStatistics }
}

export function evaluationCampaignPromotionSampleCount(
  statistics: EvaluationCampaignG3PromotionStatistic[],
): number {
  return statistics[0]?.sample_count || 0
}
