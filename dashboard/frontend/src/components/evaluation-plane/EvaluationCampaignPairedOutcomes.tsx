import type { EvaluationCampaignPairedLiveEvidence } from '../../types/evaluationCampaign'
import type { EvaluationRun } from '../../types/evaluationPlane'
import EvaluationCampaignModelReliability from './EvaluationCampaignModelReliability'
import commonStyles from './EvaluationCampaign.module.css'
import layoutStyles from './EvaluationCampaignDecisionLayout.module.css'
import pairedStyles from './EvaluationCampaignPairedEvidence.module.css'
import {
  formatCampaignStatistic,
  formatCampaignThreshold,
} from './evaluationCampaignDecisionPresentation'
import { formatMetric } from './evaluationPresentation'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import { GateVerdictBadge } from './EvaluationPrimitives'

type PromotionStatistic = EvaluationCampaignPairedLiveEvidence['promotion_statistics'][number]
type PairedStatistic = EvaluationCampaignPairedLiveEvidence['statistics'][number]

const PAIRED_STATISTIC_LABELS: Record<string, string> = {
  case_mean_quality: 'Candidate quality protection',
  case_pool_oracle_quality: 'Best available model quality',
  pool_worst_arm_reliability: 'Worst-model reliability',
  case_failure_fraction: 'Failure risk',
  case_all_arm_failure: 'All-model failure risk',
  case_max_latency_relative_delta: 'Latency risk',
}

const PROMOTION_STATISTIC_LABELS: Record<string, string> = {
  'campaign.g3.candidate_normalized_regret': 'Candidate quality gap',
  'campaign.g3.paired_normalized_regret_delta': 'Quality-gap change',
  'campaign.g3.no_information_frontier_lift': 'Improvement over a no-routing baseline',
  'campaign.g3.joint_reliability': 'End-to-end reliability',
  'campaign.g3.all_arm_failure_rate': 'Pool availability',
}

function formatMeasuredRange(values: number[], unit: string): string {
  if (values.length !== 2) return 'Inconclusive'
  return `[${formatMetric({ value: values[0], unit })}, ${formatMetric({ value: values[1], unit })}]`
}

function PromotionStatisticRow({ statistic }: { statistic: PromotionStatistic }) {
  return (
    <tr data-statistic-id={statistic.id} data-verdict={statistic.verdict}>
      <th scope="row">
        <strong>{PROMOTION_STATISTIC_LABELS[statistic.id] || 'Additional release measure'}</strong>
      </th>
      <td>{formatMetric({ value: statistic.estimate, unit: statistic.threshold.unit })}</td>
      <td>{formatMeasuredRange(statistic.confidence_interval, statistic.threshold.unit)}</td>
      <td>{formatCampaignThreshold(statistic.threshold)}</td>
      <td>
        {statistic.sample_count}
        {statistic.missing_cases ? ` · ${statistic.missing_cases} missing` : ''}
      </td>
      <td>
        <GateVerdictBadge verdict={statistic.verdict} disposition="required" />
      </td>
    </tr>
  )
}

function PromotionStatisticsTable({ statistics }: { statistics: PromotionStatistic[] }) {
  return (
    <div
      className={pairedStyles.pairedTableFrame}
      role="region"
      aria-label="Release measure matrix"
      tabIndex={0}
    >
      <table className={pairedStyles.pairedTable}>
        <caption className={commonStyles.srOnly}>Release measures</caption>
        <thead>
          <tr>
            <th scope="col">Release measure</th>
            <th scope="col">Estimate</th>
            <th scope="col">Confidence interval</th>
            <th scope="col">Threshold</th>
            <th scope="col">Cases</th>
            <th scope="col">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {statistics.map((statistic) => (
            <PromotionStatisticRow key={statistic.id} statistic={statistic} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PromotionBoundary({ evidence }: { evidence: EvaluationCampaignPairedLiveEvidence }) {
  return (
    <div className={pairedStyles.promotionBoundary}>
      <div className={pairedStyles.armReliabilityHeader}>
        <div>
          <span className={commonStyles.eyebrow}>Release decision boundary</span>
          <h5>Release measures</h5>
        </div>
        <p>
          Absolute candidate quality, paired change, frontier lift, end-to-end reliability, and
          all-model availability are evaluated together under the defined release policy.
        </p>
      </div>
      <PromotionStatisticsTable statistics={evidence.promotion_statistics} />
    </div>
  )
}

function PairedStatisticConfidence({ statistic }: { statistic: PairedStatistic }) {
  return (
    <td>
      <span>
        Δ{' '}
        {statistic.confidence_interval.length === 2
          ? `[${formatCampaignStatistic(statistic.confidence_interval[0], true)}, ${formatCampaignStatistic(statistic.confidence_interval[1], true)}]`
          : 'Inconclusive'}
      </span>
      {statistic.candidate_confidence_interval?.length === 2 ? (
        <small>
          Candidate [{formatCampaignStatistic(statistic.candidate_confidence_interval[0])},{' '}
          {formatCampaignStatistic(statistic.candidate_confidence_interval[1])}]
        </small>
      ) : null}
    </td>
  )
}

function PairedStatisticRow({ statistic }: { statistic: PairedStatistic }) {
  return (
    <tr
      data-check-id={statistic.gate_id}
      data-statistic-id={statistic.id}
      data-verdict={statistic.verdict}
    >
      <th scope="row">
        <span>{TRACK_PRESENTATION[statistic.track_id].label}</span>
        <strong>
          {PAIRED_STATISTIC_LABELS[statistic.analysis_unit] || 'Additional paired measure'}
        </strong>
        <small>
          {statistic.direction === 'higher_is_better' ? '≥' : '≤'} margin{' '}
          {formatCampaignStatistic(statistic.margin)}
        </small>
      </th>
      <td>{formatCampaignStatistic(statistic.baseline_value)}</td>
      <td>{formatCampaignStatistic(statistic.candidate_value)}</td>
      <td>{formatCampaignStatistic(statistic.delta, true)}</td>
      <PairedStatisticConfidence statistic={statistic} />
      <td>
        {statistic.sample_count}
        {statistic.missing_pairs ? ` · ${statistic.missing_pairs} missing` : ''}
      </td>
      <td>
        <GateVerdictBadge verdict={statistic.verdict} disposition="required" />
      </td>
    </tr>
  )
}

function PairedStatisticsTable({ statistics }: { statistics: PairedStatistic[] }) {
  return (
    <div
      className={pairedStyles.pairedTableFrame}
      role="region"
      aria-label="Paired live statistic matrix"
      tabIndex={0}
    >
      <table className={pairedStyles.pairedTable}>
        <caption className={commonStyles.srOnly}>Paired baseline and candidate statistics</caption>
        <thead>
          <tr>
            <th scope="col">Evaluation area / measurement</th>
            <th scope="col">Baseline</th>
            <th scope="col">Candidate</th>
            <th scope="col">Delta</th>
            <th scope="col">Confidence interval</th>
            <th scope="col">Pairs</th>
            <th scope="col">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {statistics.map((statistic) => (
            <PairedStatisticRow key={statistic.id} statistic={statistic} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DiagnosticBoundary({ evidence }: { evidence: EvaluationCampaignPairedLiveEvidence }) {
  return (
    <div className={pairedStyles.diagnosticBoundary}>
      <div className={pairedStyles.armReliabilityHeader}>
        <div>
          <span className={commonStyles.eyebrow}>Paired diagnostics</span>
          <h5>Differences by evaluation area</h5>
        </div>
        <p>
          Additional matched-result diagnostics use stable product labels, including measurements
          introduced by a newer evaluation service.
        </p>
      </div>
      <PairedStatisticsTable statistics={evidence.statistics} />
    </div>
  )
}

export default function EvaluationCampaignPairedOutcomes({
  evidence,
  runs,
}: {
  evidence: EvaluationCampaignPairedLiveEvidence
  runs: EvaluationRun[]
}) {
  return (
    <section className={layoutStyles.decisionSection} aria-labelledby="campaign-paired-live-title">
      <div className={layoutStyles.sectionHeader}>
        <div>
          <span className={commonStyles.eyebrow}>Controlled comparison</span>
          <h4 id="campaign-paired-live-title">Matched baseline and candidate outcomes</h4>
          <p>
            Confidence ranges are recalculated from matched live results. Shadow-risk measures
            remain diagnostic until production exposure is available.
          </p>
        </div>
      </div>
      <PromotionBoundary evidence={evidence} />
      <DiagnosticBoundary evidence={evidence} />
      <EvaluationCampaignModelReliability evidence={evidence} runs={runs} />
    </section>
  )
}
