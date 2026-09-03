import type {
  EvaluationCampaignArmReliabilityStatistic,
  EvaluationCampaignPairedLiveEvidence,
} from '../../types/evaluationCampaign'
import type { EvaluationRun } from '../../types/evaluationPlane'
import commonStyles from './EvaluationCampaign.module.css'
import pairedStyles from './EvaluationCampaignPairedEvidence.module.css'
import { formatCampaignStatistic } from './evaluationCampaignDecisionPresentation'
import { GateVerdictBadge } from './EvaluationPrimitives'

const MODEL_COHORT_LABELS = {
  paired: 'Matched model',
  baseline_only: 'Removed model',
  candidate_only: 'Added model',
} as const

function modelName(
  statistic: EvaluationCampaignArmReliabilityStatistic,
  evidence: EvaluationCampaignPairedLiveEvidence,
  runs: EvaluationRun[],
): string {
  const baseline = runs.find((run) => run.id === evidence.baseline_run_id)
  const candidate = runs.find((run) => run.id === evidence.candidate_run_id)
  const fromRun = (run: EvaluationRun | undefined) =>
    run?.mixture?.model_arms.find((model) => model.id === statistic.arm_id)?.model

  if (statistic.cohort === 'baseline_only') {
    return fromRun(baseline) || 'Model details unavailable'
  }
  return fromRun(candidate) || fromRun(baseline) || 'Model details unavailable'
}

export default function EvaluationCampaignModelReliability({
  evidence,
  runs,
}: {
  evidence: EvaluationCampaignPairedLiveEvidence
  runs: EvaluationRun[]
}) {
  if (!evidence.model_pool_arm_reliability.length) return null

  return (
    <div className={pairedStyles.armReliabilitySection}>
      <div className={pairedStyles.armReliabilityHeader}>
        <div>
          <span className={commonStyles.eyebrow}>Model-pool reliability</span>
          <h5>Per-model failure limits</h5>
        </div>
        <p>
          Every model shared by both pools is evaluated. A newly added model must also meet an
          absolute reliability limit; removed models remain diagnostic. The worst-model measure
          compares both complete pools on the same cases.
        </p>
      </div>
      <div
        className={pairedStyles.pairedTableFrame}
        role="region"
        aria-label="Model reliability comparison"
        tabIndex={0}
      >
        <table className={pairedStyles.pairedTable}>
          <caption className={commonStyles.srOnly}>
            Per-model baseline and candidate failure rates
          </caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Baseline failure</th>
              <th scope="col">Candidate failure</th>
              <th scope="col">Delta</th>
              <th scope="col">Confidence interval</th>
              <th scope="col">Baseline / candidate</th>
              <th scope="col">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {evidence.model_pool_arm_reliability.map((statistic) => (
              <tr key={statistic.arm_id} data-verdict={statistic.verdict}>
                <th scope="row">
                  <span>{MODEL_COHORT_LABELS[statistic.cohort]}</span>
                  <strong>{modelName(statistic, evidence, runs)}</strong>
                  <small>Failure-rate margin ≤ {formatCampaignStatistic(statistic.margin)}</small>
                </th>
                <td>{formatCampaignStatistic(statistic.baseline_failure_rate)}</td>
                <td>{formatCampaignStatistic(statistic.candidate_failure_rate)}</td>
                <td>{formatCampaignStatistic(statistic.delta, true)}</td>
                <td>
                  <span>
                    Δ{' '}
                    {statistic.confidence_interval.length === 2
                      ? `[${formatCampaignStatistic(statistic.confidence_interval[0], true)}, ${formatCampaignStatistic(statistic.confidence_interval[1], true)}]`
                      : statistic.cohort === 'paired'
                        ? 'Inconclusive'
                        : 'Not pairable'}
                  </span>
                  {statistic.candidate_confidence_interval?.length === 2 ? (
                    <small>
                      Candidate failure [
                      {formatCampaignStatistic(statistic.candidate_confidence_interval[0])},{' '}
                      {formatCampaignStatistic(statistic.candidate_confidence_interval[1])}]
                    </small>
                  ) : null}
                </td>
                <td>
                  {statistic.baseline_sample_count} / {statistic.candidate_sample_count}
                </td>
                <td>
                  <GateVerdictBadge
                    verdict={statistic.verdict}
                    disposition={statistic.cohort === 'baseline_only' ? 'advisory' : 'required'}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
