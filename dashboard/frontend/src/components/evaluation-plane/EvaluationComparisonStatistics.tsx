import type { EvaluationComparisonStatistic } from '../../types/evaluationComparison'
import { evaluationMetricLabel } from './evaluationMetricPresentation'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import { GateVerdictBadge } from './EvaluationPrimitives'
import tableStyles from './EvaluationReportTable.module.css'
import styles from './EvaluationComparisonStatistics.module.css'

const ANALYSIS_UNIT_LABELS: Record<EvaluationComparisonStatistic['analysis_unit'], string> = {
  case_mean: 'Average across matched cases',
  case_max: 'Best available outcome per case',
  case_oracle_regret: 'Gap to the best model per case',
  case_normalized_regret: 'Normalized gap to the best model',
}

const STATISTIC_LABEL_OVERRIDES: Record<string, string> = {
  'joint.realized_quality': 'Routed response quality',
  'joint.normalized_regret': 'Normalized quality gap',
  'agentic.task_score': 'Agent task score',
  'agentic.success_rate': 'Agent task success',
  'preference.agreement': 'Preference agreement',
  'safety.violation_case_rate': 'Safety violation rate',
}

function statisticLabel(statistic: EvaluationComparisonStatistic): string {
  return STATISTIC_LABEL_OVERRIDES[statistic.id] || evaluationMetricLabel(statistic)
}

const number = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 4,
  minimumFractionDigits: 0,
})

function formatValue(value: number): string {
  return number.format(value)
}

function formatDelta(value: number): string {
  if (value === 0) return '0'
  return `${value > 0 ? '+' : '−'}${formatValue(Math.abs(value))}`
}

function formatInterval(interval: number[]): string {
  if (interval.length !== 2) return 'Not estimable'
  return `[${formatValue(interval[0])}, ${formatValue(interval[1])}]`
}

function unavailableReason(statistic: EvaluationComparisonStatistic): string | null {
  if (statistic.verdict !== 'unavailable') return null
  if (statistic.sample_count < 20) {
    return `Needs at least 20 independent case units; observed ${statistic.sample_count}.`
  }
  if (
    statistic.delta_confidence_interval.length !== 2 ||
    statistic.candidate_confidence_interval.length !== 2
  ) {
    return 'Needs complete candidate and paired-delta 95% confidence intervals.'
  }
  return 'The confidence interval crosses a frozen decision boundary.'
}

export default function EvaluationComparisonStatistics({
  statistics,
}: {
  statistics: EvaluationComparisonStatistic[]
}) {
  if (statistics.length === 0) {
    return (
      <div className={styles.empty} role="status">
        This run pair does not yet contain enough matched cases for a controlled value comparison.
      </div>
    )
  }

  return (
    <div
      className={tableStyles.tableScroll}
      role="region"
      tabIndex={0}
      aria-label="Scroll scientific statistics"
    >
      <table className={`${tableStyles.table} ${styles.table}`}>
        <caption>Paired outcome comparison</caption>
        <thead>
          <tr>
            <th scope="col">Statistic</th>
            <th scope="col">Baseline</th>
            <th scope="col">Candidate</th>
            <th scope="col">Paired difference · 95% confidence range</th>
            <th scope="col">Candidate confidence range</th>
            <th scope="col">Allowed regression</th>
            <th scope="col">Cases</th>
            <th scope="col">Result</th>
          </tr>
        </thead>
        <tbody>
          {statistics.map((statistic) => {
            const reason = unavailableReason(statistic)
            return (
              <tr
                key={`${statistic.track_id}-${statistic.id}`}
                data-estimator-id={statistic.estimator_id}
                data-estimator-version={statistic.estimator_version}
                data-statistic-id={statistic.id}
              >
                <th scope="row">
                  <strong>{statisticLabel(statistic)}</strong>
                  <span>{TRACK_PRESENTATION[statistic.track_id].label}</span>
                  <span>{ANALYSIS_UNIT_LABELS[statistic.analysis_unit]}</span>
                  <small>
                    {statistic.direction === 'higher_is_better'
                      ? 'Higher is better'
                      : 'Lower is better'}
                  </small>
                </th>
                <td>{formatValue(statistic.baseline_value)}</td>
                <td>{formatValue(statistic.candidate_value)}</td>
                <td>
                  <strong>{formatDelta(statistic.delta)}</strong>
                  <span>{formatInterval(statistic.delta_confidence_interval)}</span>
                </td>
                <td>{formatInterval(statistic.candidate_confidence_interval)}</td>
                <td>±{formatValue(statistic.non_inferiority_margin)}</td>
                <td>{statistic.sample_count}</td>
                <td>
                  <GateVerdictBadge verdict={statistic.verdict} disposition="required" />
                  {reason ? <small className={styles.reason}>{reason}</small> : null}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
