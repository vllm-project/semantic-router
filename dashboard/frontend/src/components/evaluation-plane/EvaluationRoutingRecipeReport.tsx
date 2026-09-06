import type {
  EvaluationRoutingRecipeInputAvailabilityReport,
  EvaluationRoutingRecipeMetricAvailability,
  EvaluationRoutingRecipeReport as RoutingRecipeReport,
} from '../../types/evaluationRoutingRecipeReport'
import type { EvaluationRoutingRecipePlan } from '../../types/evaluationPlane'
import { EvaluationTechnicalDisclosure } from './EvaluationDisclosure'
import EvaluationIssueDetails, { type EvaluationIssueDetail } from './EvaluationIssueDetails'
import { EvaluationTag } from './EvaluationPrimitives'
import layoutStyles from './EvaluationReportLayout.module.css'
import styles from './EvaluationRoutingRecipeReport.module.css'

function percent(numerator: number, denominator: number): string {
  return denominator > 0 ? `${((numerator / denominator) * 100).toFixed(1)}%` : 'Not measured'
}

function MetricReading({
  metric,
  format = 'decimal',
}: {
  metric: EvaluationRoutingRecipeMetricAvailability
  format?: 'decimal' | 'fraction'
}) {
  if (!metric.available) {
    return <span className={styles.unavailable}>Not measured</span>
  }
  const value = metric.value || 0
  return (
    <span>
      <strong>{format === 'fraction' ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)}</strong>
      <small>{metric.sample_count} cases</small>
    </span>
  )
}

function InputTable({
  caption,
  itemLabel,
  inputs,
}: {
  caption: string
  itemLabel: string
  inputs: EvaluationRoutingRecipeInputAvailabilityReport[]
}) {
  if (inputs.length === 0) {
    return (
      <p className={styles.emptyLine}>No {caption.toLowerCase()} are reachable in this plan.</p>
    )
  }
  return (
    <div className={styles.tableScroll} tabIndex={0} role="region" aria-label={caption}>
      <table className={styles.table}>
        <caption>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Input</th>
            <th scope="col">Present</th>
            <th scope="col">Missing</th>
            <th scope="col">Error</th>
            <th scope="col">Timeout</th>
            <th scope="col">Latency p50 / p95</th>
          </tr>
        </thead>
        <tbody>
          {inputs.map((input, index) => (
            <tr key={input.id}>
              <th scope="row">
                {itemLabel} {index + 1}
              </th>
              <td>
                <strong>{percent(input.present, input.expected)}</strong>
                <small>{input.present} cases</small>
              </td>
              <td>{input.missing}</td>
              <td>{input.error}</td>
              <td>{input.timeout}</td>
              <td>
                {input.latency.available ? (
                  <span>
                    <strong>
                      {(input.latency.p50_ms || 0).toFixed(1)} /{' '}
                      {(input.latency.p95_ms || 0).toFixed(1)} ms
                    </strong>
                    <small>{input.latency.sample_count} timed</small>
                  </span>
                ) : (
                  <span className={styles.unavailable}>Not measured</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function routingTechnicalIssues(report: RoutingRecipeReport): EvaluationIssueDetail[] {
  const issues: EvaluationIssueDetail[] = []
  const addInputs = (
    inputs: EvaluationRoutingRecipeInputAvailabilityReport[],
    itemLabel: string,
  ) => {
    inputs.forEach((input, index) => {
      const label = `${itemLabel} ${index + 1}`
      issues.push({ label: `${label} identifier`, message: input.id })
      if (!input.latency.available && input.latency.reason) {
        issues.push({ label: `${label} latency status`, message: input.latency.reason })
      }
    })
  }

  addInputs(report.e1.signals, 'Signal')
  addInputs(report.e1.projections, 'Outcome estimate')
  report.e2.projection_outcomes.forEach((projection, index) => {
    const label = `Outcome estimate ${index + 1}`
    issues.push({ label: `${label} identifier`, message: projection.projection_id })
    const metrics = [
      { label: 'ranking correlation', metric: projection.spearman },
      { label: 'probability accuracy', metric: projection.brier },
      { label: 'calibration', metric: projection.ece_10 },
    ]
    metrics.forEach(({ label: metricLabel, metric }) => {
      if (metric.reason) {
        issues.push({ label: `${label} ${metricLabel} status`, message: metric.reason })
      }
    })
  })
  report.e2.top_k.forEach((topK) => {
    if (topK.feasible_oracle_recall.reason) {
      issues.push({
        label: `Top ${topK.k} recall status`,
        message: topK.feasible_oracle_recall.reason,
      })
    }
  })
  if (report.e2.oracle_regret.reason) {
    issues.push({
      label: 'Best-model quality-gap status',
      message: report.e2.oracle_regret.reason,
    })
  }
  return issues
}

function RoutingPlanSummary({ plan }: { plan: EvaluationRoutingRecipePlan }) {
  return (
    <>
      <dl className={styles.planIdentity} aria-label="Routing evaluation setup">
        <div>
          <dt>Configuration</dt>
          <dd>Recipe and pool pinned</dd>
        </div>
        <div>
          <dt>Evaluation target</dt>
          <dd>Saved with this run</dd>
        </div>
        <div>
          <dt>Routing inputs</dt>
          <dd>
            {plan.signals.length} signals · {plan.projections.length} outcome estimates
          </dd>
        </div>
        <div>
          <dt>Model pool</dt>
          <dd>
            {plan.arm_ids.length} candidates · measured at{' '}
            {plan.top_k.map((k) => `top ${k}`).join(' / ')}
          </dd>
        </div>
      </dl>
      <EvaluationTechnicalDisclosure
        className={styles.reproducibilityDetails}
        summary="Reproducibility details"
        summaryClassName={styles.reproducibilitySummary}
      >
        <dl className={styles.digestList} aria-label="Routing recipe identities">
          <div>
            <dt>Routing setup identity</dt>
            <dd>
              <code>{plan.plan_digest}</code>
            </dd>
          </div>
          <div>
            <dt>Evaluation target identity</dt>
            <dd>
              <code>{plan.target_snapshot_digest}</code>
            </dd>
          </div>
        </dl>
      </EvaluationTechnicalDisclosure>
    </>
  )
}

function RoutingDecisionInputs({ report }: { report: RoutingRecipeReport }) {
  const expected = report.e1.expected_decisions
  return (
    <>
      <div className={styles.stageHeader}>
        <div>
          <span>Decision inputs</span>
          <strong>Can the recipe make a complete, feasible choice?</strong>
        </div>
        <span>
          {report.e1.observed_decisions} / {expected} decisions
        </span>
      </div>
      <dl className={styles.rateLine}>
        <div>
          <dt>Decision coverage</dt>
          <dd>{percent(report.e1.observed_decisions, expected)}</dd>
        </div>
        <div>
          <dt>Eligibility complete</dt>
          <dd>
            {percent(report.e1.eligibility_complete, expected)}
            <small>{report.e1.eligibility_complete} cases</small>
          </dd>
        </div>
        <div>
          <dt>Selected feasible</dt>
          <dd>
            {percent(report.e1.selected_feasible, expected)}
            <small>{report.e1.selected_feasible} cases</small>
          </dd>
        </div>
      </dl>
      <InputTable caption="Signal availability" itemLabel="Signal" inputs={report.e1.signals} />
      <InputTable
        caption="Projection availability"
        itemLabel="Outcome estimate"
        inputs={report.e1.projections}
      />
    </>
  )
}

function ProjectionOutcomeTable({ report }: { report: RoutingRecipeReport }) {
  if (!report.e2.projection_outcomes.length) {
    return <p className={styles.emptyLine}>No outcome estimate is bound to later results.</p>
  }
  return (
    <div
      className={styles.tableScroll}
      tabIndex={0}
      role="region"
      aria-label="Projection outcome calibration"
    >
      <table className={styles.table}>
        <caption>Projection outcome calibration</caption>
        <thead>
          <tr>
            <th scope="col">Projection</th>
            <th scope="col">Ranking agreement</th>
            <th scope="col">Probability accuracy</th>
            <th scope="col">Calibration gap</th>
            <th scope="col">Reliability</th>
          </tr>
        </thead>
        <tbody>
          {report.e2.projection_outcomes.map((projection, index) => (
            <tr key={projection.projection_id}>
              <th scope="row">Outcome estimate {index + 1}</th>
              <td>
                <MetricReading metric={projection.spearman} />
              </td>
              <td>
                <MetricReading metric={projection.brier} />
              </td>
              <td>
                <MetricReading metric={projection.ece_10} />
              </td>
              <td>
                {projection.reliability_bins.length ? (
                  <span>
                    <strong>{projection.reliability_bins.length} bins</strong>
                    <small>
                      {projection.reliability_bins.reduce((sum, bin) => sum + bin.count, 0)} paired
                      cases
                    </small>
                  </span>
                ) : (
                  <span className={styles.unavailable}>Not measured</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RoutingOutcomeValidation({ report }: { report: RoutingRecipeReport }) {
  return (
    <>
      <div className={styles.stageHeader}>
        <div>
          <span>Outcome validation</span>
          <strong>Does the ranking preserve the feasible pool frontier?</strong>
        </div>
        <span>Observed outcomes after routing</span>
      </div>
      <ProjectionOutcomeTable report={report} />
      <div className={styles.outcomeLine}>
        <div>
          <span>Best-model coverage</span>
          <dl>
            {report.e2.top_k.map((topK) => (
              <div key={topK.k}>
                <dt>Top {topK.k}</dt>
                <dd>
                  <MetricReading metric={topK.feasible_oracle_recall} format="fraction" />
                </dd>
              </div>
            ))}
          </dl>
        </div>
        <div>
          <span>Quality gap to the best feasible model</span>
          <MetricReading metric={report.e2.oracle_regret} />
        </div>
      </div>
    </>
  )
}

export default function EvaluationRoutingRecipeReport({
  plan,
  report,
}: {
  plan: EvaluationRoutingRecipePlan
  report: RoutingRecipeReport
}) {
  return (
    <section className={layoutStyles.section} aria-labelledby="routing-recipe-report-title">
      <div className={layoutStyles.sectionHeader}>
        <div>
          <span className={layoutStyles.eyebrow}>Routing behavior</span>
          <h3 id="routing-recipe-report-title">Routing Recipe</h3>
          <p>
            Signals, model eligibility, ranking, and later outcomes are measured against the exact
            recipe and model pool used by this run.
          </p>
        </div>
        <EvaluationTag tone="info">
          {report.e1.observed_decisions} of {report.e1.expected_decisions} decisions measured
        </EvaluationTag>
      </div>
      <RoutingPlanSummary plan={plan} />
      <RoutingDecisionInputs report={report} />
      <RoutingOutcomeValidation report={report} />
      <EvaluationIssueDetails issues={routingTechnicalIssues(report)} />
    </section>
  )
}
