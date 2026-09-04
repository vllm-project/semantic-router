import type { EvaluationAnalysisPlan } from './evaluationReportAnalysis'
import { EvaluationTechnicalDisclosure } from './EvaluationDisclosure'
import styles from './EvaluationReportDiagnostics.module.css'

interface EvaluationReportAnalysisDetailsProps {
  metricCount: number
  plans: EvaluationAnalysisPlan[]
}

export default function EvaluationReportAnalysisDetails({
  metricCount,
  plans,
}: EvaluationReportAnalysisDetailsProps) {
  if (!plans.length) return null
  return (
    <EvaluationTechnicalDisclosure
      className={styles.analysisProvenance}
      summaryClassName={styles.analysisProvenanceSummary}
      summary={`Metric calculation details · ${metricCount} published metrics${
        plans.length === 1 ? '' : ` · ${plans.length} methods`
      }`}
    >
      {plans.map(([key, plan]) => (
        <dl key={key}>
          <div>
            <dt>Estimator</dt>
            <dd>
              {plan.estimator_id} · {plan.estimator_version}
            </dd>
          </div>
          <div>
            <dt>Analysis unit / grouping</dt>
            <dd>
              {plan.analysis_unit} / {plan.cluster_unit}
            </dd>
          </div>
          <div>
            <dt>Weighting</dt>
            <dd>{plan.weighting}</dd>
          </div>
          <div>
            <dt>Missing data / exclusions</dt>
            <dd>
              {plan.missingness} · {plan.exclusion_policy} · {plan.observed_exclusions} observed
            </dd>
          </div>
        </dl>
      ))}
    </EvaluationTechnicalDisclosure>
  )
}
