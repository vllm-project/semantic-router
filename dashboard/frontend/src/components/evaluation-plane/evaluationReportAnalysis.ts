import type {
  EvaluationMetric,
  EvaluationMetricAnalysisProvenance,
} from '../../types/evaluationReport'

export type EvaluationAnalysisPlan = readonly [string, EvaluationMetricAnalysisProvenance]

export function uniqueAnalysisPlans(metrics: EvaluationMetric[]): EvaluationAnalysisPlan[] {
  return [
    ...new Map(
      metrics.map((metric) => {
        const plan = metric.analysis_provenance
        const key = [
          plan.contract_version,
          plan.estimator_id,
          plan.estimator_version,
          plan.analysis_unit,
          plan.cluster_unit,
          plan.weighting,
          plan.missingness,
          plan.exclusion_policy,
          plan.observed_exclusions,
        ].join('\u0000')
        return [key, plan] as const
      }),
    ).entries(),
  ]
}
