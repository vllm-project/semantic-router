import type { EvaluationMetricAnalysisProvenance } from '../types/evaluationReport'
import { resolveMetricAnalysisCatalog } from '../contracts/metricAnalysisCatalog'

type MetricAnalysisSpecification = Pick<
  EvaluationMetricAnalysisProvenance,
  | 'estimator_id'
  | 'estimator_version'
  | 'analysis_unit'
  | 'cluster_unit'
  | 'weighting'
  | 'missingness'
  | 'exclusion_policy'
>

/** Test-only helper for constructing contract-valid metric fixtures. */
export function metricAnalysisSpecification(metricID: string): MetricAnalysisSpecification {
  const specification = resolveMetricAnalysisCatalog(metricID).specification
  return {
    estimator_id: specification.estimator_id,
    estimator_version: specification.estimator_version,
    analysis_unit: specification.analysis_unit,
    cluster_unit: specification.cluster_unit,
    weighting: specification.weighting as MetricAnalysisSpecification['weighting'],
    missingness: specification.missingness,
    exclusion_policy: specification.exclusion_policy,
  }
}
