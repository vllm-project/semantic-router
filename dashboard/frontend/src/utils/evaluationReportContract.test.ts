import { describe, expect, it } from 'vitest'

import { metricAnalysisSpecification } from '../test/evaluationMetricAnalysisFixture'
import { isEvaluationMetric } from './evaluationReportContract'

function metric() {
  return {
    id: 'routing.accuracy',
    name: 'Routing accuracy',
    track_id: 'routing',
    value: 0.8,
    unit: 'fraction',
    direction: 'higher_is_better',
    sample_count: 4,
    analysis_provenance: {
      contract_version: 'metric-analysis.v1',
      ...metricAnalysisSpecification('routing.accuracy'),
      estimator_version: 'v1',
      missingness: 'fail_closed',
      exclusion_policy: 'exclude_unavailable_evidence',
      observed_exclusions: 0,
    },
  }
}

describe('evaluation metric analysis provenance decoder', () => {
  it('requires a registered analysis plan for every metric', () => {
    expect(isEvaluationMetric(metric())).toBe(true)

    const missing = metric()
    delete (missing as { analysis_provenance?: unknown }).analysis_provenance
    expect(isEvaluationMetric(missing)).toBe(false)

    const unknown = metric()
    unknown.id = 'routing.made_up_accuracy'
    expect(isEvaluationMetric(unknown)).toBe(false)
    expect(() => metricAnalysisSpecification(unknown.id)).toThrow(/unknown evaluation metric id/)

    const malformedDynamicID = metric()
    malformedDynamicID.id = 'model_pool.arm.u-abc.quality'
    expect(isEvaluationMetric(malformedDynamicID)).toBe(false)
  })

  it('rejects forged but otherwise legal estimator fields', () => {
    const forged = metric()
    forged.analysis_provenance.weighting = 'uniform_repetition'
    expect(isEvaluationMetric(forged)).toBe(false)

    const illegal = metric()
    illegal.analysis_provenance.observed_exclusions = -1
    expect(isEvaluationMetric(illegal)).toBe(false)
  })
})
