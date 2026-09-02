import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationComparisonStatistic } from '../../types/evaluationComparison'
import EvaluationComparisonStatistics from './EvaluationComparisonStatistics'

function productSurface(markup: string): string {
  return markup.replace(/\sdata-[\w-]+="[^"]*"/g, '')
}

const statistic: EvaluationComparisonStatistic = {
  id: 'joint.normalized_regret',
  track_id: 'joint',
  estimator_id: 'paired-bootstrap-case-clustered-delta',
  estimator_version: 'v1',
  analysis_unit: 'case_normalized_regret',
  direction: 'lower_is_better',
  non_inferiority_margin: 0.05,
  baseline_value: 0.2,
  candidate_value: 0.1,
  delta: -0.1,
  confidence_level: 0.95,
  delta_confidence_interval: [-0.15, -0.05],
  candidate_confidence_interval: [0.08, 0.12],
  sample_count: 64,
  verdict: 'pass',
}

describe('EvaluationComparisonStatistics', () => {
  it('presents registered statistics as product outcomes while retaining machine identity', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationComparisonStatistics, { statistics: [statistic] }),
    )

    expect(markup).toContain('Normalized quality gap')
    expect(markup).toContain('Routing and model pool')
    expect(markup).toContain('Normalized gap to the best model')
    expect(markup).toContain('Allowed regression')
    expect(markup).toContain('data-statistic-id="joint.normalized_regret"')
    expect(markup).toContain('data-estimator-version="v1"')

    const surface = productSurface(markup)
    expect(surface).not.toContain('joint.normalized_regret')
    expect(surface).not.toMatch(/\b(?:E[0-5]|G[0-9])\b/)
    expect(surface).not.toMatch(/\b[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*\.v\d+\b/i)
  })

  it('explains an unmeasurable pair without exposing a release-check codename', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationComparisonStatistics, { statistics: [] }),
    )

    expect(markup).toContain('not yet contain enough matched cases')
    expect(markup).not.toMatch(/\b(?:E[0-5]|G[0-9])\b/)
  })
})
