import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type {
  EvaluationMetric,
  EvaluationMetricAnalysisProvenance,
} from '../../types/evaluationReport'
import { metricAnalysisSpecification } from '../../test/evaluationMetricAnalysisFixture'
import EvaluationMetricTable from './EvaluationMetricTable'

function analysisProvenance(metricID: string): EvaluationMetricAnalysisProvenance {
  return {
    contract_version: 'metric-analysis.v1',
    ...metricAnalysisSpecification(metricID),
    estimator_version: 'v1',
    missingness: 'fail_closed',
    exclusion_policy: 'exclude_unavailable_evidence',
    observed_exclusions: 0,
  }
}

const metrics = [
  {
    id: 'safety.violation_rate',
    name: 'Safety violation rate',
    track_id: 'safety',
    value: 0,
    unit: 'violations/case',
    analysis_provenance: analysisProvenance('safety.violation_rate'),
  },
  {
    id: 'routing.accuracy',
    name: 'Routing accuracy',
    track_id: 'routing',
    value: 0.8,
    unit: 'fraction',
    analysis_provenance: analysisProvenance('routing.accuracy'),
  },
] satisfies EvaluationMetric[]

describe('EvaluationMetricTable result labels', () => {
  it('distinguishes verified metrics from supporting diagnostics', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationMetricTable, {
        metrics,
        controls: false,
        evidenceLevel: 'E0',
      }),
    )

    expect(markup).toContain('Verified result · Diagnostic')
    expect(markup).toContain('Supporting diagnostic · Diagnostic')
    expect(markup).not.toContain('E0')
  })

  it('retains the same evidence boundary at higher qualification levels', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationMetricTable, {
        metrics,
        controls: false,
        evidenceLevel: 'E5',
      }),
    )

    expect(markup).toContain('Verified result · End-to-end validation')
    expect(markup).toContain('Supporting diagnostic · End-to-end validation')
    expect(markup).not.toContain('E5')
  })

  it('keeps reported names and units behind closed technical details', () => {
    const reportedName = 'arm-fast private SNIPS estimator output'
    const reportedUnit = 'private-service-unit'
    const modelMetric: EvaluationMetric = {
      id: 'model_pool.arm.fast.quality',
      name: reportedName,
      track_id: 'model_pool',
      value: 0.91,
      unit: reportedUnit,
      analysis_provenance: analysisProvenance('model_pool.arm.fast.quality'),
    }
    const markup = renderToStaticMarkup(
      createElement(EvaluationMetricTable, {
        metrics: [modelMetric],
        controls: false,
        evidenceLevel: 'E4',
      }),
    )
    const technicalDetailsStart = markup.indexOf('<details')
    const technicalDetailsEnd =
      markup.indexOf('</details>', technicalDetailsStart) + '</details>'.length
    const defaultSurface = `${markup.slice(0, technicalDetailsStart)}${markup.slice(technicalDetailsEnd)}`

    expect(defaultSurface).toContain('Model quality')
    expect(defaultSurface).toContain('0.91')
    expect(defaultSurface).not.toContain(reportedName)
    expect(defaultSurface).not.toContain(reportedUnit)
    expect(markup.slice(technicalDetailsStart)).toContain(reportedName)
    expect(markup.slice(technicalDetailsStart)).toContain(reportedUnit)
    expect(markup).not.toContain('<details open')
  })
})
