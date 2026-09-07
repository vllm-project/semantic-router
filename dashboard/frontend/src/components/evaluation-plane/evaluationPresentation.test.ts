import { describe, expect, it } from 'vitest'

import type { EvaluationReport } from '../../types/evaluationReport'
import { EVALUATION_ATTESTATION_REVISION, EVALUATION_TRACK_IDS } from '../../types/evaluationPlane'
import {
  clampFraction,
  evaluationPromotionVerdict,
  evaluationResultScopeDescription,
  evaluationResultScopeLabel,
  evidenceRank,
  formatDelta,
  formatMetric,
  formatMetricThreshold,
  metricDeltaTone,
  selectHeadlineMetrics,
} from './evaluationPresentation'

describe('evaluation presentation', () => {
  it('keeps the complete eight-track taxonomy', () => {
    expect(EVALUATION_TRACK_IDS).toEqual([
      'routing',
      'model_pool',
      'joint',
      'agentic',
      'multimodal',
      'preference',
      'safety',
      'capacity',
    ])
  })

  it('formats evidence metrics and deltas without manufacturing unavailable values', () => {
    expect(formatMetric({ value: 0.912, unit: 'ratio' })).toBe('91.2%')
    expect(formatMetric({ value: null, unit: 'ms' })).toBe('—')
    expect(formatDelta({ delta: -12.5, unit: 'ms' })).toBe('−12.5 ms')
    expect(formatDelta({ delta: -28, unit: 'ms' })).toBe('−28 ms')
    expect(formatMetric({ value: 0, unit: 'usd/request' })).toBe('$0.00 / req')
    expect(formatMetric({ value: 12.25, unit: 'requests/s' })).toBe('12.25 req/s')
    expect(formatMetric({ value: 3, unit: 'arms' })).toBe('3 models')
    expect(formatMetric({ value: 0.02, unit: 'non-inferiority-headroom' })).toBe('0.02')
    expect(formatMetric({ value: 4.2, unit: 'private-service-unit' })).toBe('4.2')
    expect(
      formatMetricThreshold({
        operator: 'private-threshold-operator',
        value: 0.01,
        unit: 'private-service-unit',
      }),
    ).toBe('Target 0.01')
    expect(clampFraction(2)).toBe(1)
    expect(evidenceRank('E5')).toBeGreaterThan(evidenceRank('E2'))
  })

  it('presents evaluation depth as product capabilities instead of internal codes', () => {
    expect((['E0', 'E1', 'E2', 'E3', 'E4', 'E5'] as const).map(evaluationResultScopeLabel)).toEqual(
      [
        'Diagnostic',
        'Signal validation',
        'Prediction validation',
        'Routing validation',
        'Model-pool validation',
        'End-to-end validation',
      ],
    )
    expect(evaluationResultScopeDescription('E0')).toContain(
      'without making a release recommendation',
    )
    expect(evaluationResultScopeDescription('E5')).toContain('final task outcomes')
  })

  it('interprets deltas using the metric direction', () => {
    expect(metricDeltaTone({ delta: -12, direction: 'lower_is_better' })).toBe('positive')
    expect(metricDeltaTone({ delta: 0.05, direction: 'higher_is_better' })).toBe('positive')
    expect(metricDeltaTone({ delta: -0.05, direction: 'higher_is_better' })).toBe('negative')
    expect(metricDeltaTone({ delta: 12, direction: 'target' })).toBe('neutral')
  })

  it('elevates only metrics independently reduced by the current contract', () => {
    const report = {
      attestation_revision: EVALUATION_ATTESTATION_REVISION,
      run: { evidence_level: 'E0', track_ids: ['routing', 'safety', 'capacity'] },
      metrics: [
        {
          id: 'routing.accuracy',
          name: 'Routing accuracy',
          track_id: 'routing',
          value: 1,
          unit: 'fraction',
        },
        {
          id: 'safety.violation_rate',
          name: 'Safety violation rate',
          track_id: 'safety',
          value: 0,
          unit: 'violations/case',
        },
        {
          id: 'capacity.success_rate',
          name: 'Capacity success rate',
          track_id: 'capacity',
          value: 1,
          unit: 'fraction',
        },
      ],
    } as EvaluationReport

    expect(selectHeadlineMetrics(report).map((metric) => metric.id)).toEqual([
      'safety.violation_rate',
      'capacity.success_rate',
    ])
  })

  it('prioritizes trackless system headlines for non-joint runs', () => {
    const report = {
      attestation_revision: EVALUATION_ATTESTATION_REVISION,
      run: { evidence_level: 'E2', track_ids: ['capacity'] },
      metrics: [
        {
          id: 'safety.violation_rate',
          name: 'System violation rate',
          value: 0,
          unit: 'violations/case',
        },
        {
          id: 'capacity.success_rate',
          name: 'Capacity success rate',
          track_id: 'capacity',
          value: 1,
          unit: 'fraction',
        },
      ],
    } as EvaluationReport

    expect(selectHeadlineMetrics(report, 1).map((metric) => metric.id)).toEqual([
      'safety.violation_rate',
    ])
  })

  it('derives promotion from required gate evidence', () => {
    const report = {
      summary: { verdict: 'pass' },
      gates: [
        { id: 'G0', disposition: 'required', verdict: 'pass' },
        { id: 'G4', disposition: 'required', verdict: 'unavailable' },
        { id: 'G9', disposition: 'not_applicable', verdict: 'not_applicable' },
      ],
    } as EvaluationReport

    expect(evaluationPromotionVerdict(report)).toBe('unavailable')
  })
})
