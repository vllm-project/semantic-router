import { describe, expect, it } from 'vitest'

import { EVALUATION_TRACK_IDS } from '../../types/evaluationPlane'
import { clampFraction, evidenceRank, formatDelta, formatMetric } from './evaluationPresentation'

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
    expect(formatMetric({ value: null, unit: 'ms' })).toBe('Unavailable')
    expect(formatDelta({ delta: -12.5, unit: 'ms' })).toBe('−12.5 ms')
    expect(clampFraction(2)).toBe(1)
    expect(evidenceRank('E5')).toBeGreaterThan(evidenceRank('E2'))
  })
})
