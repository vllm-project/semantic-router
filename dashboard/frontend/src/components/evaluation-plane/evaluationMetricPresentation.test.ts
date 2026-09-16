import { describe, expect, it } from 'vitest'

import { evaluationMetricLabel } from './evaluationMetricPresentation'

describe('evaluation metric presentation', () => {
  it('uses product labels for known and model-specific measurements', () => {
    expect(
      evaluationMetricLabel({ id: 'preference.online_snips_reward', track_id: 'preference' }),
    ).toBe('Candidate preference reward estimate')
    expect(
      evaluationMetricLabel({
        id: 'model_pool.arm.private-model-id.quality',
        track_id: 'model_pool',
      }),
    ).toBe('Model quality')
  })

  it('does not echo unknown service measurement identifiers', () => {
    const internalID = 'routing.private_estimator.v27'
    const label = evaluationMetricLabel({ id: internalID, track_id: 'routing' })

    expect(label).toBe('Routing measurement')
    expect(label).not.toContain(internalID)
  })
})
