import { describe, expect, it } from 'vitest'

import { algorithmFields, mergeAlgorithmFields } from './configPageDecisionAlgorithmSupport'

describe('decision algorithm manager mapping', () => {
  it('keeps executor settings beside type and the rest under the type key', () => {
    const algorithm = {
      type: 'escalate',
      minimum_candidates: 2,
      budget: { deadline: '60s', max_calls: 2 },
      on_error: 'last_attempt',
      escalate: { order: 'size', accept: { confidence: { method: 'hybrid', gte: 0.72 } } },
      extension: { retained: true },
    }
    const fields = algorithmFields(algorithm)
    expect(fields).toEqual({
      minimum_candidates: 2,
      budget: algorithm.budget,
      on_error: 'last_attempt',
      order: 'size',
      accept: algorithm.escalate.accept,
    })
    expect(mergeAlgorithmFields(algorithm, 'escalate', fields)).toEqual(algorithm)
  })

  it('edits the System One selector settings as one field', () => {
    const algorithm = {
      type: 'systemone',
      systemone: { instructions: 'Choose by the candidate descriptions.', timeout_ms: 5000 },
    }
    const fields = algorithmFields(algorithm)
    expect(fields).toEqual({ systemone: algorithm.systemone })
    expect(mergeAlgorithmFields(algorithm, 'systemone', fields)).toEqual(algorithm)
  })

  it('drops executor settings from selectors', () => {
    expect(
      mergeAlgorithmFields({ type: 'fusion', on_error: 'skip' }, 'multi_factor', {
        budget: { deadline: '10s', max_calls: 1 },
        on_error: 'skip',
        weights: { quality: 0.6, cost: 0.4 },
      }),
    ).toEqual({ type: 'multi_factor', multi_factor: { weights: { quality: 0.6, cost: 0.4 } } })
  })
})

describe('System One escalate editor', () => {
  const budget = { deadline: '30s', max_calls: 2 }
  const quality = {
    type: 'uncalibrated',
    accept: { native: [{ question_type: 'choice', field: 'top_probability', gte: 0.9 }] },
  }
  const stages = [{ name: 'fast', model: 'decision-kai', kind: 'native' }]
  it('round-trips the budget beside the stages', () => {
    const algorithm = { type: 'escalate', budget, escalate: { quality, stages } }
    const fields = algorithmFields(algorithm)
    expect(fields).toEqual({ budget, quality, stages })
    expect(mergeAlgorithmFields(algorithm, 'escalate', fields)).toEqual(algorithm)
    expect(
      mergeAlgorithmFields(algorithm, 'escalate', {
        ...fields,
        budget: { deadline: '45s', max_calls: 3 },
      }),
    ).toEqual({ ...algorithm, budget: { deadline: '45s', max_calls: 3 } })
  })
  it('preserves calibrated quality when editing the budget', () => {
    const algorithm = {
      type: 'escalate',
      budget,
      escalate: {
        stages,
        quality: {
          type: 'calibrated',
          calibration: 'validated-kai',
          loss: 'bundle_error',
          max_risk: 0.05,
        },
      },
    }
    expect(
      mergeAlgorithmFields(algorithm, 'escalate', {
        ...algorithmFields(algorithm),
        budget: { deadline: '60s', max_calls: 4 },
      }),
    ).toEqual({ ...algorithm, budget: { deadline: '60s', max_calls: 4 } })
  })
  it('removes settings the new type does not take when switching types', () => {
    expect(
      mergeAlgorithmFields({ type: 'static', minimum_candidates: 2 }, 'escalate', {
        budget,
        quality,
        stages,
      }),
    ).toEqual({ type: 'escalate', budget, escalate: { quality, stages } })
    expect(
      mergeAlgorithmFields(
        { type: 'escalate', budget, on_error: 'fail', escalate: { quality, stages } },
        'static',
        {},
      ),
    ).toEqual({ type: 'static' })
  })
})
