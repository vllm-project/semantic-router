import { describe, expect, it } from 'vitest'

import { algorithmFields, mergeAlgorithmFields } from './configPageDecisionAlgorithmSupport'

describe('decision algorithm manager mapping', () => {
  it('round-trips nested selector configuration without flattening canonical YAML', () => {
    const algorithm = {
      type: 'confidence',
      minimum_candidates: 2,
      confidence: { confidence_method: 'hybrid', threshold: 0.72, on_error: 'skip' },
      extension: { retained: true },
    }
    expect(mergeAlgorithmFields(algorithm, 'confidence', algorithmFields(algorithm))).toEqual(
      algorithm,
    )
  })

  it('keeps prompt fallback at the algorithm level', () => {
    const algorithm = {
      type: 'prompt',
      on_error: 'fallback',
      prompt: { model: 'router', instructions: 'Choose.' },
    }
    expect(mergeAlgorithmFields(algorithm, 'prompt', algorithmFields(algorithm))).toEqual(algorithm)
  })
})

describe('native algorithm editor', () => {
  const budget = { deadline: '30s', max_calls: 2 }
  const quality = {
    type: 'uncalibrated',
    acceptance: {
      rules: [{ question_type: 'choice', field: 'top_probability', predicate: { gte: 0.9 } }],
    },
  }
  const stages = [{ name: 'fast', model: 'decision-kai', kind: 'native' }]
  it('round-trips cascade budget, quality, and stages at the algorithm level', () => {
    const algorithm = { type: 'cascade', budget, quality, stages }
    const fields = algorithmFields(algorithm)
    expect(fields).toEqual({ budget, quality, stages })
    expect(mergeAlgorithmFields(algorithm, 'cascade', fields)).toEqual(algorithm)
    expect(
      mergeAlgorithmFields(algorithm, 'cascade', {
        ...fields,
        budget: { deadline: '45s', max_calls: 3 },
      }),
    ).toEqual({ ...algorithm, budget: { deadline: '45s', max_calls: 3 } })
  })
  it('preserves calibrated acceptance when editing the execution budget', () => {
    const algorithm = {
      type: 'cascade',
      budget,
      stages,
      quality: {
        type: 'calibrated',
        calibration: 'validated-kai',
        loss: 'bundle_error',
        max_risk: 0.05,
      },
    }
    expect(
      mergeAlgorithmFields(algorithm, 'cascade', {
        ...algorithmFields(algorithm),
        budget: { deadline: '60s', max_calls: 4 },
      }),
    ).toEqual({ ...algorithm, budget: { deadline: '60s', max_calls: 4 } })
  })
  it('removes incompatible shared fields when switching execution types', () => {
    expect(
      mergeAlgorithmFields(
        { type: 'static', minimum_candidates: 2, on_error: 'fallback' },
        'cascade',
        { budget, quality, stages },
      ),
    ).toEqual({ type: 'cascade', budget, quality, stages })
    expect(
      mergeAlgorithmFields({ type: 'cascade', budget, quality, stages }, 'static', {}),
    ).toEqual({ type: 'static' })
  })
})
