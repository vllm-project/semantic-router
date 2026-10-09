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
  const quality = {
    type: 'uncalibrated',
    acceptance: {
      rules: [{ question_type: 'choice', field: 'top_probability', predicate: { gte: 0.9 } }],
    },
  }
  const stages = [{ name: 'fast', model: 'decision-kai', kind: 'native' }]
  it.each(['cascade', 'policy'] as const)(
    'round-trips %s fields at their canonical level',
    (type) => {
      const algorithm = {
        type,
        quality,
        stages,
        ...(type === 'policy'
          ? { policy: { source: './policy.json', sha256: 'a'.repeat(64), cost_weight: 0.01 } }
          : {}),
      }
      expect(mergeAlgorithmFields(algorithm, type, algorithmFields(algorithm))).toEqual(algorithm)
    },
  )
  it('removes incompatible shared fields when switching execution types', () => {
    expect(
      mergeAlgorithmFields(
        { type: 'static', minimum_candidates: 2, on_error: 'fallback' },
        'cascade',
        { quality, stages },
      ),
    ).toEqual({ type: 'cascade', quality, stages })
    expect(mergeAlgorithmFields({ type: 'cascade', quality, stages }, 'static', {})).toEqual({
      type: 'static',
    })
  })
})
