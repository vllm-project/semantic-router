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
