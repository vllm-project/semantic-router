import { describe, expect, it } from 'vitest'

import {
  decisionModelRefsForForm,
  decisionModelRefsForSave,
} from './configPageDecisionFormSupport'
import type { DecisionConfig } from './configPageSupport'

describe('decisionModelRefsForSave', () => {
  it('round-trips max_completion_tokens through load, edit, and save', () => {
    const existing: DecisionConfig['modelRefs'] = [
      {
        model: 'model-a',
        use_reasoning: false,
        lora_name: 'adapter-a',
        weight: 0.5,
        max_completion_tokens: 256,
      },
    ]

    const loaded = decisionModelRefsForForm(existing)
    expect(loaded[0].max_completion_tokens).toBe(256)

    const edited = [{ ...loaded[0], max_completion_tokens: 128 }]
    const saved = decisionModelRefsForSave(edited)
    expect(saved).toEqual([
      {
        model: 'model-a',
        use_reasoning: false,
        lora_name: 'adapter-a',
        weight: 0.5,
        max_completion_tokens: 128,
      },
    ])

    const resaved = decisionModelRefsForSave(loaded)
    expect(resaved[0].max_completion_tokens).toBe(256)
  })

  it('copies a newly entered ceiling and omits an unset field', () => {
    expect(
      decisionModelRefsForSave([
        { model: 'model-a', use_reasoning: false, max_completion_tokens: 64 },
        { model: 'model-b', use_reasoning: true },
      ]),
    ).toEqual([
      { model: 'model-a', use_reasoning: false, max_completion_tokens: 64 },
      { model: 'model-b', use_reasoning: true },
    ])
  })

  it('rejects a non-positive or non-integer ceiling', () => {
    expect(() =>
      decisionModelRefsForSave([
        { model: 'model-a', use_reasoning: false, max_completion_tokens: 0 },
      ]),
    ).toThrow(/max_completion_tokens must be a finite integer >= 1/)
    expect(() =>
      decisionModelRefsForSave([
        { model: 'model-a', use_reasoning: false, max_completion_tokens: 1.5 },
      ]),
    ).toThrow(/max_completion_tokens must be a finite integer >= 1/)
  })
})
