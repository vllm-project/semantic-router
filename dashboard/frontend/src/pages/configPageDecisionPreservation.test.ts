import { describe, expect, it } from 'vitest'

import { decisionRulesForSave } from './configPageDecisionFormSupport'
import { mergeDecisionForSave, type DecisionConfig } from './configPageSupport'

describe('decision editor preservation', () => {
  it('preserves prompt algorithms and non-form fields during edits', () => {
    const existing: DecisionConfig = {
      name: 'prompt-route',
      description: 'before',
      priority: 10,
      tier: 1,
      annotations: { owner: 'routing' },
      rules: {
        operator: 'AND',
        conditions: [{ type: 'keyword', name: 'complex' }],
      },
      modelRefs: [
        { model: 'model-a', use_reasoning: false },
        { model: 'model-b', use_reasoning: true },
      ],
      plugins: [],
      algorithm: {
        type: 'prompt',
        on_error: 'fallback',
        prompt: {
          model: 'router-small',
          instructions: 'Choose.',
          timeout_seconds: 5,
        },
      },
    }
    const updated = mergeDecisionForSave(existing, {
      name: existing.name,
      description: 'after',
      priority: 20,
      rules: existing.rules,
      modelRefs: existing.modelRefs,
      plugins: [],
    })

    expect(updated.description).toBe('after')
    expect(updated.algorithm).toEqual(existing.algorithm)
    expect(updated.annotations).toEqual(existing.annotations)
    expect(updated.tier).toBe(1)
  })

  it('preserves recursive rule trees during form edits', () => {
    const existing: DecisionConfig['rules'] = {
      operator: 'AND',
      conditions: [
        { type: 'keyword', name: 'complex' },
        {
          operator: 'OR',
          conditions: [
            { type: 'metadata', name: 'canary' },
            {
              type: 'systemone',
              name: 'risk',
              label: 'RISKY',
              predicate: { gte: 0.8 },
              on_error: 'match',
            },
          ],
        },
      ],
    }

    const result = decisionRulesForSave(existing)

    expect(result.operator).toBe(existing.operator)
    expect(result.conditions).toEqual(existing.conditions)
    expect(result).not.toBe(existing)
  })

  it('rejects a root unknown policy combined with a System One on_error policy', () => {
    expect(() =>
      decisionRulesForSave({
        operator: 'AND',
        on_unknown: 'fail_request',
        conditions: [{ type: 'systemone', name: 'risk', on_error: 'match' }],
      }),
    ).toThrow(/cannot be combined/)
  })

  it('rejects on_error on a composite root', () => {
    expect(() =>
      decisionRulesForSave({
        operator: 'AND',
        on_error: 'match',
        conditions: [{ type: 'keyword', name: 'urgent' }],
      }),
    ).toThrow(/only applies to leaf conditions/)
  })

  it('keeps a single root condition instead of saving an unconditional match', () => {
    expect(decisionRulesForSave({ type: 'jailbreak', name: 'prompt_injection' })).toEqual({
      type: 'jailbreak',
      name: 'prompt_injection',
    })
    const labelled: DecisionConfig['rules'] = {
      type: 'systemone',
      name: 'risk',
      label: 'RISKY',
      predicate: { gte: 0.8 },
      on_unknown: 'no_match',
    }
    expect(decisionRulesForSave(labelled)).toEqual(labelled)
    expect(decisionRulesForSave({})).toEqual({})
  })

  it('validates a single root condition', () => {
    expect(() => decisionRulesForSave({ type: 'keyword' })).toThrow(
      'The root condition needs both type and name.',
    )
    expect(() =>
      decisionRulesForSave({
        type: 'systemone',
        name: 'risk',
        on_error: 'match',
        on_unknown: 'fail_request',
      }),
    ).toThrow(/cannot be combined/)
  })
})
