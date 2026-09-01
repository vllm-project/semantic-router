import { describe, expect, it } from 'vitest'

import {
  decisionRulesConflict,
  decisionRulesForSave,
  mergeDecisionForSave,
  type DecisionConfig,
} from './configPageSupport'

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
      on_unknown: 'fail_request',
      conditions: [
        { type: 'keyword', name: 'complex' },
        {
          operator: 'OR',
          conditions: [
            { type: 'metadata', name: 'canary' },
            {
              type: 'classifier',
              name: 'risk',
              label: 'RISKY',
              predicate: { gte: 0.8 },
              on_error: 'match',
            },
          ],
        },
      ],
    }

    const result = decisionRulesForSave(existing, {
      operator: 'AND',
      on_unknown: 'match',
      conditions: [{ type: 'keyword', name: 'flattened' }],
    })

    expect(result.operator).toBe(existing.operator)
    expect(result.conditions).toEqual(existing.conditions)
    expect(result.on_unknown).toBe('match')
    expect(result).not.toBe(existing)
  })
})

describe('decision rules on_unknown + on_error conflict', () => {
  const conflictingLeaf = {
    type: 'classifier',
    name: 'risk',
    label: 'RISKY',
    predicate: { gte: 0.5 },
    on_error: 'no_match' as const,
  }

  it('flags a flat conflicting rule set', () => {
    expect(
      decisionRulesConflict({
        operator: 'AND',
        on_unknown: 'no_match',
        conditions: [conflictingLeaf],
      }),
    ).toBe(true)
  })

  it('flags on_error nested below the root', () => {
    expect(
      decisionRulesConflict({
        operator: 'AND',
        on_unknown: 'match',
        conditions: [{ operator: 'OR', conditions: [conflictingLeaf] }],
      }),
    ).toBe(true)
  })

  it('flags a preserved nested tree when the form adds on_unknown', () => {
    const preserved = decisionRulesForSave(
      { operator: 'AND', conditions: [{ operator: 'OR', conditions: [conflictingLeaf] }] },
      { operator: 'AND', conditions: [], on_unknown: 'no_match' },
    )
    expect(decisionRulesConflict(preserved)).toBe(true)
  })

  it('accepts either field alone', () => {
    expect(decisionRulesConflict({ operator: 'AND', conditions: [conflictingLeaf] })).toBe(false)
    expect(
      decisionRulesConflict({
        operator: 'AND',
        on_unknown: 'no_match',
        conditions: [{ type: 'keyword', name: 'x' }],
      }),
    ).toBe(false)
  })
})
