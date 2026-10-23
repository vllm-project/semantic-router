import { describe, expect, it } from 'vitest'

import type { RuleIssue } from '../lib/decisionRuleAst'
import { RAW_EXPRESSION_INCOMPLETE } from '../lib/decisionRuleIssues'
import { decisionModelRefsForSave, decisionRulesForSave } from './configPageDecisionFormSupport'
import { mergeDecisionForSave, replaceDecisionByName, type DecisionConfig } from './configPageSupport'

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

    const result = decisionRulesForSave(existing)

    expect(result).toEqual(existing)
    expect(result).not.toBe(existing)
  })

  it('keeps a single root condition and unconditional rules', () => {
    expect(decisionRulesForSave({ type: 'keyword', name: 'complex' })).toEqual({
      type: 'keyword',
      name: 'complex',
    })
    const classifier: DecisionConfig['rules'] = {
      type: 'classifier',
      name: 'risk',
      label: 'RISKY',
      predicate: { gte: 0.8 },
      on_unknown: 'no_match',
    }
    expect(decisionRulesForSave(classifier)).toEqual(classifier)
    expect(decisionRulesForSave({})).toEqual({})
    expect(decisionRulesForSave({ operator: 'AND', conditions: [] })).toEqual({
      operator: 'AND',
      conditions: [],
    })
  })

  it('leaves rule validity to the Router', () => {
    // The Router rejects this on_error and on_unknown conflict when the
    // config is written; the form passes the rules through unchanged.
    const conflicting: DecisionConfig['rules'] = {
      operator: 'AND',
      on_unknown: 'fail_request',
      conditions: [{ type: 'classifier', name: 'risk', label: 'RISKY', on_error: 'match' }],
    }
    expect(decisionRulesForSave(conflicting)).toEqual(conflicting)
  })

  it('blocks a save while the editor holds an incomplete raw expression', () => {
    const rawIssue: RuleIssue = {
      path: [],
      severity: 'error',
      message: `${RAW_EXPRESSION_INCOMPLETE}: expected ")" at column 18`,
    }
    expect(() => decisionRulesForSave({ type: 'keyword', name: 'complex' }, [rawIssue])).toThrow(
      /Raw expression is incomplete: expected "\)" at column 18/,
    )
    // Other editor issues never block on their own.
    expect(
      decisionRulesForSave({ type: 'keyword', name: 'complex' }, [
        { path: [], severity: 'error', message: 'NOT requires exactly one child condition, got 0' },
      ]),
    ).toEqual({ type: 'keyword', name: 'complex' })
  })
})

describe('decision save placement and model references', () => {
  it('replaces an edited decision in place', () => {
    const config = {
      decisions: [
        { name: 'a', description: '', priority: 1, rules: {}, modelRefs: [] },
        { name: 'b', description: '', priority: 2, rules: {}, modelRefs: [] },
        { name: 'c', description: '', priority: 3, rules: {}, modelRefs: [] },
      ] as DecisionConfig[],
    }
    const next = { ...config.decisions[1], description: 'edited' }
    expect(replaceDecisionByName(config, 'b', next)).toBe(true)
    expect(config.decisions.map((decision) => decision.name)).toEqual(['a', 'b', 'c'])
    expect(config.decisions[1].description).toBe('edited')
    expect(replaceDecisionByName(config, 'missing', next)).toBe(false)
  })

  it('keeps model reference keys the form does not model', () => {
    const refs = decisionModelRefsForSave([
      {
        model: ' model-a ',
        use_reasoning: true,
        reasoning_effort: '',
        weight: 2,
        ...({ fallback_rank: 1, metadata: { team: 'x' } } as Record<string, unknown>),
      } as DecisionConfig['modelRefs'][number],
    ])
    expect(refs).toEqual([
      { model: 'model-a', use_reasoning: true, weight: 2, fallback_rank: 1, metadata: { team: 'x' } },
    ])
  })
})
