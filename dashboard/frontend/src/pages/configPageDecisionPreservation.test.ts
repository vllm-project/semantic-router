import { describe, expect, it } from 'vitest'

import type { RuleIssue } from '../lib/decisionRuleAst'
import {
  ON_UNKNOWN_ON_ERROR_CONFLICT_MESSAGE,
  RAW_EXPRESSION_INCOMPLETE,
  ruleSignalScope,
} from '../lib/decisionRuleValidation'
import { decisionModelRefsForSave, decisionRulesForSave } from './configPageDecisionFormSupport'
import { mergeDecisionForSave, replaceDecisionByName, type DecisionConfig } from './configPageSupport'

const scope = ruleSignalScope(
  [
    { signalType: 'keyword', name: 'complex' },
    { signalType: 'metadata', name: 'canary' },
    { signalType: 'classifier', name: 'risk', labels: ['RISKY'], labelRequired: true },
  ],
  true,
)

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

    const result = decisionRulesForSave(existing, scope)

    expect(result).toEqual(existing)
    expect(result).not.toBe(existing)
  })

  it('rejects a root unknown policy combined with a classifier on_error policy', () => {
    expect(() =>
      decisionRulesForSave(
        {
          operator: 'AND',
          on_unknown: 'fail_request',
          conditions: [
            { type: 'classifier', name: 'risk', label: 'RISKY', predicate: { gte: 0.8 }, on_error: 'match' },
          ],
        },
        scope,
      ),
    ).toThrow(ON_UNKNOWN_ON_ERROR_CONFLICT_MESSAGE)
  })

  it('rejects a single-leaf root combining on_error and on_unknown', () => {
    expect(() =>
      decisionRulesForSave(
        { type: 'classifier', name: 'risk', label: 'RISKY', predicate: { gte: 0.8 }, on_error: 'match', on_unknown: 'fail_request' },
        scope,
      ),
    ).toThrow(ON_UNKNOWN_ON_ERROR_CONFLICT_MESSAGE)
  })

  it('rejects on_error on a composite root', () => {
    expect(() =>
      decisionRulesForSave(
        {
          operator: 'AND',
          on_error: 'match',
          conditions: [{ type: 'keyword', name: 'complex' }],
        },
        scope,
      ),
    ).toThrow(/only apply to leaf conditions/)
  })

  it('keeps a single root condition and unconditional rules', () => {
    expect(decisionRulesForSave({ type: 'keyword', name: 'complex' }, scope)).toEqual({
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
    expect(decisionRulesForSave(classifier, scope)).toEqual(classifier)
    expect(decisionRulesForSave({}, scope)).toEqual({})
    expect(decisionRulesForSave({ operator: 'AND', conditions: [] }, scope)).toEqual({
      operator: 'AND',
      conditions: [],
    })
  })

  it('blocks a save with path-qualified, Go-equivalent errors', () => {
    expect(() => decisionRulesForSave({ type: 'keyword' }, scope)).toThrow(
      'signal condition of type "keyword" requires a name',
    )
    expect(() =>
      decisionRulesForSave(
        {
          operator: 'AND',
          conditions: [
            { type: 'keyword', name: 'complex' },
            { operator: 'NOT', conditions: [] },
          ],
        },
        scope,
      ),
    ).toThrow('rules.conditions[1]: NOT requires exactly one child condition, got 0')
    // The recipe scope is strict: an undeclared reference is an error.
    expect(() =>
      decisionRulesForSave({ type: 'keyword', name: 'root-only' }, scope),
    ).toThrow('signal keyword("root-only") is not declared in this recipe')
    expect(() =>
      decisionRulesForSave(
        { type: 'classifier', name: 'risk', label: 'SAFE', predicate: { gte: 0.8 } },
        scope,
      ),
    ).toThrow(/requires a declared label/)
  })

  it('blocks a save while the editor holds an incomplete raw expression', () => {
    const rawIssue: RuleIssue = {
      path: [],
      severity: 'error',
      message: `${RAW_EXPRESSION_INCOMPLETE}: expected ")" at column 18`,
    }
    expect(() =>
      decisionRulesForSave({ type: 'keyword', name: 'complex' }, scope, [rawIssue]),
    ).toThrow(/Raw expression is incomplete: expected "\)" at column 18/)
    // Warnings and other editor issues never block on their own.
    expect(
      decisionRulesForSave({ type: 'keyword', name: 'complex' }, scope, [
        { path: [], severity: 'warning', message: 'advice' },
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
