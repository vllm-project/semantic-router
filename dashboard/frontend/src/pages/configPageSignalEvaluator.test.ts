import { describe, expect, it } from 'vitest'

import { keywordOccurrences, signalEvaluator, signalThreshold } from './configPageSignalEvaluator'

describe('signal evaluators', () => {
  it('follows the binding order: rule deployment, recipe, global, default', () => {
    const tasks = { default_deployment: 'primary', default_bindings: { pii: 'primary' } } as never
    expect(signalEvaluator('pii', { deployment: 'local' }, { tasks })).toEqual({
      kind: 'model',
      deployment: 'local',
      source: 'rule',
    })
    expect(
      signalEvaluator(
        'pii',
        {},
        {
          tasks,
          recipeBindings: { pii: { deployment: 'recipe-pii' } },
          globalBindings: { pii: 'global-pii' },
        },
      ),
    ).toEqual({
      kind: 'model',
      deployment: 'recipe-pii',
      source: 'recipe binding',
    })
    expect(
      signalEvaluator('pii', {}, { tasks, globalBindings: { pii: 'global-pii' } }).deployment,
    ).toBe('global-pii')
    expect(signalEvaluator('pii', {}, { tasks }).source).toBe('default binding')
    expect(
      signalEvaluator('systemone', {}, { tasks, decisionModel: 'decision-kai' }).deployment,
    ).toBe('decision-kai')
    expect(signalEvaluator('keyword', { method: 'bm25' }, {})).toEqual({
      kind: 'rule',
      method: 'bm25',
    })
    expect(signalEvaluator('keyword', {}, {})).toEqual({ kind: 'rule', method: 'literal' })
    expect(signalEvaluator('context', {}, {})).toEqual({ kind: 'rule' })
  })

  it('locates literal keywords as words and regular expressions as written', () => {
    expect(
      keywordOccurrences('Urgent: fix it ASAP, not asaply', { keywords: ['asap', 'urgent'] }),
    ).toEqual([
      { start: 15, end: 19 },
      { start: 0, end: 6 },
    ])
    expect(keywordOccurrences('One? Two? Three', { method: 'regex', keywords: ['[?？]'] })).toEqual(
      [
        { start: 3, end: 4 },
        { start: 8, end: 9 },
      ],
    )
    expect(
      keywordOccurrences('1. a\n2. b', { method: 'regex', keywords: ['(?m)^\\s*\\d+\\.\\s+'] }),
    ).toHaveLength(2)
    expect(keywordOccurrences('Code', { keywords: ['code'], case_sensitive: true })).toEqual([])
    expect(keywordOccurrences('anything', { method: 'bm25', keywords: ['code'] })).toBeNull()
  })

  it('reads the threshold a rule states', () => {
    expect(signalThreshold({ predicate: { gte: 0.7 } })).toBe(0.7)
    expect(signalThreshold({ threshold: 0.3 })).toBe(0.3)
    expect(signalThreshold({})).toBeUndefined()
  })
})
