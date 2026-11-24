import { describe, expect, it } from 'vitest'

import type { BoolExprNode } from '@/types/dsl'
import { incompleteConditionCount, ruleTreeToWhen, whenToRuleTree } from './editorRules'

const pos = { Line: 1, Column: 1 }
const ref = (signalType: string, signalName: string, fields?: Record<string, unknown>) =>
  ({
    type: 'signal_ref',
    signalType,
    signalName,
    pos,
    ...(fields ? { fields } : {}),
  }) as BoolExprNode
const and = (left: BoolExprNode, right: BoolExprNode) =>
  ({ type: 'and', left, right, pos }) as BoolExprNode
const or = (left: BoolExprNode, right: BoolExprNode) =>
  ({ type: 'or', left, right, pos }) as BoolExprNode
const not = (expr: BoolExprNode) => ({ type: 'not', expr, pos }) as BoolExprNode

describe('route WHEN expressions in the shared rule tree', () => {
  it('keeps the group a NOT applies to', () => {
    const expr = and(
      ref('keyword', 'legal_terms'),
      not(and(ref('keyword', 'opinion_request'), not(ref('keyword', 'risk_markers')))),
    )
    expect(ruleTreeToWhen(whenToRuleTree(expr))).toBe(
      'keyword("legal_terms") AND NOT (keyword("opinion_request") AND NOT keyword("risk_markers"))',
    )
  })

  it('flattens chained operators into one group and parenthesises OR', () => {
    const tree = whenToRuleTree(
      and(and(ref('domain', 'math'), ref('keyword', 'urgent')), or(ref('a', 'x'), ref('b', 'y'))),
    )
    expect(tree.operator).toBe('AND')
    expect((tree.conditions as unknown[]).length).toBe(3)
    expect(ruleTreeToWhen(tree)).toBe('domain("math") AND keyword("urgent") AND (a("x") OR b("y"))')
  })

  it('writes condition qualifiers the AST carries instead of dropping them', () => {
    const tree = whenToRuleTree(ref('complexity', 'difficulty', { label: 'hard' }))
    expect(tree).toEqual({ label: 'hard', type: 'complexity', name: 'difficulty' })
    expect(ruleTreeToWhen(tree)).toBe('complexity("difficulty", label: "hard")')
  })

  it('treats an empty tree as a decision that always matches', () => {
    expect(whenToRuleTree(null)).toEqual({})
    expect(ruleTreeToWhen({})).toBe('')
  })

  it('leaves out conditions that do not name a signal yet', () => {
    const draft = { operator: 'AND', conditions: [{ type: 'domain', name: 'math' }, {}] }
    expect(ruleTreeToWhen(draft)).toBe('domain("math")')
    expect(incompleteConditionCount(draft)).toBe(1)
    expect(ruleTreeToWhen({ operator: 'OR', conditions: [{ type: 'domain' }] })).toBe('')
  })
})
