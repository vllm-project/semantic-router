import { describe, expect, it } from 'vitest'

import type { BoolExprNode } from '@/types/dsl'

import {
  canonicalToRuleDocument,
  normalizeForDsl,
  ruleDocumentToCanonical,
  ruleSubtreeEquals,
  type RuleNode,
} from './decisionRuleAst'
import { boolExprToRuleNode, parseRuleExpr, serializeRuleExpr } from './decisionRuleDsl'
import { serializeBoolExpr } from './dslMutations'

const leaf = (signalType: string, signalName: string, extra: Partial<RuleNode> = {}): RuleNode =>
  ({ signalType, signalName, ...extra }) as RuleNode

const parsed = (text: string) => {
  const result = parseRuleExpr(text)
  expect(result.error).toBeUndefined()
  return result.tree
}

describe('serializeRuleExpr', () => {
  it('writes leaf fields in Go decompileRuleLeaf order', () => {
    expect(
      serializeRuleExpr(
        leaf('classifier', 'intent', { label: 'code', predicate: { gte: 0.7, lt: 0.95 }, onError: 'match' }),
      ),
    ).toEqual({
      text: 'classifier("intent", label: "code", predicate: { gte: 0.7, lt: 0.95 }, on_error: "match")',
      errors: [],
    })
  })

  it('parenthesizes by Go precedence: OR < AND < NOT', () => {
    const tree: RuleNode = {
      operator: 'OR',
      conditions: [
        { operator: 'AND', conditions: [leaf('keyword', 'a'), leaf('keyword', 'b')] },
        { operator: 'NOT', conditions: [{ operator: 'OR', conditions: [leaf('keyword', 'c'), leaf('keyword', 'd')] }] },
      ],
    }
    expect(serializeRuleExpr(tree).text).toBe(
      '(keyword("a") AND keyword("b")) OR NOT (keyword("c") OR keyword("d"))',
    )
    const and: RuleNode = {
      operator: 'AND',
      conditions: [{ operator: 'OR', conditions: [leaf('keyword', 'a'), leaf('keyword', 'b')] }, leaf('keyword', 'c')],
    }
    expect(serializeRuleExpr(and).text).toBe('(keyword("a") OR keyword("b")) AND keyword("c")')
  })

  it('writes exponent-form predicates as exact decimals', () => {
    const text = (gte: number) => serializeRuleExpr(leaf('embedding', 'e', { predicate: { gte } })).text
    expect(text(1e-21)).toBe('embedding("e", predicate: { gte: 0.000000000000000000001 })')
    expect(text(1.2345678901234e-15)).toBe('embedding("e", predicate: { gte: 0.0000000000000012345678901234 })')
    expect(text(-2.5e-7)).toBe('embedding("e", predicate: { gte: -0.00000025 })')
    expect(text(1.5e21)).toBe('embedding("e", predicate: { gte: 1500000000000000000000 })')
    expect(parseRuleExpr(text(1.2345678901234e-15)).tree).toMatchObject({ predicate: { gte: 1.2345678901234e-15 } })
  })

  it('escapes quotes and backslashes in names', () => {
    expect(serializeRuleExpr(leaf('keyword', 'say "hi" \\ bye')).text).toBe('keyword("say \\"hi\\" \\\\ bye")')
  })

  it('reports empty groups and NOT cardinality instead of hiding them', () => {
    const empty = serializeRuleExpr({ operator: 'AND', conditions: [leaf('keyword', 'a'), { operator: 'OR', conditions: [] }] })
    expect(empty.text).toBe('keyword("a") AND (? OR ?)')
    expect(empty.errors.map((e) => e.message)).toEqual(['OR combination requires at least one child condition'])
    expect(empty.errors[0].path).toEqual([1])

    const not = serializeRuleExpr({ operator: 'NOT', conditions: [] as unknown as [RuleNode] })
    expect(not.text).toBe('NOT (?)')
    expect(not.errors[0].message).toBe('NOT requires exactly one child condition, got 0')
  })

  it('treats a childless root AND as the unconditional (no WHEN) expression', () => {
    expect(serializeRuleExpr({ operator: 'AND', conditions: [] })).toEqual({ text: '', errors: [] })
    expect(serializeRuleExpr(null)).toEqual({ text: '', errors: [] })
  })

  it('explicitly rejects fields the DSL cannot express', () => {
    const result = serializeRuleExpr(leaf('keyword', 'a', { extra: { weight: 2 } }))
    expect(result.errors.map((e) => e.message)).toEqual([
      'field weight is not supported by the router and cannot be expressed in DSL',
    ])
  })
})

describe('parseRuleExpr', () => {
  it('round-trips fields, escapes and precedence', () => {
    const texts = [
      'classifier("intent", label: "code", predicate: { gte: 0.7, lt: 0.95 }, on_error: "match")',
      'keyword("say \\"hi\\" \\\\ bye")',
      '(keyword("a") AND keyword("b")) OR NOT (keyword("c") OR keyword("d"))',
      'keyword("a") AND (keyword("b") AND keyword("c"))',
      'NOT NOT decision("route", label: "fast")',
      'projection("effort_high") OR (complexity("needs_reasoning:hard") AND domain("math"))',
    ]
    for (const text of texts) {
      expect(serializeRuleExpr(parsed(text)).text).toBe(text)
    }
  })

  it('accepts Go lexer forms: bare names, optional commas, negative and integer numbers, comments', () => {
    const tree = parsed('classifier(intent label: code, predicate: {gt: -1 lte: 2}) # trailing comment')
    expect(tree).toMatchObject({
      signalType: 'classifier',
      signalName: 'intent',
      label: 'code',
      predicate: { gt: -1, lte: 2 },
    })
  })

  it('unescapes Go string escapes', () => {
    expect(parsed('keyword("tab\\there \\u00e9 \\x41")')).toMatchObject({ signalName: 'tab\there é A' })
  })

  it('reads AND as binding tighter than OR', () => {
    expect(serializeRuleExpr(parsed('keyword("a") OR keyword("b") AND keyword("c")')).text).toBe(
      'keyword("a") OR (keyword("b") AND keyword("c"))',
    )
  })

  it('flattens same-operator chains but keeps parenthesized groups', () => {
    expect(serializeRuleExpr(parsed('keyword("a") OR keyword("b") OR keyword("c")')).text).toBe(
      'keyword("a") OR keyword("b") OR keyword("c")',
    )
    const tree = parsed('keyword("a") OR keyword("b") OR keyword("c")') as { conditions: RuleNode[] }
    expect(tree.conditions).toHaveLength(3)
  })

  it('gives every parsed node a fresh id', () => {
    const tree = parsed('keyword("a") AND NOT keyword("a")') as { id: string; conditions: RuleNode[] }
    const ids = [tree.id, tree.conditions[0].id, tree.conditions[1].id]
    expect(new Set(ids).size).toBe(3)
  })

  it('treats empty text as unconditional', () => {
    expect(parseRuleExpr('')).toEqual({ tree: null })
    expect(parseRuleExpr('  \n ')).toEqual({ tree: null })
  })

  it.each([
    ['a missing ")"', 'keyword("a"', 12],
    ['a trailing AND', 'keyword("a") AND', 17],
    ['two adjacent conditions', 'keyword("a") keyword("b")', 14],
    ['a bare word', 'keyword', 8],
    ['a lowercase operator', 'keyword("a") and keyword("b")', 14],
    ['an unknown field key', 'keyword("a", weight: 2)', 14],
    ['an empty name', 'keyword("")', 9],
    ['an unknown predicate key', 'classifier("a", predicate: { eq: 1 })', 30],
    ['a missing group ")"', '(keyword("a") OR keyword("b")', 30],
    ['an unterminated string', 'keyword("a)', 9],
  ])('rejects %s with its column', (_, text, column) => {
    const result = parseRuleExpr(text)
    expect(result.tree).toBeNull()
    expect(result.error?.offset).toBe(column - 1)
    expect(result.error?.message).toMatch(new RegExp(`at column ${column}$`))
  })
})

describe('DSL AST conversion and Builder serialization', () => {
  const pos = { Line: 1, Column: 1 } as never
  const ref = (signalType: string, signalName: string, fields?: Record<string, never>): BoolExprNode => ({
    type: 'signal_ref',
    signalType,
    signalName,
    ...(fields ? { fields } : {}),
    pos,
  })
  const and = (left: BoolExprNode, right: BoolExprNode): BoolExprNode => ({ type: 'and', left, right, pos })
  const or = (left: BoolExprNode, right: BoolExprNode): BoolExprNode => ({ type: 'or', left, right, pos })

  it('reads signal_ref fields and keeps unknown ones as extra', () => {
    const node = boolExprToRuleNode(
      ref('classifier', 'x', { label: 'y', predicate: { gte: 0.7 }, on_error: 'match', weight: 3 } as never),
    )
    expect(node).toMatchObject({
      signalType: 'classifier',
      signalName: 'x',
      label: 'y',
      predicate: { gte: 0.7 },
      onError: 'match',
      extra: { weight: 3 },
    })
  })

  it('flattens binary same-operator chains like the Go compiler', () => {
    const node = boolExprToRuleNode(and(and(ref('keyword', 'a'), ref('keyword', 'b')), or(ref('keyword', 'c'), or(ref('keyword', 'd'), ref('keyword', 'e')))))
    expect(ruleDocumentToCanonical({ tree: node, rootShape: 'group' })).toEqual({
      operator: 'AND',
      conditions: [
        { type: 'keyword', name: 'a' },
        { type: 'keyword', name: 'b' },
        {
          operator: 'OR',
          conditions: [{ type: 'keyword', name: 'c' }, { type: 'keyword', name: 'd' }, { type: 'keyword', name: 'e' }],
        },
      ],
    })
  })

  it('fails closed instead of dropping an unconvertible child', () => {
    expect(boolExprToRuleNode(and(ref('keyword', 'a'), { type: 'mystery' } as never))).toBeNull()
    expect(boolExprToRuleNode({ type: 'not', expr: null } as never)).toBeNull()
  })

  it('emits signal reference fields through serializeBoolExpr', () => {
    expect(
      serializeBoolExpr(and(ref('classifier', 'x', { label: 'y', predicate: { gte: 0.7 } } as never), ref('keyword', 'k'))),
    ).toBe('classifier("x", label: "y", predicate: { gte: 0.7 }) AND keyword("k")')
    expect(serializeBoolExpr(or(ref('keyword', 'a'), ref('keyword', 'b')))).toBe('keyword("a") OR keyword("b")')
    expect(serializeBoolExpr(null)).toBe('')
  })

  it('gives equal subtrees for the same edit through the DSL and canonical adapters', () => {
    const canonical = {
      operator: 'AND' as const,
      conditions: [
        {
          operator: 'OR' as const,
          conditions: [
            { type: 'classifier', name: 'intent', label: 'b', predicate: { gte: 0.7 }, on_error: 'match' as const },
            { type: 'keyword', name: 'urgent' },
          ],
        },
        { operator: 'NOT' as const, conditions: [{ type: 'keyword', name: 'spam' }] },
      ],
    }
    const viaCanonical = canonicalToRuleDocument(canonical).tree
    const viaDsl = parsed(serializeRuleExpr(viaCanonical).text)
    expect(ruleSubtreeEquals(normalizeForDsl(viaDsl), normalizeForDsl(viaCanonical))).toBe(true)
    expect(ruleDocumentToCanonical({ tree: viaDsl, rootShape: 'group' })).toEqual(canonical)
  })
})
