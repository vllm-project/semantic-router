import { describe, expect, it } from 'vitest'

import type { BoolExprNode } from '@/types/dsl'

import { serializeBoolExpr, serializeFields, updateRoute, updateSignal } from './dslMutations'

describe('DSL string literals written by Builder mutations', () => {
  it('escapes quotes, backslashes, and newlines in field values', () => {
    expect(
      serializeFields({
        description: 'Prompts with numbered list items such as "1. ..."',
        pattern: String.raw`(?m)^\s*\d+\.\s+\bstep\b`,
        note: 'first line\nsecond line',
      }),
    ).toBe(
      [
        String.raw`  description: "Prompts with numbered list items such as \"1. ...\""`,
        String.raw`  pattern: "(?m)^\\s*\\d+\\.\\s+\\bstep\\b"`,
        String.raw`  note: "first line\nsecond line"`,
      ].join('\n'),
    )
  })

  it('keeps an unchanged signal save parseable', () => {
    const source = [
      'SIGNAL structure numbered_steps {',
      String.raw`  description: "Prompts with numbered list items such as \"1. ...\""`,
      '}',
      '',
    ].join('\n')

    expect(
      updateSignal(source, 'structure', 'numbered_steps', {
        description: 'Prompts with numbered list items such as "1. ..."',
      }),
    ).toBe(source)
  })

  it('escapes route descriptions and model references', () => {
    const updated = updateRoute('ROUTE support {\n  PRIORITY 1\n}\n', 'support', {
      description: 'Answers "how do I" questions',
      priority: 5,
      models: [{ model: 'model "a"', effort: 'high' }],
      plugins: [],
    })

    expect(updated).toContain(
      String.raw`ROUTE support (description = "Answers \"how do I\" questions") {`,
    )
    expect(updated).toContain(String.raw`MODEL "model \"a\"" (effort = "high")`)
  })
})

describe('route conditions written by the Builder route form', () => {
  const pos = { Line: 1, Column: 1 }
  const keyword = (signalName: string): BoolExprNode => ({
    type: 'signal_ref',
    signalType: 'keyword',
    signalName,
    pos,
  })
  const and = (left: BoolExprNode, right: BoolExprNode): BoolExprNode => ({
    type: 'and',
    left,
    right,
    pos,
  })
  const or = (left: BoolExprNode, right: BoolExprNode): BoolExprNode => ({
    type: 'or',
    left,
    right,
    pos,
  })
  const not = (expr: BoolExprNode): BoolExprNode => ({ type: 'not', expr, pos })

  it('keeps the AND group a NOT applies to', () => {
    expect(
      serializeBoolExpr(
        and(
          keyword('legal_terms'),
          not(and(keyword('opinion_request'), not(keyword('risk_markers')))),
        ),
      ),
    ).toBe(
      'keyword("legal_terms") AND NOT (keyword("opinion_request") AND NOT keyword("risk_markers"))',
    )
  })

  it('writes other NOT operands as before', () => {
    expect(serializeBoolExpr(not(keyword('a')))).toBe('NOT keyword("a")')
    expect(serializeBoolExpr(not(or(keyword('a'), keyword('b'))))).toBe(
      'NOT (keyword("a") OR keyword("b"))',
    )
    expect(serializeBoolExpr(not(not(keyword('a'))))).toBe('NOT NOT keyword("a")')
  })
})
