import { describe, expect, it } from 'vitest'

import { serializeBoolExpr } from '@/lib/dslMutations'
import type { BoolExprNode } from '@/types/dsl'

import { boolExprToRuleNode, parseExprText, serializeNode } from './ExpressionBuilderSupport'

const pos = { Line: 1, Column: 1 }

describe('expression builder condition fields', () => {
  it('writes the fields of a route condition in both condition serializers', () => {
    const when: BoolExprNode = {
      type: 'and',
      left: { type: 'signal_ref', signalType: 'domain', signalName: 'business', pos },
      right: {
        type: 'signal_ref',
        signalType: 'classifier',
        signalName: 'generic-safety-score',
        fields: { label: 'unsafe', predicate: { gte: 0.5 } },
        pos,
      },
      pos,
    }
    const expected =
      'domain("business") AND classifier("generic-safety-score", label: "unsafe", predicate: { gte: 0.5 })'

    expect(serializeBoolExpr(when)).toBe(expected)
    const tree = boolExprToRuleNode(when as unknown as Record<string, unknown>)
    expect(tree && serializeNode(tree)).toBe(expected)
  })

  it('reads condition fields from raw expression text', () => {
    const tree = parseExprText(
      'classifier("risk", label: "unsafe" predicate: {gte: 0.5, lt: 1}) OR keyword("urgent", terms: ["a", "b"], strict: true)',
    )

    expect(tree).toEqual({
      operator: 'OR',
      conditions: [
        {
          signalType: 'classifier',
          signalName: 'risk',
          fields: { label: 'unsafe', predicate: { gte: 0.5, lt: 1 } },
        },
        {
          signalType: 'keyword',
          signalName: 'urgent',
          fields: { terms: ['a', 'b'], strict: true },
        },
      ],
    })
    expect(tree && serializeNode(tree)).toBe(
      'classifier("risk", label: "unsafe", predicate: { gte: 0.5, lt: 1 }) OR keyword("urgent", terms: ["a", "b"], strict: true)',
    )
  })
})
