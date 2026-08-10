import { describe, expect, it } from 'vitest'

import {
  collectLeafMetadata,
  decisionRuleSetToExprText,
  decisionRuleSetToRuleNode,
  exprTextToDecisionRuleSet,
  ruleNodeToDecisionRuleSet,
  validateDecisionRules,
} from './configPageDecisionRuleBridge'
import type { DecisionRuleSet } from './configPageSupport'

// Mirrors the `safe_hybrid_route` decision checked into config/config.yaml: an AND root with a
// nested OR and a nested NOT, the exact shape ConfigPage's flat editor could view but not edit.
const safeHybridRoute: DecisionRuleSet = {
  operator: 'AND',
  conditions: [
    { type: 'domain', name: 'business' },
    {
      operator: 'OR',
      conditions: [
        { type: 'keyword', name: 'urgent_keywords' },
        { type: 'complexity', name: 'needs_reasoning:hard' },
      ],
    },
    {
      operator: 'NOT',
      conditions: [{ type: 'jailbreak', name: 'prompt_injection' }],
    },
  ],
}

describe('decisionRuleSetToExprText / exprTextToDecisionRuleSet round trip', () => {
  it('round-trips a deeply nested AND/OR/NOT tree without structural loss', () => {
    const text = decisionRuleSetToExprText(safeHybridRoute)
    const metadata = collectLeafMetadata(safeHybridRoute)
    const { rules, error } = exprTextToDecisionRuleSet(text, metadata, safeHybridRoute)

    expect(error).toBeNull()
    expect(rules).toEqual(safeHybridRoute)
  })

  it('round-trips a single-condition rule set despite text serialization dropping the wrapper', () => {
    const single: DecisionRuleSet = {
      operator: 'AND',
      conditions: [{ type: 'keyword', name: 'x' }],
    }
    const text = decisionRuleSetToExprText(single)
    expect(text).toBe('keyword("x")')

    const { rules, error } = exprTextToDecisionRuleSet(text, collectLeafMetadata(single), single)
    expect(error).toBeNull()
    expect(rules).toEqual(single)
  })

  it('treats an empty rule set as an empty string, not a placeholder expression', () => {
    const empty: DecisionRuleSet = { operator: 'AND', conditions: [] }
    expect(decisionRuleSetToExprText(empty)).toBe('')

    const { rules, error } = exprTextToDecisionRuleSet('', new Map(), safeHybridRoute)
    expect(error).toBeNull()
    expect(rules).toEqual(empty)
  })

  it('keeps the previous rules and reports an error for unparseable text', () => {
    const { rules, error } = exprTextToDecisionRuleSet(
      'domain("business") AND (',
      collectLeafMetadata(safeHybridRoute),
      safeHybridRoute,
    )
    expect(error).toContain('Could not parse')
    expect(rules).toBe(safeHybridRoute)
  })

  it('preserves label/predicate/on_error metadata for leaves the edit did not touch', () => {
    const withMetadata: DecisionRuleSet = {
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
    }
    const metadata = collectLeafMetadata(withMetadata)
    const node = decisionRuleSetToRuleNode(withMetadata)

    // Simulate a structural edit elsewhere in the tree: the classifier leaf itself is untouched.
    const rebuilt = ruleNodeToDecisionRuleSet(node, metadata)

    expect(rebuilt).toEqual(withMetadata)
  })

  it('drops metadata for a leaf whose signal identity changed', () => {
    const original: DecisionRuleSet = {
      operator: 'AND',
      conditions: [
        {
          type: 'classifier',
          name: 'risk',
          label: 'RISKY',
          predicate: { gte: 0.8 },
          on_error: 'match',
        },
      ],
    }
    const metadata = collectLeafMetadata(original)

    // The user retargeted the leaf to a different signal name in the editor.
    const edited = ruleNodeToDecisionRuleSet(
      { operator: 'AND', conditions: [{ signalType: 'classifier', signalName: 'other' }] },
      metadata,
    )

    expect(edited).toEqual({
      operator: 'AND',
      conditions: [{ type: 'classifier', name: 'other' }],
    })
  })
})

describe('validateDecisionRules', () => {
  const availableSignals = [
    { signalType: 'domain', name: 'business' },
    { signalType: 'keyword', name: 'urgent_keywords' },
  ]

  it('requires at least one condition', () => {
    expect(validateDecisionRules({ operator: 'AND', conditions: [] }, availableSignals)).toEqual([
      'At least one condition is required.',
    ])
  })

  it('flags a NOT node with more than one child, matching backend evalNOT semantics', () => {
    const warnings = validateDecisionRules(
      {
        operator: 'NOT',
        conditions: [
          { type: 'domain', name: 'business' },
          { type: 'keyword', name: 'urgent_keywords' },
        ],
      },
      availableSignals,
    )
    expect(warnings).toContain('NOT must have exactly one child')
  })

  it('flags a reference to an undefined signal', () => {
    const warnings = validateDecisionRules(
      { operator: 'AND', conditions: [{ type: 'domain', name: 'unknown' }] },
      availableSignals,
    )
    expect(warnings).toEqual(['Signal domain("unknown") is not defined'])
  })

  it('passes for a valid, fully-defined tree', () => {
    expect(
      validateDecisionRules(
        { operator: 'AND', conditions: [{ type: 'domain', name: 'business' }] },
        availableSignals,
      ),
    ).toEqual([])
  })
})
