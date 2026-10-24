import { describe, expect, it } from 'vitest'

import {
  assignRuleIds,
  canonicalToRuleDocument,
  normalizeForDsl,
  ruleDocumentToCanonical,
  ruleSubtreeEquals,
  type CanonicalRule,
  type RuleGroupNode,
  type RuleLeaf,
  type RuleNode,
} from './decisionRuleAst'

const roundTrip = (rules: CanonicalRule) => ruleDocumentToCanonical(canonicalToRuleDocument(rules))

const collectIds = (node: RuleNode | null): string[] => {
  if (!node) return []
  const own = node.id ? [node.id] : []
  return 'conditions' in node ? [...own, ...node.conditions.flatMap(collectIds)] : own
}

const containsKey = (value: unknown, key: string): boolean => {
  if (Array.isArray(value)) return value.some((item) => containsKey(item, key))
  if (typeof value !== 'object' || value === null) return false
  return Object.entries(value).some(([k, v]) => k === key || containsKey(v, key))
}

describe('decision rule AST bridge', () => {
  const cases: Array<[string, CanonicalRule]> = [
    ['an empty root', {}],
    ['a catch-all root AND', { operator: 'AND', conditions: [] }],
    ['a single leaf root', { type: 'keyword', name: 'urgent' }],
    [
      'nested same-operator groups',
      {
        operator: 'AND',
        conditions: [
          { operator: 'AND', conditions: [{ type: 'keyword', name: 'a' }, { type: 'keyword', name: 'b' }] },
          { type: 'keyword', name: 'c' },
        ],
      },
    ],
    ['a single-child OR', { operator: 'OR', conditions: [{ type: 'domain', name: 'math' }] }],
    [
      'NOT of an AND that contains a NOT',
      {
        operator: 'NOT',
        conditions: [
          {
            operator: 'AND',
            conditions: [
              { type: 'keyword', name: 'opinion' },
              { operator: 'NOT', conditions: [{ type: 'keyword', name: 'risk' }] },
            ],
          },
        ],
      },
    ],
    [
      'leaves with label, predicate and on_error',
      {
        operator: 'OR',
        conditions: [
          { type: 'classifier', name: 'intent', label: 'code', predicate: { gte: 0.7, lt: 0.95 }, on_error: 'match' },
          { type: 'decision', name: 'route', label: 'fast', on_error: 'no_match' },
        ],
      },
    ],
    [
      'a root on_unknown',
      { operator: 'AND', conditions: [{ type: 'classifier', name: 'intent', label: 'code', predicate: { gte: 0.6 } }], on_unknown: 'fail_request' },
    ],
    ['on_unknown on an empty root', { on_unknown: 'match' }],
    [
      'unknown leaf and root keys',
      {
        operator: 'AND',
        conditions: [{ type: 'keyword', name: 'a', weight: 2, notes: { owner: 'ops' } }],
        x_owner: 'routing',
      },
    ],
    ['unknown keys on an empty root', { x_owner: 'routing' }],
  ]

  it.each(cases)('round-trips %s losslessly', (_, rules) => {
    expect(roundTrip(rules)).toEqual(rules)
  })

  it('keeps repeated classifier references distinct by node, not by type and name', () => {
    const rules: CanonicalRule = {
      operator: 'OR',
      conditions: [
        { type: 'classifier', name: 'x', label: 'b', predicate: { gte: 0.9 } },
        { type: 'classifier', name: 'x', label: 'a', predicate: { gt: 0.5 }, on_error: 'match' },
      ],
    }
    const doc = canonicalToRuleDocument(rules)
    const [first, second] = (doc.tree as { conditions: RuleLeaf[] }).conditions
    expect(first.id).not.toEqual(second.id)
    expect(first.label).toBe('b')
    expect(second.label).toBe('a')
    expect(second.onError).toBe('match')
    expect(ruleDocumentToCanonical(doc)).toEqual(rules)
  })

  it('gives every node a unique id and strips ids from the canonical output', () => {
    const doc = canonicalToRuleDocument(cases[5][1])
    const ids = collectIds(doc.tree)
    expect(ids).toHaveLength(5)
    expect(new Set(ids).size).toBe(ids.length)
    expect(containsKey(ruleDocumentToCanonical(doc), 'id')).toBe(false)
  })

  it('classifies the root shape', () => {
    expect(canonicalToRuleDocument({}).rootShape).toBe('empty')
    expect(canonicalToRuleDocument(undefined)).toEqual({ tree: null, rootShape: 'empty' })
    const catchAll = canonicalToRuleDocument({ operator: 'AND', conditions: [] })
    expect(catchAll.rootShape).toBe('catch_all')
    expect(catchAll.tree).toMatchObject({ operator: 'AND', conditions: [] })
    expect(canonicalToRuleDocument({ type: 'keyword', name: 'a' }).rootShape).toBe('leaf')
    expect(canonicalToRuleDocument(cases[3][1]).rootShape).toBe('group')
  })

  it('reads a typeless node with leaf fields as a malformed leaf, not as unconditional', () => {
    const doc = canonicalToRuleDocument({ label: 'code' })
    expect(doc.rootShape).toBe('leaf')
    expect(doc.tree).toMatchObject({ signalType: '', signalName: '', label: 'code' })
    expect(ruleDocumentToCanonical(doc)).toEqual({ label: 'code' })
  })

  it('normalizes operators like Go in the tree but writes back the original spelling', () => {
    const canonical = {
      operator: ' or ' as never,
      conditions: [{ conditions: [{ type: 'keyword', name: 'a' }] }],
    }
    const doc = canonicalToRuleDocument(canonical)
    const root = doc.tree as RuleGroupNode
    expect(root.operator).toBe('OR')
    expect((root.conditions[0] as RuleGroupNode).operator).toBe('AND')
    expect(ruleDocumentToCanonical(doc)).toEqual(canonical)
    expect(roundTrip({ operator: 'AND' })).toEqual({ operator: 'AND' })
    expect(roundTrip({ conditions: [] })).toEqual({ conditions: [] })
  })

  it('writes the canonical operator once the user changes it', () => {
    const doc = canonicalToRuleDocument({
      operator: 'and' as never,
      conditions: [{ conditions: [{ type: 'keyword', name: 'a' }] }],
    })
    const root = doc.tree as RuleGroupNode
    const child = root.conditions[0] as RuleGroupNode
    const edited = { ...doc, tree: { ...root, operator: 'OR', conditions: [{ ...child, operator: 'OR' }] } as RuleNode }
    expect(ruleDocumentToCanonical(edited)).toEqual({
      operator: 'OR',
      conditions: [{ operator: 'OR', conditions: [{ type: 'keyword', name: 'a' }] }],
    })
  })

  it('keeps unknown keys from an empty root once a condition is added', () => {
    const doc = canonicalToRuleDocument({ x_note: 'keep' } as never)
    expect(doc.rootExtra).toEqual({ x_note: 'keep' })
    expect(ruleDocumentToCanonical({ ...doc, tree: { signalType: 'keyword', signalName: 'a' }, rootShape: 'leaf' }))
      .toEqual({ type: 'keyword', name: 'a', x_note: 'keep' })
  })

  it('keeps an unsupported operator verbatim so validation can name it', () => {
    expect(roundTrip({ operator: 'XOR' as never, conditions: [{ type: 'keyword', name: 'a' }] })).toEqual({
      operator: 'XOR',
      conditions: [{ type: 'keyword', name: 'a' }],
    })
  })

  it('keeps a non-string operator verbatim instead of reading it as AND', () => {
    const rules = { operator: 5, conditions: [{ type: 'keyword', name: 'a' }] } as never
    expect(roundTrip(rules)).toEqual(rules)
    expect(roundTrip({ operator: 5 } as never)).toEqual({ operator: 5 })
  })

  it('reads a null rule field as unset, like Go', () => {
    expect(roundTrip({ type: 'keyword', name: 'k', predicate: null, on_error: null } as never)).toEqual({
      type: 'keyword',
      name: 'k',
    })
    expect(roundTrip({ operator: 'AND', conditions: [{ type: 'keyword', name: 'k' }], label: null } as never))
      .toEqual({ operator: 'AND', conditions: [{ type: 'keyword', name: 'k' }] })
  })

  it('only emits defined predicate numbers and never emits undefined keys', () => {
    const doc = canonicalToRuleDocument({ type: 'classifier', name: 'x', label: 'a', predicate: { gte: 0.5 } })
    const leaf = doc.tree as RuleLeaf
    leaf.predicate = { gte: 0.5, lt: undefined }
    leaf.onError = undefined
    const out = ruleDocumentToCanonical(doc)
    expect(out).toEqual({ type: 'classifier', name: 'x', label: 'a', predicate: { gte: 0.5 } })
    expect(Object.keys(out.predicate ?? {})).toEqual(['gte'])
    expect(Object.keys(out)).toEqual(['type', 'name', 'label', 'predicate'])
  })

  it('compares subtrees structurally, ignoring ids', () => {
    const a = canonicalToRuleDocument(cases[6][1]).tree
    const b = canonicalToRuleDocument(cases[6][1]).tree
    expect(collectIds(a)).not.toEqual(collectIds(b))
    expect(ruleSubtreeEquals(a, b)).toBe(true)
    const changed = canonicalToRuleDocument({
      ...cases[6][1],
      conditions: [{ ...cases[6][1].conditions![0], label: 'chat' }, cases[6][1].conditions![1]],
    }).tree
    expect(ruleSubtreeEquals(a, changed)).toBe(false)
    expect(ruleSubtreeEquals(null, null)).toBe(true)
    expect(ruleSubtreeEquals(a, null)).toBe(false)
  })

  it('normalizes to the DSL shape: flattened, unwrapped and without empty groups', () => {
    const nested = canonicalToRuleDocument(cases[3][1]).tree
    const flat = canonicalToRuleDocument({
      operator: 'AND',
      conditions: [{ type: 'keyword', name: 'a' }, { type: 'keyword', name: 'b' }, { type: 'keyword', name: 'c' }],
    }).tree
    expect(ruleSubtreeEquals(normalizeForDsl(nested), normalizeForDsl(flat))).toBe(true)
    expect(ruleSubtreeEquals(nested, flat)).toBe(false)

    const wrapped = canonicalToRuleDocument(cases[4][1]).tree
    expect(normalizeForDsl(wrapped)).toMatchObject({ signalType: 'domain', signalName: 'math' })
    expect(normalizeForDsl(canonicalToRuleDocument({ operator: 'AND', conditions: [] }).tree)).toBeNull()
  })

  it('assigns ids only where they are missing', () => {
    const tree: RuleNode = {
      id: 'keep',
      operator: 'AND',
      conditions: [{ signalType: 'keyword', signalName: 'a' }],
    }
    const withIds = assignRuleIds(tree)
    expect(withIds.id).toBe('keep')
    expect((withIds as { conditions: RuleNode[] }).conditions[0].id).toMatch(/^r\d+$/)
    expect(assignRuleIds(withIds)).toBe(withIds)
  })
})
