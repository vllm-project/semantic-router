import { describe, expect, it } from 'vitest'

import { canonicalToRuleDocument, type CanonicalRule, type SignalDescriptor } from './decisionRuleAst'
import {
  formatRuleIssuePath,
  numericPredicateError,
  ON_UNKNOWN_ON_ERROR_CONFLICT_MESSAGE,
  ruleSignalScope,
  validateRuleDocument,
} from './decisionRuleValidation'

const descriptors: SignalDescriptor[] = [
  { signalType: 'keyword', name: 'urgent' },
  { signalType: 'metadata', name: 'tenant' },
  { signalType: 'projection', name: 'effort_high' },
  { signalType: 'complexity', name: 'reasoning:easy' },
  { signalType: 'complexity', name: 'reasoning:medium' },
  { signalType: 'complexity', name: 'reasoning:hard' },
  { signalType: 'classifier', name: 'intent', labels: ['code', 'chat'], labelRequired: true, localClassifier: false },
  { signalType: 'classifier', name: 'local_intent', labels: ['code'], labelRequired: true, localClassifier: true },
  {
    signalType: 'classifier',
    name: 'bound_intent',
    labels: ['code'],
    labelRequired: true,
    localClassifier: false,
    predicateRequired: false,
  },
  { signalType: 'decision', name: 'route', labels: ['fast', 'deep'], labelRequired: true },
  { signalType: 'decision', name: 'difficulty', labels: ['0', '1', '2'], labelForbidden: true },
  { signalType: 'decision', name: 'is_code', labels: ['false', 'true'], labelForbidden: true },
]
const strict = ruleSignalScope(descriptors, true)
const loose = ruleSignalScope(descriptors, false)

const errors = (rules: CanonicalRule, scope = strict) =>
  validateRuleDocument(canonicalToRuleDocument(rules), scope)
    .filter((issue) => issue.severity === 'error')
    .map((issue) => issue.message)

const warnings = (rules: CanonicalRule, scope = strict) =>
  validateRuleDocument(canonicalToRuleDocument(rules), scope)
    .filter((issue) => issue.severity === 'warning')
    .map((issue) => issue.message)

// Cast so tests can also feed values the Go validator must reject.
const classifier = (extra: Record<string, unknown> = {}): CanonicalRule => ({
  type: 'classifier',
  name: 'intent',
  label: 'code',
  predicate: { gte: 0.6 },
  on_error: 'no_match',
  ...extra,
}) as CanonicalRule

describe('validateRuleDocument', () => {
  it('accepts unconditional decisions', () => {
    expect(validateRuleDocument(canonicalToRuleDocument({}), strict)).toEqual([])
    expect(validateRuleDocument(canonicalToRuleDocument({ operator: 'AND', conditions: [] }), strict)).toEqual([])
  })

  it('accepts a valid nested tree', () => {
    expect(
      errors({
        operator: 'AND',
        conditions: [
          { operator: 'OR', conditions: [classifier(), { type: 'keyword', name: 'urgent' }] },
          { operator: 'NOT', conditions: [{ type: 'complexity', name: 'reasoning:hard' }] },
          { type: 'projection', name: 'effort_high' },
          { type: 'decision', name: 'route', label: 'fast', on_error: 'match' },
        ],
      }),
    ).toEqual([])
  })

  describe('node shape and arity (validator_rule_operator.go)', () => {
    it('requires a type on a leaf', () => {
      expect(errors({ name: 'urgent' })[0]).toMatch(/^leaf condition requires a type/)
    })

    it('rejects a node that is both a leaf and a combination', () => {
      expect(errors({ type: 'keyword', name: 'urgent', operator: 'AND', conditions: [] })).toContain(
        'condition must be either a leaf (type/name) or a combination (operator/conditions), not both',
      )
    })

    it('accepts an empty operator or empty conditions on a leaf, as validateRuleNodeShape does', () => {
      expect(errors({ type: 'keyword', name: 'urgent', operator: '' } as unknown as CanonicalRule)).toEqual([])
      expect(errors({ type: 'keyword', name: 'urgent', conditions: [] })).toEqual([])
    })

    it('treats an empty on_unknown as unset', () => {
      expect(errors({ type: 'keyword', name: 'urgent', on_unknown: '' } as unknown as CanonicalRule)).toEqual([])
      expect(errors({ ...classifier(), on_unknown: '' } as unknown as CanonicalRule)).toEqual([])
    })

    it('rejects an unsupported operator', () => {
      expect(errors({ operator: 'XOR' as never, conditions: [{ type: 'keyword', name: 'urgent' }] })).toEqual([
        'unsupported operator "XOR" (expected AND, OR, NOT)',
      ])
    })

    it('requires exactly one child under NOT', () => {
      const two = { type: 'keyword', name: 'urgent' }
      expect(errors({ operator: 'NOT', conditions: [two, two] })).toEqual([
        'NOT requires exactly one child condition, got 2',
      ])
      expect(errors({ operator: 'NOT', conditions: [] })).toEqual(['NOT requires exactly one child condition, got 0'])
    })

    it('requires a child on every non-root group and on a root OR', () => {
      expect(errors({ operator: 'OR', conditions: [] })).toEqual(['OR combination requires at least one child condition'])
      expect(
        errors({ operator: 'AND', conditions: [{ type: 'keyword', name: 'urgent' }, { operator: 'AND', conditions: [] }] }),
      ).toEqual(['AND combination requires at least one child condition'])
      expect(errors({ operator: 'OR', conditions: [{ type: 'keyword', name: 'urgent' }] })).toEqual([])
    })

    it('anchors issues to the node path and id', () => {
      const doc = canonicalToRuleDocument({
        operator: 'AND',
        conditions: [{ type: 'keyword', name: 'urgent' }, { operator: 'OR', conditions: [] }],
      })
      const [issue] = validateRuleDocument(doc, strict)
      expect(issue.path).toEqual([1])
      expect(issue.nodeId).toBe((doc.tree as { conditions: Array<{ id: string }> }).conditions[1].id)
      expect(formatRuleIssuePath(issue.path)).toBe('rules.conditions[1]')
    })
  })

  describe('leaf contract (validator_decision.go)', () => {
    it('rejects a signal type outside the generated decision contract', () => {
      expect(errors({ type: 'hallucination', name: 'h' })).toEqual(['unsupported signal type "hallucination"'])
    })

    it('requires a name', () => {
      expect(errors({ type: 'keyword', name: '  ' })).toEqual(['signal condition of type "keyword" requires a name'])
    })

    it('allows a label only on classifier and decision leaves', () => {
      expect(errors({ type: 'keyword', name: 'urgent', label: 'x' })).toEqual([
        'label is only supported on classifier and decision conditions',
      ])
    })

    it('checks on_error values and where they apply', () => {
      expect(errors(classifier({ on_error: 'retry' }))).toEqual([
        'condition classifier("intent") on_error must be no_match or match',
      ])
      expect(errors({ type: 'keyword', name: 'urgent', on_error: 'match' })).toEqual([
        'condition keyword("urgent") on_error is only supported for classifier and decision conditions',
      ])
    })

    it('rejects on_unknown together with any on_error using the Go message', () => {
      expect(
        errors({ operator: 'AND', conditions: [classifier({ on_error: 'match' })], on_unknown: 'no_match' }),
      ).toEqual([ON_UNKNOWN_ON_ERROR_CONFLICT_MESSAGE])
      expect(ON_UNKNOWN_ON_ERROR_CONFLICT_MESSAGE).toBe(
        'condition on_error has no effect when rules.on_unknown is set; remove one of them',
      )
    })

    it('rejects an invalid on_unknown and an on_unknown below the root', () => {
      expect(errors({ type: 'keyword', name: 'urgent', on_unknown: 'retry' as never })).toEqual([
        'rules on_unknown must be no_match, match, or fail_request',
      ])
      expect(errors({ operator: 'AND', conditions: [{ type: 'keyword', name: 'urgent', on_unknown: 'match' }] })).toEqual([
        'on_unknown is only supported on the root rules node',
      ])
    })

    it('rejects unknown fields explicitly', () => {
      expect(errors({ type: 'keyword', name: 'urgent', weight: 2 })).toEqual(['unsupported field weight'])
      expect(errors({ x_owner: 'ops' })).toEqual(['unsupported field x_owner'])
    })
  })

  describe('numeric predicates (validator_numeric_predicate.go)', () => {
    it.each([
      [{ gte: Number.NaN }, 'predicate values must be finite'],
      [{}, 'predicate must set at least one comparator'],
      [{ gt: 0.1, gte: 0.2 }, 'predicate cannot set both gt and gte'],
      [{ lt: 0.8, lte: 0.9 }, 'predicate cannot set both lt and lte'],
      [{ gte: 0.9, lte: 0.1 }, 'predicate defines an empty numeric range'],
      [{ gt: 0.5, lte: 0.5 }, 'predicate defines an empty numeric range'],
      [{ eq: 0.5 }, 'predicate field eq is not supported (expected gt, gte, lt, lte)'],
    ])('rejects %j', (predicate, message) => {
      expect(numericPredicateError(predicate as never)).toBe(message)
      expect(errors(classifier({ predicate }))).toEqual([`condition classifier("intent") ${message}`])
    })

    it('accepts a closed single-point range', () => {
      expect(numericPredicateError({ gte: 0.5, lte: 0.5 })).toBeNull()
    })
  })

  describe('classifier leaves', () => {
    it('requires a declared classifier', () => {
      expect(errors(classifier({ name: 'missing' }), loose)).toEqual([
        'classifier condition references unknown signal "missing"',
      ])
    })

    it('requires a declared label', () => {
      expect(errors(classifier({ label: undefined }))).toEqual(['classifier condition "intent" requires a declared label'])
      expect(errors(classifier({ label: 'other' }))).toEqual(['classifier condition "intent" requires a declared label'])
    })

    it('requires a predicate unless a label_scores operating point supplies one', () => {
      expect(errors(classifier({ predicate: undefined }))).toEqual([
        'classifier condition "intent" requires a score predicate',
      ])
      expect(errors(classifier({ name: 'bound_intent', predicate: undefined }))).toEqual([])
    })

    it('allows only gte >= 0.5 on a local classifier', () => {
      const local = (predicate: Record<string, number>) => errors(classifier({ name: 'local_intent', predicate }))
      expect(local({ gte: 0.5 })).toEqual([])
      expect(local({ gte: 0.4 })).toEqual(['local classifier condition "local_intent" supports only predicate.gte >= 0.5'])
      expect(local({ gt: 0.6 })).toEqual(['local classifier condition "local_intent" supports only predicate.gte >= 0.5'])
      expect(local({ gte: 0.6, lte: 0.9 })).toEqual([
        'local classifier condition "local_intent" supports only predicate.gte >= 0.5',
      ])
    })
  })

  describe('decision leaves (validator_decision_signal.go)', () => {
    it('requires a declared question', () => {
      expect(errors({ type: 'decision', name: 'missing', on_error: 'match' }, loose)).toEqual([
        'decision condition references unknown signal "missing"',
      ])
    })

    it('forbids a label on score and noul questions', () => {
      expect(errors({ type: 'decision', name: 'difficulty', on_error: 'match' })).toEqual([])
      expect(errors({ type: 'decision', name: 'difficulty', label: '1', on_error: 'match' })).toEqual([
        'decision condition "difficulty" takes no label',
      ])
      expect(errors({ type: 'decision', name: 'is_code', label: 'true', on_error: 'match' })).toEqual([
        'decision condition "is_code" takes no label',
      ])
    })

    it('requires a declared option key on choice, set and span questions', () => {
      const message = 'decision condition "route" requires a declared choice key or label as its label'
      expect(errors({ type: 'decision', name: 'route', on_error: 'match' })).toEqual([message])
      expect(errors({ type: 'decision', name: 'route', label: 'slow', on_error: 'match' })).toEqual([message])
    })
  })

  describe('complexity leaves (validator_complexity.go)', () => {
    it('requires a <rule>:<level> name with a known level', () => {
      expect(errors({ type: 'complexity', name: 'reasoning' })[0]).toMatch(/must name a difficulty level/)
      expect(errors({ type: 'complexity', name: 'reasoning:extreme' })[0]).toMatch(/unsupported difficulty "extreme"/)
      expect(errors({ type: 'complexity', name: 'reasoning:medium' })).toEqual([])
    })

    it('requires a declared rule even outside recipe scope', () => {
      expect(errors({ type: 'complexity', name: 'other:hard' }, loose)).toEqual([
        'complexity condition references unknown signal "other:hard"',
      ])
    })
  })

  describe('signal references (validator_signal_references.go)', () => {
    it('rejects an undeclared name in recipe scope and warns outside it', () => {
      const rules = { type: 'keyword', name: 'missing' }
      expect(errors(rules)).toEqual(['signal keyword("missing") is not declared in this recipe'])
      expect(errors(rules, loose)).toEqual([])
      expect(warnings(rules, loose)).toEqual(['signal keyword("missing") is not declared in this recipe'])
    })

    it('always rejects an undeclared metadata name', () => {
      expect(errors({ type: 'metadata', name: 'missing' }, loose)).toEqual([
        'metadata condition references unknown signal "missing"',
      ])
    })

    it('resolves projection outputs', () => {
      expect(errors({ type: 'projection', name: 'effort_high' })).toEqual([])
      expect(errors({ type: 'projection', name: 'effort_low' })).toEqual([
        'signal projection("effort_low") is not declared in this recipe',
      ])
    })
  })

  describe('unknown-outcome warning (validator_decision_unknown.go)', () => {
    it('warns about a classifier or decision leaf without on_error when on_unknown is unset', () => {
      const unguarded = classifier({ on_error: undefined })
      expect(errors(unguarded)).toEqual([])
      expect(warnings(unguarded)).toEqual([
        'condition "intent" evaluates as unknown when its model backend fails; set rules.on_unknown or the condition\'s on_error',
      ])
      expect(warnings({ ...unguarded, on_unknown: 'no_match' })).toEqual([])
    })

    it('skips the warning when on_unknown is edited outside the document', () => {
      // The Builder ROUTE header owns on_unknown, so the canvas cannot judge it.
      const issues = validateRuleDocument(canonicalToRuleDocument(classifier({ on_error: undefined })), strict, {
        onUnknownManagedElsewhere: true,
      })
      expect(issues).toEqual([])
    })
  })
})
