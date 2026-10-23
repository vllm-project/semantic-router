/**
 * Decision-rule validation that mirrors the Go config validators:
 * validator_rule_operator.go (node shape and arity), validator_decision.go
 * (labels, on_error, classifier predicates), validator_decision_signal.go
 * (decision-model questions), validator_numeric_predicate.go,
 * validator_complexity.go, validator_signal_references.go and the
 * validator_decision_unknown.go warning. Errors block a save; warnings do not.
 */

import { DECISION_SIGNAL_TYPES } from '@/generated/routerConfigContract'

import {
  isRuleLeaf,
  type DecisionRuleDocument,
  type NumericPredicate,
  type RuleIssue,
  type RuleLeaf,
  type RuleNode,
  type SignalDescriptor,
} from './decisionRuleAst'

export type { RuleIssue }

export interface RuleSignalScope {
  descriptors: SignalDescriptor[]
  byKey: Map<string, SignalDescriptor>
  /**
   * Recipe-scoped validation: Go rejects an undeclared reference as
   * "not declared in this recipe". Without it such a reference is a warning.
   */
  strictNames: boolean
  /** Types whose referenced names the Router declares itself, so any name resolves. */
  implicitTypes: Set<string>
}

export function signalDescriptorKey(signalType: string, name: string): string {
  return `${signalType.trim().toLowerCase()}\u0000${name}`
}

export function ruleSignalScope(
  descriptors: SignalDescriptor[],
  strictNames: boolean,
  implicitTypes: Iterable<string> = [],
): RuleSignalScope {
  const byKey = new Map<string, SignalDescriptor>()
  for (const descriptor of descriptors) {
    byKey.set(signalDescriptorKey(descriptor.signalType, descriptor.name), descriptor)
  }
  return { descriptors, byKey, strictNames, implicitTypes: new Set(implicitTypes) }
}

/** Prefix of the blocking issue an editor reports for an unparsable raw expression. */
export const RAW_EXPRESSION_INCOMPLETE = 'Raw expression is incomplete'

/** The message of an editor's unparsable raw-expression issue, if it reported one. */
export function incompleteRawExpression(issues: RuleIssue[]): string | null {
  return issues.find((issue) => issue.message.startsWith(RAW_EXPRESSION_INCOMPLETE))?.message ?? null
}


export const ON_UNKNOWN_ON_ERROR_CONFLICT_MESSAGE =
  'condition on_error has no effect when rules.on_unknown is set; remove one of them'

const UNKNOWN_POLICIES = ['no_match', 'match', 'fail_request']
const ON_ERROR_VALUES = ['no_match', 'match']
const COMPLEXITY_LEVELS = ['easy', 'hard', 'medium']
const PREDICATE_KEYS = ['gt', 'gte', 'lt', 'lte']
const DECISION_TYPES = new Set<string>(DECISION_SIGNAL_TYPES)
const ALWAYS_DECLARED_TYPES = new Set(['classifier', 'decision', 'metadata', 'complexity'])

interface ValidationCtx {
  scope: RuleSignalScope
  issues: RuleIssue[]
  onUnknownSet: boolean
  unknownPolicyWarnings: boolean
}

export interface ValidateRuleDocumentOptions {
  /**
   * The rules' on_unknown is edited outside this document (the Builder ROUTE
   * header), so the missing-failure-policy warning cannot be judged here.
   */
  onUnknownManagedElsewhere?: boolean
}

function push(ctx: ValidationCtx, node: RuleNode, path: number[], severity: RuleIssue['severity'], message: string) {
  ctx.issues.push({ nodeId: node.id, path, severity, message })
}

function describeLeaf(leaf: RuleLeaf): string {
  return `${leaf.signalType || '?'}(${JSON.stringify(leaf.signalName)})`
}

function isLabelled(signalType: string): boolean {
  const type = signalType.toLowerCase()
  return type === 'classifier' || type === 'decision'
}

export function numericPredicateError(predicate: NumericPredicate): string | null {
  const raw = predicate as Record<string, unknown>
  for (const key of Object.keys(raw)) {
    if (raw[key] === undefined) continue
    if (!PREDICATE_KEYS.includes(key)) return `predicate field ${key} is not supported (expected gt, gte, lt, lte)`
    if (typeof raw[key] !== 'number') return `predicate ${key} must be a number`
  }
  const values = PREDICATE_KEYS.map((key) => raw[key]).filter((value): value is number => typeof value === 'number')
  if (values.some((value) => !Number.isFinite(value))) return 'predicate values must be finite'
  if (values.length === 0) return 'predicate must set at least one comparator'
  const { gt, gte, lt, lte } = predicate
  if (gt !== undefined && gte !== undefined) return 'predicate cannot set both gt and gte'
  if (lt !== undefined && lte !== undefined) return 'predicate cannot set both lt and lte'
  const lower = gt ?? gte
  const upper = lt ?? lte
  if (lower !== undefined && upper !== undefined) {
    if (lower > upper || (lower === upper && (gt !== undefined || lt !== undefined))) {
      return 'predicate defines an empty numeric range'
    }
  }
  return null
}

// Canonical input may carry any on_error string, including an empty one.
function hasOnError(leaf: RuleLeaf): boolean {
  const onError = leaf.onError as string | undefined
  return onError !== undefined && onError !== ''
}

function treeSetsOnError(node: RuleNode): boolean {
  if (isRuleLeaf(node)) return hasOnError(node)
  return node.conditions.some(treeSetsOnError)
}

function validateExtra(ctx: ValidationCtx, node: RuleNode, path: number[], root: boolean) {
  for (const key of Object.keys(node.extra ?? {})) {
    if (isRuleLeaf(node) && (key === 'operator' || key === 'conditions')) continue
    if (key === 'on_unknown' && !root) {
      push(ctx, node, path, 'error', 'on_unknown is only supported on the root rules node')
    } else if (['type', 'name', 'label', 'predicate', 'on_error', 'operator', 'conditions', 'on_unknown'].includes(key)) {
      push(ctx, node, path, 'error', `field ${key} has an invalid value`)
    } else {
      push(ctx, node, path, 'error', `unsupported field ${key}`)
    }
  }
}

function validateGroup(ctx: ValidationCtx, node: Exclude<RuleNode, RuleLeaf>, path: number[], root: boolean) {
  const operator = node.operator as string
  if (operator !== 'AND' && operator !== 'OR' && operator !== 'NOT') {
    push(ctx, node, path, 'error', `unsupported operator "${operator}" (expected AND, OR, NOT)`)
  } else if (operator === 'NOT' && node.conditions.length !== 1) {
    push(ctx, node, path, 'error', `NOT requires exactly one child condition, got ${node.conditions.length}`)
  } else if (node.conditions.length === 0 && !(root && operator === 'AND')) {
    // Only a root AND matches on zero children; a childless OR never matches.
    push(ctx, node, path, 'error', `${operator} combination requires at least one child condition`)
  }
  node.conditions.forEach((child, index) => validateNode(ctx, child, [...path, index], false))
}

function referenceIssue(ctx: ValidationCtx, leaf: RuleLeaf, path: number[], message: string) {
  push(ctx, leaf, path, ctx.scope.strictNames ? 'error' : 'warning', message)
}

function validateClassifierLeaf(ctx: ValidationCtx, leaf: RuleLeaf, path: number[], descriptor: SignalDescriptor) {
  const name = JSON.stringify(leaf.signalName)
  // Builder descriptors carry no labels; the label contract is only checkable
  // when the scope describes the classifier.
  if (descriptor.labels === undefined) return
  if (!leaf.label || !descriptor.labels.includes(leaf.label)) {
    push(ctx, leaf, path, 'error', `classifier condition ${name} requires a declared label`)
  }
  if (descriptor.predicateRequired === false) return
  if (leaf.predicate === undefined) {
    push(ctx, leaf, path, 'error', `classifier condition ${name} requires a score predicate`)
    return
  }
  if (descriptor.localClassifier) {
    const { gt, gte, lt, lte } = leaf.predicate
    if (gte === undefined || gte < 0.5 || gt !== undefined || lt !== undefined || lte !== undefined) {
      push(ctx, leaf, path, 'error', `local classifier condition ${name} supports only predicate.gte >= 0.5`)
    }
  }
}

function validateDecisionLeaf(ctx: ValidationCtx, leaf: RuleLeaf, path: number[], descriptor: SignalDescriptor) {
  const name = JSON.stringify(leaf.signalName)
  if (descriptor.labelForbidden) {
    if (leaf.label) push(ctx, leaf, path, 'error', `decision condition ${name} takes no label`)
    return
  }
  if (descriptor.labels === undefined) return
  if (!leaf.label || !descriptor.labels.includes(leaf.label)) {
    push(ctx, leaf, path, 'error', `decision condition ${name} requires a declared choice key or label as its label`)
  }
}

function validateComplexityName(ctx: ValidationCtx, leaf: RuleLeaf, path: number[]) {
  const name = leaf.signalName
  const index = name.lastIndexOf(':')
  if (index <= 0 || index === name.length - 1) {
    push(ctx, leaf, path, 'error',
      `complexity ${JSON.stringify(name)} must name a difficulty level as "<rule>:<${COMPLEXITY_LEVELS.join('|')}>"`)
    return false
  }
  const difficulty = name.slice(index + 1)
  if (!COMPLEXITY_LEVELS.includes(difficulty)) {
    push(ctx, leaf, path, 'error',
      `complexity ${JSON.stringify(name)} has unsupported difficulty ${JSON.stringify(difficulty)}; valid levels: ${COMPLEXITY_LEVELS.join(', ')}`)
    return false
  }
  return true
}

function validateLeaf(ctx: ValidationCtx, leaf: RuleLeaf, path: number[]) {
  const extra = leaf.extra ?? {}
  if (!leaf.signalType) {
    push(ctx, leaf, path, 'error',
      'leaf condition requires a type (name, label, predicate, and on_error only apply to leaf conditions)')
    return
  }
  // Go only sees a combination half when the operator is non-empty or there
  // are children; `operator: ""` and `conditions: []` on a leaf are inert.
  const hasOperator = extra.operator !== undefined && extra.operator !== ''
  const hasConditions = extra.conditions !== undefined &&
    !(Array.isArray(extra.conditions) && extra.conditions.length === 0)
  if (hasOperator || hasConditions) {
    push(ctx, leaf, path, 'error',
      'condition must be either a leaf (type/name) or a combination (operator/conditions), not both')
  }
  const type = leaf.signalType.trim().toLowerCase()
  if (!DECISION_TYPES.has(type)) {
    push(ctx, leaf, path, 'error', `unsupported signal type ${JSON.stringify(leaf.signalType)}`)
    return
  }
  const labelled = isLabelled(type)
  const described = describeLeaf(leaf)
  if (leaf.label && !labelled) {
    push(ctx, leaf, path, 'error', 'label is only supported on classifier and decision conditions')
  }
  if (leaf.onError !== undefined && hasOnError(leaf)) {
    if (!ON_ERROR_VALUES.includes(leaf.onError)) {
      push(ctx, leaf, path, 'error', `condition ${described} on_error must be no_match or match`)
    } else if (!labelled) {
      push(ctx, leaf, path, 'error',
        `condition ${described} on_error is only supported for classifier and decision conditions`)
    }
  }
  if (leaf.predicate !== undefined) {
    const error = numericPredicateError(leaf.predicate)
    if (error) push(ctx, leaf, path, 'error', `condition ${described} ${error}`)
  }

  if (leaf.signalName.trim() === '') {
    push(ctx, leaf, path, 'error', `signal condition of type ${JSON.stringify(type)} requires a name`)
    return
  }
  if (leaf.signalName.trim() !== leaf.signalName) {
    referenceIssue(ctx, leaf, path,
      `signal ${type} reference ${JSON.stringify(leaf.signalName)} has surrounding whitespace`)
  }
  if (type === 'complexity' && !validateComplexityName(ctx, leaf, path)) return

  const descriptor = ctx.scope.byKey.get(signalDescriptorKey(type, leaf.signalName))
  if (!descriptor) {
    const message = `signal ${described} is not declared in this recipe`
    // Classifier, decision, metadata and complexity references are checked by
    // Go even outside recipe scope, so those stay blocking.
    if (ALWAYS_DECLARED_TYPES.has(type)) {
      push(ctx, leaf, path, 'error', `${type} condition references unknown signal ${JSON.stringify(leaf.signalName)}`)
    } else if (!ctx.scope.implicitTypes.has(type)) {
      referenceIssue(ctx, leaf, path, message)
    }
  } else if (type === 'classifier') {
    validateClassifierLeaf(ctx, leaf, path, descriptor)
  } else if (type === 'decision') {
    validateDecisionLeaf(ctx, leaf, path, descriptor)
  }

  if (labelled && ctx.unknownPolicyWarnings && !ctx.onUnknownSet && !hasOnError(leaf)) {
    push(ctx, leaf, path, 'warning',
      `condition ${JSON.stringify(leaf.signalName)} evaluates as unknown when its model backend fails; set rules.on_unknown or the condition's on_error`)
  }
}

function validateNode(ctx: ValidationCtx, node: RuleNode, path: number[], root: boolean) {
  validateExtra(ctx, node, path, root)
  if (isRuleLeaf(node)) validateLeaf(ctx, node, path)
  else validateGroup(ctx, node, path, root)
}

export function validateRuleDocument(
  doc: DecisionRuleDocument,
  scope: RuleSignalScope,
  opts: ValidateRuleDocumentOptions = {},
): RuleIssue[] {
  // Go treats an empty on_unknown as unset (`node.OnUnknown != ""`).
  const onUnknown = doc.onUnknown === undefined || (doc.onUnknown as string) === '' ? undefined : doc.onUnknown
  const ctx: ValidationCtx = {
    scope,
    issues: [],
    onUnknownSet: onUnknown !== undefined,
    unknownPolicyWarnings: !opts.onUnknownManagedElsewhere,
  }
  const rootAnchor = { path: [], nodeId: doc.tree?.id }
  if (onUnknown !== undefined) {
    if (!UNKNOWN_POLICIES.includes(onUnknown)) {
      ctx.issues.push({ ...rootAnchor, severity: 'error', message: 'rules on_unknown must be no_match, match, or fail_request' })
    }
    if (doc.tree && treeSetsOnError(doc.tree)) {
      ctx.issues.push({ ...rootAnchor, severity: 'error', message: ON_UNKNOWN_ON_ERROR_CONFLICT_MESSAGE })
    }
  }
  for (const key of Object.keys(doc.rootExtra ?? {})) {
    const message = key === 'operator' ? 'field operator has an invalid value' : `unsupported field ${key}`
    ctx.issues.push({ path: [], severity: 'error', message })
  }
  if (doc.tree) validateNode(ctx, doc.tree, [], true)
  return ctx.issues
}

export function blockingRuleIssues(issues: RuleIssue[]): RuleIssue[] {
  return issues.filter((issue) => issue.severity === 'error')
}

/** "rules.conditions[0].conditions[1]" style location, matching Go error paths. */
export function formatRuleIssuePath(path: number[]): string {
  return ['rules', ...path.map((index) => `conditions[${index}]`)].join('.')
}
