/**
 * Decision-rule AST bridge.
 *
 * The canonical config rule tree (`routing.decisions[].rules`) is the source of
 * truth. The visual tree defined here is a lossless, id-carrying mirror of it:
 * every canonical key either maps to a typed field or is kept in `extra`, so a
 * canonical -> tree -> canonical round trip returns the same document. DSL text
 * is a derived view (see decisionRuleDsl.ts); canonical rules never round-trip
 * through it, because the Go DSL flattens same-operator groups, unwraps
 * one-child groups and wraps single-leaf roots.
 */

import { ROUTER_CONFIG_SCHEMA } from '@/generated/routerConfigContract'

import type {
  DecisionCondition,
  DecisionRuleSet,
  NumericPredicate,
} from '../pages/configPageSupport'

export type { NumericPredicate }

export type RuleOperator = 'AND' | 'OR' | 'NOT'
export type RuleOnError = 'match' | 'no_match'
export type RuleOnUnknown = 'no_match' | 'match' | 'fail_request'

// Option lists come from the generated Router config schema, not a copy here.
const RULE_NODE_PROPERTIES = ROUTER_CONFIG_SCHEMA.$defs.RuleNode.properties
export const RULE_ON_ERROR_VALUES = RULE_NODE_PROPERTIES.on_error.enum as RuleOnError[]
export const RULE_ON_UNKNOWN_VALUES = RULE_NODE_PROPERTIES.on_unknown.enum as RuleOnUnknown[]
export const RULE_PREDICATE_KEYS = Object.keys(
  ROUTER_CONFIG_SCHEMA.$defs.NumericPredicate.properties,
) as Array<keyof NumericPredicate>

export interface RuleLeaf {
  id?: string
  signalType: string
  signalName: string
  label?: string
  predicate?: NumericPredicate
  onError?: RuleOnError
  /** Canonical keys the editor does not model; kept so nothing is lost. */
  extra?: Record<string, unknown>
}

export interface RuleGroup<Op extends RuleOperator = RuleOperator> {
  id?: string
  operator: Op
  conditions: RuleNode[]
  extra?: Record<string, unknown>
  /** How the canonical node spelled its combination, when not canonically. */
  written?: GroupSpelling
}

/**
 * Go normalizes `and`, ` Or ` and an omitted operator, and treats a missing
 * `conditions` key as empty. The original spelling is written back while the
 * group keeps that meaning, so an edit elsewhere does not rewrite it.
 */
export interface GroupSpelling {
  /** The operator as written; undefined when the key was omitted. */
  operator?: string
  /** False when the node had no `conditions` key. */
  hasConditions: boolean
}

export type RuleNode =
  | RuleLeaf
  | RuleGroup<'AND' | 'OR'>
  | (RuleGroup<'NOT'> & { conditions: [RuleNode] })

export type RuleGroupNode = Exclude<RuleNode, RuleLeaf>

export type SignalDescriptor = {
  signalType: string
  name: string
  /** Classifier labels or decision-question option keys, offered as suggestions. */
  labels?: string[]
}

/** A finding the editor reports to its host, anchored to a tree node. */
export interface RuleIssue {
  nodeId?: string
  path: number[]
  severity: 'error' | 'warning'
  message: string
}

export type CanonicalRule = DecisionRuleSet
export type RootShape = 'empty' | 'catch_all' | 'leaf' | 'group'

export interface DecisionRuleDocument {
  /** null <=> an empty (`{}`) root. A catch-all root keeps its empty AND. */
  tree: RuleNode | null
  rootShape: RootShape
  onUnknown?: RuleOnUnknown
  /** Unknown keys on an empty root; a non-empty root keeps them on `tree.extra`. */
  rootExtra?: Record<string, unknown>
}

const LEAF_KEYS = new Set(['type', 'name', 'label', 'predicate', 'on_error'])
const GROUP_KEYS = new Set(['operator', 'conditions'])

let ruleNodeCounter = 0

export function newRuleNodeId(): string {
  ruleNodeCounter += 1
  return `r${ruleNodeCounter}`
}

export function isRuleLeaf(node: RuleNode): node is RuleLeaf {
  return 'signalType' in node
}

export function isRuleGroup(node: RuleNode): node is RuleGroupNode {
  return 'operator' in node
}

/** Fills a missing id on every node; existing ids are left unchanged. */
export function assignRuleIds(tree: RuleNode): RuleNode
export function assignRuleIds(tree: RuleNode | null): RuleNode | null
export function assignRuleIds(tree: RuleNode | null): RuleNode | null {
  if (!tree) return null
  const id = tree.id ?? newRuleNodeId()
  if (isRuleLeaf(tree)) return tree.id ? tree : { ...tree, id }
  const conditions = tree.conditions.map((child) => assignRuleIds(child))
  const unchanged = tree.id && conditions.every((child, index) => child === tree.conditions[index])
  return unchanged ? tree : ({ ...tree, id, conditions } as RuleNode)
}

export function deriveRootShape(tree: RuleNode | null): RootShape {
  if (!tree) return 'empty'
  if (isRuleLeaf(tree)) return 'leaf'
  if (tree.operator === 'AND' && tree.conditions.length === 0) return 'catch_all'
  return 'group'
}

/** Readable predicate such as "gte 0.7, lt 0.9" for summaries and badges. */
export function describePredicate(predicate: NumericPredicate | undefined): string {
  if (!predicate) return ''
  const raw = predicate as Record<string, unknown>
  return Object.keys(raw)
    .filter((key) => raw[key] !== undefined)
    .sort((a, b) => sortIndex(a) - sortIndex(b))
    .map((key) => `${key} ${String(raw[key])}`)
    .join(', ')
}

function sortIndex(key: string): number {
  const index = (RULE_PREDICATE_KEYS as readonly string[]).indexOf(key)
  return index === -1 ? RULE_PREDICATE_KEYS.length : index
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasOwn(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key)
}

function nonEmptyRecord(record: Record<string, unknown>): Record<string, unknown> | undefined {
  return Object.keys(record).length > 0 ? record : undefined
}

function normalizeOperator(value: unknown): string {
  return typeof value === 'string' ? value.trim().toUpperCase() : ''
}

// A typeless node that still carries leaf fields is a malformed leaf, not an
// empty combination: Go rejects it with "leaf condition requires a type", and
// reading it as unconditional would silently widen the decision.
function isLeafShaped(raw: Record<string, unknown>): boolean {
  if (raw.type !== undefined && raw.type !== null && raw.type !== '') return true
  return ['name', 'label', 'predicate', 'on_error'].some(
    (key) => raw[key] !== undefined && raw[key] !== '',
  )
}

function canonicalToLeaf(raw: Record<string, unknown>, root: boolean): RuleLeaf {
  const extra: Record<string, unknown> = {}
  const leaf: RuleLeaf = { id: newRuleNodeId(), signalType: '', signalName: '' }
  if (typeof raw.type === 'string') leaf.signalType = raw.type
  else if (raw.type !== undefined) extra.type = raw.type
  if (typeof raw.name === 'string') leaf.signalName = raw.name
  else if (raw.name !== undefined) extra.name = raw.name
  if (typeof raw.label === 'string') leaf.label = raw.label
  else if (raw.label !== undefined) extra.label = raw.label
  if (isPlainObject(raw.predicate)) leaf.predicate = { ...raw.predicate } as NumericPredicate
  else if (raw.predicate !== undefined) extra.predicate = raw.predicate
  if (typeof raw.on_error === 'string') leaf.onError = raw.on_error as RuleOnError
  else if (raw.on_error !== undefined) extra.on_error = raw.on_error
  for (const [key, value] of Object.entries(raw)) {
    if (LEAF_KEYS.has(key) || value === undefined) continue
    if (root && key === 'on_unknown') continue
    extra[key] = value
  }
  const kept = nonEmptyRecord(extra)
  if (kept) leaf.extra = kept
  return leaf
}

// Go decodes a YAML null on a rule field to its zero value, which means unset.
const NULLABLE_KEYS = new Set([...LEAF_KEYS, ...GROUP_KEYS, 'on_unknown'])
function withoutNullFields(raw: Record<string, unknown>): Record<string, unknown> {
  if (!Object.entries(raw).some(([key, value]) => value === null && NULLABLE_KEYS.has(key))) return raw
  return Object.fromEntries(Object.entries(raw).filter(([key, value]) => !(value === null && NULLABLE_KEYS.has(key))))
}

function canonicalToNode(input: unknown, root: boolean): RuleNode {
  const raw = isPlainObject(input) ? withoutNullFields(input) : input
  if (!isPlainObject(raw)) {
    // A non-object child cannot be modelled; keep it verbatim so the Router
    // rejects it on save instead of the save dropping it.
    return { id: newRuleNodeId(), signalType: '', signalName: '', extra: { value: raw } }
  }
  if (isLeafShaped(raw)) return canonicalToLeaf(raw, root)

  const extra: Record<string, unknown> = {}
  const rawConditions = raw.conditions
  const conditions = Array.isArray(rawConditions)
    ? rawConditions.map((child) => canonicalToNode(child, false))
    : []
  if (rawConditions !== undefined && !Array.isArray(rawConditions)) extra.conditions = rawConditions
  if (raw.operator !== undefined && typeof raw.operator !== 'string') extra.operator = raw.operator
  // Go trims and upper-cases the operator and defaults an omitted one to AND.
  // An unsupported operator is kept verbatim so the Router can name it.
  const operator = (normalizeOperator(raw.operator) || 'AND') as RuleOperator
  for (const [key, value] of Object.entries(raw)) {
    if (GROUP_KEYS.has(key) || LEAF_KEYS.has(key) || value === undefined) continue
    if (root && key === 'on_unknown') continue
    extra[key] = value
  }
  const group = { id: newRuleNodeId(), operator, conditions } as RuleGroupNode
  const writtenOperator = typeof raw.operator === 'string' ? raw.operator : undefined
  if (writtenOperator !== operator || rawConditions === undefined) {
    group.written = { operator: writtenOperator, hasConditions: rawConditions !== undefined }
  }
  const kept = nonEmptyRecord(extra)
  if (kept) group.extra = kept
  return group
}

function isEmptyRoot(raw: Record<string, unknown>): boolean {
  if (isLeafShaped(raw)) return false
  if (normalizeOperator(raw.operator) !== '') return false
  return raw.conditions === undefined
}

export function canonicalToRuleDocument(rules: CanonicalRule | undefined | null): DecisionRuleDocument {
  const raw: Record<string, unknown> = isPlainObject(rules) ? withoutNullFields(rules) : {}
  const onUnknown = typeof raw.on_unknown === 'string' ? (raw.on_unknown as RuleOnUnknown) : undefined
  const unknownExtra = raw.on_unknown !== undefined && typeof raw.on_unknown !== 'string'
    ? { on_unknown: raw.on_unknown }
    : undefined

  if (isEmptyRoot(raw)) {
    const rootExtra: Record<string, unknown> = { ...unknownExtra }
    for (const [key, value] of Object.entries(raw)) {
      if (key === 'on_unknown' || (key === 'operator' && typeof value === 'string') || value === undefined) continue
      rootExtra[key] = value
    }
    return withOptional({ tree: null, rootShape: 'empty' }, onUnknown, nonEmptyRecord(rootExtra))
  }

  const tree = canonicalToNode(raw, true)
  if (unknownExtra) tree.extra = { ...tree.extra, ...unknownExtra }
  return withOptional({ tree, rootShape: deriveRootShape(tree) }, onUnknown, undefined)
}

function withOptional(
  doc: DecisionRuleDocument,
  onUnknown: RuleOnUnknown | undefined,
  rootExtra: Record<string, unknown> | undefined,
): DecisionRuleDocument {
  if (onUnknown !== undefined) doc.onUnknown = onUnknown
  if (rootExtra) doc.rootExtra = rootExtra
  return doc
}

function predicateToCanonical(predicate: NumericPredicate): NumericPredicate {
  const out: Record<string, unknown> = {}
  for (const key of RULE_PREDICATE_KEYS) {
    if (typeof predicate[key] === 'number') out[key] = predicate[key]
  }
  // Non-numeric or unknown predicate keys are invalid, but are written back
  // so the Router reports them instead of the save dropping them.
  for (const [key, value] of Object.entries(predicate)) {
    if (value === undefined || hasOwn(out, key)) continue
    out[key] = value
  }
  return out as NumericPredicate
}

function nodeToCanonical(node: RuleNode): DecisionCondition {
  const out: Record<string, unknown> = {}
  if (isRuleLeaf(node)) {
    if (node.signalType !== '') out.type = node.signalType
    if (node.signalName !== '') out.name = node.signalName
    if (node.label !== undefined) out.label = node.label
    if (node.predicate !== undefined) out.predicate = predicateToCanonical(node.predicate)
    if (node.onError !== undefined) out.on_error = node.onError
  } else {
    const written = node.written
    const sameOperator = written !== undefined && (normalizeOperator(written.operator) || 'AND') === node.operator
    if (!sameOperator) out.operator = node.operator
    else if (written.operator !== undefined) out.operator = written.operator
    if (!written || written.hasConditions || node.conditions.length > 0) {
      out.conditions = node.conditions.map(nodeToCanonical)
    }
  }
  for (const [key, value] of Object.entries(node.extra ?? {})) {
    if (value !== undefined && !hasOwn(out, key)) out[key] = value
  }
  return out as DecisionCondition
}

export function ruleDocumentToCanonical(doc: DecisionRuleDocument): CanonicalRule {
  const out: Record<string, unknown> = doc.tree ? { ...nodeToCanonical(doc.tree) } : {}
  if (doc.onUnknown !== undefined) out.on_unknown = doc.onUnknown
  // Unknown keys read from an empty root stay on the root after the first
  // condition is added, so the Router keeps reporting them instead of a save
  // dropping them.
  for (const [key, value] of Object.entries(doc.rootExtra ?? {})) {
    if (value !== undefined && !hasOwn(out, key)) out[key] = value
  }
  return out as CanonicalRule
}

function valuesEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a === 'number' && typeof b === 'number') return Number.isNaN(a) && Number.isNaN(b)
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false
    return a.every((value, index) => valuesEqual(value, b[index]))
  }
  if (!isPlainObject(a) || !isPlainObject(b)) return false
  const aKeys = Object.keys(a).filter((key) => a[key] !== undefined)
  const bKeys = Object.keys(b).filter((key) => b[key] !== undefined)
  if (aKeys.length !== bKeys.length) return false
  return aKeys.every((key) => valuesEqual(a[key], b[key]))
}

/** Structural equality that ignores node ids. */
export function ruleSubtreeEquals(a: RuleNode | null | undefined, b: RuleNode | null | undefined): boolean {
  if (!a || !b) return !a && !b
  if (!valuesEqual(a.extra ?? {}, b.extra ?? {})) return false
  if (isRuleLeaf(a) || isRuleLeaf(b)) {
    if (!isRuleLeaf(a) || !isRuleLeaf(b)) return false
    return a.signalType === b.signalType &&
      a.signalName === b.signalName &&
      a.label === b.label &&
      a.onError === b.onError &&
      valuesEqual(a.predicate, b.predicate)
  }
  if (a.operator !== b.operator || a.conditions.length !== b.conditions.length) return false
  return a.conditions.every((child, index) => ruleSubtreeEquals(child, b.conditions[index]))
}

/**
 * The semantic normal form a tree takes after a DSL round trip: same-operator
 * groups are flattened, one-child AND/OR groups are unwrapped and childless
 * AND/OR groups disappear. It decides whether raw DSL text changed meaning; it
 * is never written back to the canonical config.
 */
export function normalizeForDsl(tree: RuleNode | null): RuleNode | null {
  if (!tree) return null
  if (isRuleLeaf(tree)) return tree
  if (tree.operator === 'NOT') {
    const conditions = tree.conditions
      .map((child) => normalizeForDsl(child))
      .filter((child): child is RuleNode => child !== null)
    return { ...tree, conditions } as RuleNode
  }
  const conditions: RuleNode[] = []
  for (const child of tree.conditions) {
    const normalized = normalizeForDsl(child)
    if (!normalized) continue
    if (isRuleGroup(normalized) && normalized.operator === tree.operator && !normalized.extra) {
      conditions.push(...normalized.conditions)
    } else {
      conditions.push(normalized)
    }
  }
  if (tree.extra) return { ...tree, conditions } as RuleNode
  if (conditions.length === 0) return null
  if (conditions.length === 1) return conditions[0]
  return { ...tree, conditions } as RuleNode
}
