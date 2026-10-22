import {
  describePredicate,
  isRuleGroup,
  isRuleLeaf,
  newRuleNodeId,
  type NumericPredicate,
  type RuleLeaf,
  type RuleNode,
} from '@/lib/decisionRuleAst'

// The rule tree model lives in lib/decisionRuleAst.ts so the canonical bridge,
// the DSL module and the validators share it without depending on components.
export type {
  NumericPredicate,
  RuleGroup,
  RuleIssue,
  RuleLeaf,
  RuleNode,
  SignalDescriptor,
} from '@/lib/decisionRuleAst'
export { assignRuleIds, describePredicate, newRuleNodeId } from '@/lib/decisionRuleAst'
export { boolExprToRuleNode } from '@/lib/decisionRuleDsl'

export type NodePath = number[]

export interface DragDataSignal {
  kind: 'signal'
  signalType: string
  signalName: string
}

export interface DragDataOperator {
  kind: 'operator'
  operator: 'AND' | 'OR' | 'NOT'
}

export interface DragDataTreeNode {
  kind: 'tree-node'
  path: NodePath
}

export type DragData = DragDataSignal | DragDataOperator | DragDataTreeNode

export const DRAG_MIME = 'application/x-expr-builder'

export function isLeaf(n: RuleNode): n is RuleLeaf {
  return isRuleLeaf(n)
}

export function isOperator(n: RuleNode): n is Exclude<RuleNode, RuleLeaf> {
  return isRuleGroup(n)
}

/** Compact label / predicate / on_error badges for a leaf node. */
export function leafMetaBadges(leaf: RuleLeaf): string[] {
  const badges: string[] = []
  if (leaf.label) badges.push(leaf.label)
  const predicate = describePredicate(leaf.predicate)
  if (predicate) badges.push(predicate)
  if (leaf.onError) badges.push(`on_error ${leaf.onError}`)
  return badges
}

export function pathEq(a: NodePath, b: NodePath): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index])
}

export function pathStartsWith(path: NodePath, prefix: NodePath): boolean {
  if (prefix.length > path.length) return false
  return prefix.every((value, index) => value === path[index])
}

export function getNodeAtPath(root: RuleNode, path: NodePath): RuleNode | null {
  if (path.length === 0) return root
  if (!isOperator(root)) return null
  const [head, ...tail] = path
  if (head < 0 || head >= root.conditions.length) return null
  return getNodeAtPath(root.conditions[head], tail)
}

export function replaceAtPath(root: RuleNode, path: NodePath, replacement: RuleNode): RuleNode {
  if (path.length === 0) return replacement
  if (!isOperator(root)) return root
  const [head, ...tail] = path
  const newChild = replaceAtPath(root.conditions[head], tail, replacement)
  const newConditions = root.conditions.map((child, index) => (index === head ? newChild : child))
  return { ...root, conditions: newConditions } as RuleNode
}

// Removing a child never collapses the parent: Go accepts a one-child AND/OR,
// so the explicit Unwrap action is the only collapse. A group left with no
// children is removed from its own parent.
export function removeAtPath(root: RuleNode, path: NodePath): RuleNode | null {
  if (path.length === 0) return null
  if (!isOperator(root)) return root
  const [head, ...tail] = path
  const newChild = tail.length === 0 ? null : removeAtPath(root.conditions[head], tail)
  const newConditions = newChild
    ? root.conditions.map((child, index) => (index === head ? newChild : child))
    : root.conditions.filter((_, index) => index !== head)
  if (newConditions.length === 0) return null
  return { ...root, conditions: newConditions } as RuleNode
}

export function insertAtPath(root: RuleNode, path: NodePath, insertIdx: number, node: RuleNode): RuleNode {
  if (path.length === 0) {
    if (isOperator(root)) {
      const conditions = [...root.conditions]
      conditions.splice(insertIdx, 0, node)
      return { ...root, conditions } as RuleNode
    }
    return { id: newRuleNodeId(), operator: 'AND', conditions: [root, node] }
  }
  if (!isOperator(root)) return root
  const [head, ...tail] = path
  const newChild = insertAtPath(root.conditions[head], tail, insertIdx, node)
  const newConditions = root.conditions.map((child, index) => (index === head ? newChild : child))
  return { ...root, conditions: newConditions } as RuleNode
}

export function addChildAtPath(root: RuleNode, path: NodePath, child: RuleNode): RuleNode {
  const target = getNodeAtPath(root, path)
  if (!target) return root
  if (isOperator(target)) {
    const newTarget = { ...target, conditions: [...target.conditions, child] } as RuleNode
    return replaceAtPath(root, path, newTarget)
  }
  const wrapped: RuleNode = { id: newRuleNodeId(), operator: 'AND', conditions: [target, child] }
  return replaceAtPath(root, path, wrapped)
}

export function encodeDrag(data: DragData): string {
  return JSON.stringify(data)
}

export function decodeDrag(raw: string): DragData | null {
  try {
    return JSON.parse(raw) as DragData
  } catch {
    return null
  }
}

export function makeDragNode(data: DragData): RuleNode | null {
  if (data.kind === 'signal') {
    return { id: newRuleNodeId(), signalType: data.signalType, signalName: data.signalName }
  }
  if (data.kind === 'operator') {
    return data.operator === 'NOT'
      ? { id: newRuleNodeId(), operator: 'NOT', conditions: [] as unknown as [RuleNode] }
      : { id: newRuleNodeId(), operator: data.operator, conditions: [] }
  }
  return null
}

// Leaf edit dialog: the gt/gte/lt/lte inputs as text drafts.
export const PREDICATE_FIELDS = ['gt', 'gte', 'lt', 'lte'] as const
export type PredicateDraft = Record<(typeof PREDICATE_FIELDS)[number], string>

export function predicateDraft(predicate: NumericPredicate | undefined): PredicateDraft {
  const draft = { gt: '', gte: '', lt: '', lte: '' }
  for (const key of PREDICATE_FIELDS) {
    const value = predicate?.[key]
    if (value !== undefined) draft[key] = String(value)
  }
  return draft
}

// Keys outside gt/gte/lt/lte have no input here; they are carried over so
// validation keeps reporting them instead of the dialog dropping them.
export function predicateFromDraft(
  draft: PredicateDraft,
  initial: NumericPredicate | undefined,
): { predicate?: NumericPredicate; invalid: string[] } {
  const predicate: NumericPredicate = {}
  const raw = (initial ?? {}) as Record<string, unknown>
  for (const key of Object.keys(raw)) {
    if (raw[key] !== undefined && !(PREDICATE_FIELDS as readonly string[]).includes(key)) {
      ;(predicate as Record<string, unknown>)[key] = raw[key]
    }
  }
  const invalid: string[] = []
  for (const key of PREDICATE_FIELDS) {
    const text = draft[key].trim()
    if (text === '') continue
    const value = Number(text)
    if (!Number.isFinite(value)) invalid.push(key)
    else predicate[key] = value
  }
  return { predicate: Object.keys(predicate).length > 0 ? predicate : undefined, invalid }
}
