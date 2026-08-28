import {
  isLeaf,
  isOperator,
  parseExprText,
  serializeNode,
  type RuleNode,
  type SignalDescriptor,
} from '../components/ExpressionBuilderSupport'
import type { DecisionCondition, DecisionRuleSet, NumericPredicate } from './configPageSupport'

// Leaf-only fields DecisionCondition carries that ExpressionBuilder's RuleNode has no room for.
// Preserved out-of-band during a round trip so editing a rule's structure never silently drops
// classifier labels, numeric predicates, or on-error behavior for leaves the edit didn't touch.
export interface DecisionLeafMetadata {
  label?: string
  predicate?: NumericPredicate
  on_error?: 'no_match' | 'match'
}

// Leaves are matched between the pre-edit and post-edit tree by "type::name" identity, in
// traversal order (a FIFO queue per key). A leaf that keeps its type and name keeps its
// metadata; a leaf whose type or name changed is treated as a different leaf and starts fresh.
export type DecisionLeafMetadataIndex = Map<string, DecisionLeafMetadata[]>

function leafMetadataKey(type: string | undefined, name: string | undefined): string {
  return `${type ?? ''}::${name ?? ''}`
}

export function collectLeafMetadata(
  ruleSet: DecisionRuleSet | undefined,
): DecisionLeafMetadataIndex {
  const index: DecisionLeafMetadataIndex = new Map()
  const visit = (condition: DecisionCondition) => {
    if (condition.operator || condition.conditions?.length) {
      condition.conditions?.forEach(visit)
      return
    }
    const { label, predicate, on_error } = condition
    if (!label && !predicate && !on_error) return
    const key = leafMetadataKey(condition.type, condition.name)
    const queue = index.get(key) ?? []
    queue.push({ label, predicate, on_error })
    index.set(key, queue)
  }
  ruleSet?.conditions?.forEach(visit)
  return index
}

function conditionToRuleNode(condition: DecisionCondition): RuleNode {
  if (condition.operator === 'NOT') {
    const [child] = condition.conditions ?? []
    return {
      operator: 'NOT',
      conditions: [child ? conditionToRuleNode(child) : { signalType: '', signalName: '' }],
    }
  }
  if (condition.operator === 'AND' || condition.operator === 'OR') {
    return {
      operator: condition.operator,
      conditions: (condition.conditions ?? []).map(conditionToRuleNode),
    }
  }
  return { signalType: condition.type ?? '', signalName: condition.name ?? '' }
}

// Returns null for an empty rule set (no conditions authored yet) rather than a synthetic
// operator node, so the editor renders its empty-canvas state instead of a dangling "(? AND ?)".
export function decisionRuleSetToRuleNode(ruleSet: DecisionRuleSet | undefined): RuleNode | null {
  if (!ruleSet?.conditions?.length) return null
  return conditionToRuleNode({ operator: ruleSet.operator, conditions: ruleSet.conditions })
}

export function decisionRuleSetToExprText(ruleSet: DecisionRuleSet | undefined): string {
  const node = decisionRuleSetToRuleNode(ruleSet)
  return node ? serializeNode(node) : ''
}

function ruleNodeToCondition(
  node: RuleNode,
  metadata: DecisionLeafMetadataIndex,
): DecisionCondition {
  if (isLeaf(node)) {
    const key = leafMetadataKey(node.signalType, node.signalName)
    const queue = metadata.get(key)
    const preserved = queue?.shift()
    return {
      type: node.signalType,
      name: node.signalName,
      ...(preserved?.label ? { label: preserved.label } : {}),
      ...(preserved?.predicate ? { predicate: preserved.predicate } : {}),
      ...(preserved?.on_error ? { on_error: preserved.on_error } : {}),
    }
  }
  return {
    operator: node.operator,
    conditions: node.conditions.map((child) => ruleNodeToCondition(child, metadata)),
  }
}

// A tree collapsed down to a single surviving condition (e.g. after removing siblings) loses
// its wrapping AND/OR in RuleNode form. Re-wrap it so the result is always a valid DecisionRuleSet.
export function ruleNodeToDecisionRuleSet(
  node: RuleNode | null,
  metadata: DecisionLeafMetadataIndex = new Map(),
): DecisionRuleSet {
  if (!node) return { operator: 'AND', conditions: [] }
  if (isOperator(node)) {
    const condition = ruleNodeToCondition(node, metadata)
    return { operator: condition.operator ?? 'AND', conditions: condition.conditions ?? [] }
  }
  return { operator: 'AND', conditions: [ruleNodeToCondition(node, metadata)] }
}

export interface ExprTextParseResult {
  rules: DecisionRuleSet
  error: string | null
}

// Parses DSL text produced by ExpressionBuilder back into a DecisionRuleSet. Blank text is a
// valid "no conditions" state; unparsable text keeps `previous` and reports an actionable error
// instead of silently discarding the operator's edit.
export function exprTextToDecisionRuleSet(
  text: string,
  metadata: DecisionLeafMetadataIndex,
  previous: DecisionRuleSet,
): ExprTextParseResult {
  if (!text.trim()) {
    return { rules: { operator: 'AND', conditions: [] }, error: null }
  }
  const node = parseExprText(text)
  if (!node) {
    return { rules: previous, error: `Could not parse rule expression: "${text}"` }
  }
  return { rules: ruleNodeToDecisionRuleSet(node, metadata), error: null }
}

// Validates directly against DecisionCondition rather than through decisionRuleSetToRuleNode:
// that conversion silently keeps only a NOT node's first child (mirroring how the backend's
// evalNOT already treats any other count as a non-match), which would hide a malformed NOT
// from validation instead of blocking the save with an actionable error.
function validateCondition(
  condition: DecisionCondition,
  availableSignals: SignalDescriptor[],
): string[] {
  if (condition.operator) {
    const children = condition.conditions ?? []
    const warnings =
      condition.operator === 'NOT' && children.length !== 1
        ? ['NOT must have exactly one child']
        : []
    return warnings.concat(children.flatMap((child) => validateCondition(child, availableSignals)))
  }
  const isDefined = availableSignals.some(
    (signal) => signal.signalType === condition.type && signal.name === condition.name,
  )
  return isDefined ? [] : [`Signal ${condition.type}("${condition.name}") is not defined`]
}

export function validateDecisionRules(
  ruleSet: DecisionRuleSet | undefined,
  availableSignals: SignalDescriptor[],
): string[] {
  if (!ruleSet?.conditions?.length) return ['At least one condition is required.']
  return validateCondition(
    { operator: ruleSet.operator, conditions: ruleSet.conditions },
    availableSignals,
  )
}
