// Reading helpers for the canonical rule tree: a leaf {type, name, label,
// predicate, on_error}, a group {operator: AND | OR | NOT, conditions}, or an
// empty tree, which matches every request.

export type RuleNode = Record<string, unknown>
export type GroupOperator = 'AND' | 'OR' | 'NOT'

export function isRecord(value: unknown): value is RuleNode {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

export function operatorOf(node: RuleNode): GroupOperator | null {
  const operator = typeof node.operator === 'string' ? node.operator.toUpperCase() : ''
  if (operator === 'AND' || operator === 'OR' || operator === 'NOT') return operator
  return Array.isArray(node.conditions) ? 'AND' : null
}

export function childrenOf(node: RuleNode): RuleNode[] {
  return Array.isArray(node.conditions) ? node.conditions.filter(isRecord) : []
}

export function isEmptyNode(node: RuleNode): boolean {
  return Object.keys(node).filter((key) => key !== 'on_unknown').length === 0
}

function quoted(value: unknown): string {
  return JSON.stringify(String(value ?? ''))
}

/** The rule tree as the DSL writes it, for summaries and the read-only view. */
export function ruleExpression(node: unknown): string {
  if (!isRecord(node) || isEmptyNode(node)) return 'always'
  const operator = operatorOf(node)
  if (!operator) {
    const parts = [quoted(node.name)]
    if (node.label) parts.push(`label: ${quoted(node.label)}`)
    const bounds = isRecord(node.predicate) ? Object.entries(node.predicate) : []
    if (bounds.length)
      parts.push(`predicate: {${bounds.map(([key, value]) => `${key}: ${value}`).join(', ')}}`)
    if (node.on_error) parts.push(`on_error: ${quoted(node.on_error)}`)
    return `${node.type || '?'}(${parts.join(', ')})`
  }
  const children = childrenOf(node)
  if (operator === 'NOT') {
    const inner = children[0] ? ruleExpression(children[0]) : '?'
    return children[0] && operatorOf(children[0]) === 'AND' ? `NOT (${inner})` : `NOT ${inner}`
  }
  const flatten = (current: RuleNode): string[] =>
    operatorOf(current) === operator
      ? childrenOf(current).flatMap(flatten)
      : [ruleExpression(current)]
  const parts = children.flatMap(flatten)
  if (parts.length === 0) return operator === 'AND' ? 'always' : 'never'
  if (operator === 'OR') return parts.length === 1 ? parts[0] : `(${parts.join(' OR ')})`
  return parts.join(' AND ')
}

/** Every signal condition in the tree, in reading order, with whether it is negated. */
export function ruleLeaves(
  node: unknown,
  negated = false,
): Array<{ node: RuleNode; negated: boolean }> {
  if (!isRecord(node) || isEmptyNode(node)) return []
  const operator = operatorOf(node)
  if (!operator) return [{ node, negated }]
  return childrenOf(node).flatMap((child) =>
    ruleLeaves(child, operator === 'NOT' ? !negated : negated),
  )
}

export function countRuleConditions(node: unknown): number {
  return ruleLeaves(node).length
}

/** The top-level connective of a tree, for compact summaries. */
export function ruleConnective(node: unknown): 'AND' | 'OR' {
  return isRecord(node) && operatorOf(node) === 'OR' ? 'OR' : 'AND'
}
