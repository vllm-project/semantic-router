import type { BoolExprNode } from '@/types/dsl'
import {
  childrenOf,
  isEmptyNode,
  isRecord,
  operatorOf,
  ruleExpression,
  type RuleNode,
} from '../configPageDecisionRuleSupport'

// The Builder edits a route's WHEN with the same rule tree editor as the
// Decisions page. The compiler's AST is a binary expression; the editor works
// on the canonical n-ary tree, and the canonical writer prints it back as DSL.

/** The canonical rule tree of a route's WHEN expression. */
export function whenToRuleTree(expr: BoolExprNode | null | undefined): RuleNode {
  if (!expr) return {}
  switch (expr.type) {
    case 'signal_ref':
      return { ...(expr.fields ?? {}), type: expr.signalType, name: expr.signalName }
    case 'not':
      return { operator: 'NOT', conditions: [whenToRuleTree(expr.expr)] }
    case 'and':
    case 'or': {
      const operator = expr.type === 'and' ? 'AND' : 'OR'
      const flatten = (node: BoolExprNode): RuleNode[] =>
        node.type === expr.type
          ? [...flatten(node.left), ...flatten(node.right)]
          : [whenToRuleTree(node)]
      return { operator, conditions: flatten(expr) }
    }
    default:
      return {}
  }
}

/** Drops conditions that are still being filled in, so a draft never writes invalid DSL. */
function completeRuleTree(node: RuleNode): RuleNode | null {
  if (!isRecord(node) || isEmptyNode(node)) return null
  const operator = operatorOf(node)
  if (!operator) {
    return typeof node.type === 'string' && node.type && typeof node.name === 'string' && node.name
      ? node
      : null
  }
  const conditions = childrenOf(node)
    .map(completeRuleTree)
    .filter((child): child is RuleNode => child !== null)
  if (conditions.length === 0) return null
  if (operator === 'NOT') return { ...node, operator, conditions: conditions.slice(0, 1) }
  return conditions.length === 1 ? conditions[0] : { ...node, operator, conditions }
}

/** The WHEN expression of a rule tree, or an empty string when the route always matches. */
export function ruleTreeToWhen(node: RuleNode): string {
  const complete = completeRuleTree(node)
  return complete ? ruleExpression(complete) : ''
}

/** Conditions of a draft tree that cannot be written yet. */
export function incompleteConditionCount(node: RuleNode, root = true): number {
  if (!isRecord(node)) return 0
  if (isEmptyNode(node)) return root ? 0 : 1
  const operator = operatorOf(node)
  if (!operator) return typeof node.type === 'string' && node.type && node.name ? 0 : 1
  const children = childrenOf(node)
  if (children.length === 0) return 1
  return children.reduce((total, child) => total + incompleteConditionCount(child, false), 0)
}
