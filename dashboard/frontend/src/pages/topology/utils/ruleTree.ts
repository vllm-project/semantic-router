import { RuleCombination, RuleCondition, RuleNode, SignalType } from '../types'

export interface RulePreviewLine {
  key: string
  depth: number
  kind: 'operator' | 'condition' | 'more'
  text: string
  title?: string
}

interface RulePreviewOptions {
  includeRootOperator?: boolean
  maxLines?: number
}

export function isRuleCombination(node: RuleNode): node is RuleCombination {
  return 'operator' in node && Array.isArray(node.conditions)
}

export function collectRuleConditions(rule: RuleNode): RuleCondition[] {
  if (!isRuleCombination(rule)) {
    return [rule]
  }

  return rule.conditions.flatMap((condition) => collectRuleConditions(condition))
}

export function collectRuleSignalTypes(rule: RuleCombination): SignalType[] {
  return Array.from(new Set(collectRuleConditions(rule).map((condition) => condition.type)))
}

export function summarizeRuleNode(node: RuleNode): string {
  if (!isRuleCombination(node)) {
    return `${node.type}: ${node.name}`
  }

  const childSummaries = node.conditions.map((condition) => summarizeRuleNode(condition))
  if (childSummaries.length === 0) {
    return `${node.operator}: no conditions`
  }

  const separator = node.operator === 'AND' ? ' & ' : node.operator === 'NOT' ? ' / ' : ' | '
  return `${node.operator}: ${childSummaries.join(separator)}`
}

function countRulePreviewLines(node: RuleNode, includeOperator: boolean): number {
  if (!isRuleCombination(node)) {
    return 1
  }

  return (includeOperator ? 1 : 0) + node.conditions.reduce(
    (total, condition) => total + countRulePreviewLines(condition, true),
    0
  )
}

export function buildRulePreviewLines(
  rule: RuleNode,
  options: RulePreviewOptions = {}
): RulePreviewLine[] {
  const includeRootOperator = options.includeRootOperator ?? true
  const totalLineCount = countRulePreviewLines(rule, includeRootOperator)
  const maxLines = options.maxLines ?? totalLineCount
  const reserveMoreLine = totalLineCount > maxLines ? 1 : 0
  const visibleBudget = Math.max(0, maxLines - reserveMoreLine)
  const lines: RulePreviewLine[] = []

  const pushLine = (line: RulePreviewLine): boolean => {
    if (lines.length >= visibleBudget) {
      return false
    }

    lines.push(line)
    return true
  }

  const walk = (node: RuleNode, depth: number, includeOperator: boolean, key: string): boolean => {
    if (isRuleCombination(node)) {
      if (includeOperator && !pushLine({
        key: `${key}-operator`,
        depth,
        kind: 'operator',
        text: node.operator,
        title: summarizeRuleNode(node),
      })) {
        return false
      }

      const childDepth = includeOperator ? depth + 1 : depth
      for (let index = 0; index < node.conditions.length; index += 1) {
        if (!walk(node.conditions[index], childDepth, true, `${key}-${index}`)) {
          return false
        }
      }

      return true
    }

    const label = `${node.type}: ${node.name}`
    return pushLine({
      key: `${key}-condition`,
      depth,
      kind: 'condition',
      text: label,
      title: label,
    })
  }

  walk(rule, 0, includeRootOperator, 'rule')

  const hiddenLineCount = totalLineCount - lines.length
  if (hiddenLineCount > 0) {
    lines.push({
      key: 'rule-more',
      depth: 0,
      kind: 'more',
      text: `+${hiddenLineCount} more`,
      title: summarizeRuleNode(rule),
    })
  }

  return lines
}
