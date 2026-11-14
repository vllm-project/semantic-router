/**
 * Issues the shared decision-rule editor reports to its host.
 *
 * The editor only reports what stops it writing the host's format: a raw
 * expression that does not parse, and a tree the DSL cannot express. Whether
 * a rule is valid is the Router's decision, made when the host saves the
 * config (ConfigPage) or compiles the DSL (Builder), so the Router's rule
 * validators are not repeated here.
 */

import type { RuleIssue } from './decisionRuleAst'

export type { RuleIssue }

/** Prefix of the issue an editor reports for an unparsable raw expression. */
export const RAW_EXPRESSION_INCOMPLETE = 'Raw expression is incomplete'

/** The message of an editor's unparsable raw-expression issue, if it reported one. */
export function incompleteRawExpression(issues: RuleIssue[]): string | null {
  return issues.find((issue) => issue.message.startsWith(RAW_EXPRESSION_INCOMPLETE))?.message ?? null
}

export function blockingRuleIssues(issues: RuleIssue[]): RuleIssue[] {
  return issues.filter((issue) => issue.severity === 'error')
}

/** "rules.conditions[0].conditions[1]" style location, matching Go error paths. */
export function formatRuleIssuePath(path: number[]): string {
  return ['rules', ...path.map((index) => `conditions[${index}]`)].join('.')
}
