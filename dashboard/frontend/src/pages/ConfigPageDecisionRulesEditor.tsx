import { useCallback, useMemo } from 'react'

import DecisionRuleEditor from '../components/DecisionRuleEditor'
import {
  canonicalToRuleDocument,
  describePredicate,
  ruleDocumentToCanonical,
  type DecisionRuleDocument,
  type RuleIssue,
} from '../lib/decisionRuleAst'
import { ruleSignalScope, type RuleSignalScope } from '../lib/decisionRuleValidation'
import { isSingleConditionRules } from './configPageDecisionFormSupport'
import type { DecisionCondition, DecisionRuleSet } from './configPageSupport'
import styles from './ConfigPageDecisionsSection.module.css'

interface ConfigPageDecisionRulesEditorProps {
  value: DecisionRuleSet
  onChange?: (value: DecisionRuleSet) => void
  readOnly?: boolean
  /** Signals of the selected routing scope (a named recipe sees only its own). */
  signalScope?: RuleSignalScope
  onIssuesChange?: (issues: RuleIssue[]) => void
}

const EMPTY_SCOPE = ruleSignalScope([], true)

function RuleSummary({ node }: { node: DecisionCondition }) {
  if (node.operator || node.conditions?.length) {
    return (
      <article className={styles.viewCard}>
        <div className={styles.viewHeading}>
          <span className={styles.viewTitle}>{node.operator || 'AND'} group</span>
          <span className={styles.viewBadge}>{node.conditions?.length || 0} conditions</span>
        </div>
        <div className={styles.viewStack}>
          {(node.conditions || []).map((condition, index) => (
            <RuleSummary key={index} node={condition} />
          ))}
        </div>
      </article>
    )
  }

  const predicate = describePredicate(node.predicate)
  const details = [
    node.label ? `Label: ${node.label}` : null,
    predicate ? `Predicate: ${predicate}` : null,
    node.on_error ? `On error: ${node.on_error}` : null,
  ].filter((detail): detail is string => Boolean(detail))
  return (
    <article className={styles.viewCard}>
      <div className={styles.viewHeading}>
        <span className={styles.viewTitle}>
          {node.type || 'Incomplete condition'}: {node.name || 'Not set'}
        </span>
      </div>
      {details.length > 0 ? (
        <div className={styles.viewBadgeRow}>
          {details.map((detail) => (
            <span key={detail} className={styles.viewBadge}>
              {detail}
            </span>
          ))}
        </div>
      ) : null}
    </article>
  )
}

function RulesView({ value }: { value: DecisionRuleSet }) {
  const singleCondition = isSingleConditionRules(value)
  const { on_unknown: onUnknown, ...rootCondition } = value
  const operator = value.operator || ''
  const conditions = singleCondition ? [rootCondition] : value.conditions || []

  if (!operator && conditions.length === 0) {
    if (!onUnknown) return <span>Unconditional match</span>
    return (
      <div className={styles.viewBadgeRow}>
        <span className={styles.viewBadge}>Unconditional match</span>
        <span className={styles.viewBadge}>On unknown: {onUnknown}</span>
      </div>
    )
  }
  return (
    <div className={styles.viewStack}>
      <div className={styles.viewBadgeRow}>
        <span className={styles.viewBadge}>
          {singleCondition ? 'Single condition' : `${operator || 'AND'} root`}
        </span>
        {onUnknown ? <span className={styles.viewBadge}>On unknown: {onUnknown}</span> : null}
      </div>
      {conditions.map((condition, index) => (
        <RuleSummary key={index} node={condition} />
      ))}
    </div>
  )
}

export default function ConfigPageDecisionRulesEditor({
  value,
  onChange,
  readOnly = false,
  signalScope = EMPTY_SCOPE,
  onIssuesChange,
}: ConfigPageDecisionRulesEditorProps) {
  // The canonical rules are the source of truth; the editor keeps its own
  // node ids and only resyncs when this document differs structurally.
  const document = useMemo(() => canonicalToRuleDocument(value), [value])
  const handleChange = useCallback(
    (next: DecisionRuleDocument) => onChange?.(ruleDocumentToCanonical(next)),
    [onChange],
  )

  if (readOnly) return <RulesView value={value} />

  return (
    <div className={styles.ruleEditorFrame}>
      <DecisionRuleEditor
        value={document}
        onChange={handleChange}
        signalScope={signalScope}
        onIssuesChange={onIssuesChange}
        idPrefix="config-decision-rules"
        title="Decision rule tree"
        showOnUnknown
        onUnknownHelp={
          <p className={styles.editorHelp}>
            When a signal evaluator fails, no_match skips this decision, match selects it, and
            fail_request rejects the request. Leave this empty to use a classifier condition&apos;s
            on_error policy.
          </p>
        }
      />
    </div>
  )
}
