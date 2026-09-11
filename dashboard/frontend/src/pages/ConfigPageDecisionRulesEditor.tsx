import type { FieldSchema } from '../lib/dslSchemas'
import { FieldEditor } from './builderPageFormPrimitives'
import type { DecisionCondition, DecisionRuleSet } from './configPageSupport'
import styles from './ConfigPageDecisionsSection.module.css'

interface ConfigPageDecisionRulesEditorProps {
  value: DecisionRuleSet
  onChange?: (value: DecisionRuleSet) => void
  readOnly?: boolean
}

const OPERATORS = ['AND', 'OR', 'NOT'] as const
const CONDITION_SCHEMA: FieldSchema = {
  key: 'condition',
  label: 'Condition',
  type: 'rule',
}

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

  const details = [
    node.label ? `Label: ${node.label}` : null,
    node.predicate ? `Predicate: ${JSON.stringify(node.predicate)}` : null,
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

export default function ConfigPageDecisionRulesEditor({
  value,
  onChange,
  readOnly = false,
}: ConfigPageDecisionRulesEditorProps) {
  const operator = value.operator || ''
  const conditions = value.conditions || []

  if (readOnly) {
    if (!operator && conditions.length === 0) return <span>Unconditional match</span>
    return (
      <div className={styles.viewStack}>
        <div className={styles.viewBadgeRow}>
          <span className={styles.viewBadge}>{operator || 'AND'} root</span>
          {value.on_unknown ? (
            <span className={styles.viewBadge}>On unknown: {value.on_unknown}</span>
          ) : null}
        </div>
        {conditions.map((condition, index) => (
          <RuleSummary key={index} node={condition} />
        ))}
      </div>
    )
  }

  const updateConditions = (nextConditions: DecisionCondition[]) =>
    onChange?.({ ...value, conditions: nextConditions })

  return (
    <div className={styles.editorList}>
      <div className={styles.editorGridConditions}>
        <label className={styles.editorControlLabel}>
          <span className={styles.editorControlLabelText}>Root behavior</span>
          <select
            value={operator}
            className={styles.editorSelect}
            onChange={(event) => {
              const nextOperator = event.target.value as DecisionRuleSet['operator'] | ''
              if (!nextOperator) {
                onChange?.({})
                return
              }
              onChange?.({
                ...value,
                operator: nextOperator,
                conditions:
                  nextOperator === 'NOT'
                    ? [conditions[0] || {}]
                    : conditions.length > 0
                      ? conditions
                      : [{}],
              })
            }}
          >
            <option value="">Unconditional match</option>
            {OPERATORS.map((candidate) => (
              <option key={candidate} value={candidate}>
                {candidate} group
              </option>
            ))}
          </select>
        </label>
        {operator ? (
          <label className={styles.editorControlLabel}>
            <span className={styles.editorControlLabelText}>On unknown</span>
            <select
              value={value.on_unknown || ''}
              className={styles.editorSelect}
              onChange={(event) =>
                onChange?.({
                  ...value,
                  on_unknown: (event.target.value || undefined) as DecisionRuleSet['on_unknown'],
                })
              }
            >
              <option value="">Use condition policy</option>
              <option value="no_match">No match</option>
              <option value="match">Match</option>
              <option value="fail_request">Fail request</option>
            </select>
          </label>
        ) : null}
      </div>
      {operator ? (
        <p className={styles.editorHelp}>
          When a signal evaluator fails, no_match skips this decision, match selects it, and
          fail_request rejects the request. Leave this empty to use a classifier condition&apos;s
          on_error policy.
        </p>
      ) : null}

      {operator
        ? conditions.map((condition, index) => (
            <section key={index} className={styles.editorCard}>
              <div className={styles.editorMetaRow}>
                <strong>Condition {index + 1}</strong>
                <button
                  type="button"
                  className={styles.editorButtonDanger}
                  disabled={operator === 'NOT'}
                  onClick={() =>
                    updateConditions(
                      conditions.filter((_, conditionIndex) => conditionIndex !== index),
                    )
                  }
                >
                  Remove
                </button>
              </div>
              <FieldEditor
                schema={{ ...CONDITION_SCHEMA, label: `Condition ${index + 1}` }}
                value={condition}
                onChange={(nextCondition) =>
                  updateConditions(
                    conditions.map((current, conditionIndex) =>
                      conditionIndex === index
                        ? ((nextCondition || {}) as DecisionCondition)
                        : current,
                    ),
                  )
                }
              />
            </section>
          ))
        : null}

      {operator && operator !== 'NOT' ? (
        <button
          type="button"
          className={styles.editorButtonSecondary}
          onClick={() => updateConditions([...conditions, {}])}
        >
          Add Condition
        </button>
      ) : null}
    </div>
  )
}
