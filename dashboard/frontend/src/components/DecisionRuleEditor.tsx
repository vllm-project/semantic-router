import { useMemo, useState } from 'react'

import ExpressionBuilder from './ExpressionBuilder'
import type { SignalDescriptor } from './ExpressionBuilderSupport'
import {
  collectLeafMetadata,
  decisionRuleSetToExprText,
  exprTextToDecisionRuleSet,
  validateDecisionRules,
} from '../pages/configPageDecisionRuleBridge'
import type { DecisionRuleSet } from '../pages/configPageSupport'

interface DecisionRuleEditorProps {
  value: DecisionRuleSet
  onChange: (next: DecisionRuleSet) => void
  availableSignals: SignalDescriptor[]
}

// Shared recursive rule-tree editor for decisions: the same ExpressionBuilder canvas Builder
// route conditions use, adapted to DecisionRuleSet's on-disk shape (label/predicate/on_error
// leaves, arbitrarily nested AND/OR/NOT) via configPageDecisionRuleBridge.
export default function DecisionRuleEditor({
  value,
  onChange,
  availableSignals,
}: DecisionRuleEditorProps) {
  const [parseError, setParseError] = useState<string | null>(null)
  const exprText = useMemo(() => decisionRuleSetToExprText(value), [value])
  const warnings = useMemo(
    () => validateDecisionRules(value, availableSignals),
    [value, availableSignals],
  )

  const handleChange = (expr: string) => {
    const { rules, error } = exprTextToDecisionRuleSet(expr, collectLeafMetadata(value), value)
    setParseError(error)
    if (!error) onChange(rules)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div
        style={{ minHeight: '320px', maxHeight: '50vh', display: 'flex', flexDirection: 'column' }}
      >
        <ExpressionBuilder
          value={exprText}
          onChange={handleChange}
          availableSignals={availableSignals}
        />
      </div>
      {parseError && (
        <p role="alert" style={{ color: 'var(--color-danger)', fontSize: '0.875rem', margin: 0 }}>
          {parseError}
        </p>
      )}
      {!parseError && warnings.length > 0 && (
        <ul
          role="note"
          style={{
            color: 'var(--color-warning)',
            fontSize: '0.875rem',
            margin: 0,
            paddingLeft: '1.25rem',
          }}
        >
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
