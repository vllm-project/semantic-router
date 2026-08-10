import type { DecisionCondition, NumericPredicate } from './configPageSupport'

function formatPredicate(predicate: NumericPredicate): string {
  return (Object.entries(predicate) as [keyof NumericPredicate, number | undefined][])
    .filter(([, threshold]) => threshold !== undefined)
    .map(([comparator, threshold]) => `${comparator} ${threshold}`)
    .join(', ')
}

interface DecisionConditionViewProps {
  condition: DecisionCondition
  depth: number
}

// Recursively renders a DecisionCondition, which is either a leaf ({type, name, ...metadata})
// or a nested group ({operator, conditions}). ConfigPage's decision viewer previously read
// `cond.type`/`cond.name` unconditionally, so a nested group rendered as "undefined: undefined".
export function DecisionConditionView({ condition, depth }: DecisionConditionViewProps) {
  const indent = { marginLeft: depth > 0 ? '1rem' : 0 }

  if (condition.operator) {
    return (
      <div style={{ ...indent, display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.75rem',
            fontWeight: 600,
            opacity: 0.75,
          }}
        >
          {condition.operator}
        </span>
        {(condition.conditions ?? []).map((child, index) => (
          <DecisionConditionView key={index} condition={child} depth={depth + 1} />
        ))}
      </div>
    )
  }

  const badges = [
    condition.label,
    condition.predicate ? formatPredicate(condition.predicate) : null,
    condition.on_error ? `on_error: ${condition.on_error}` : null,
  ].filter((badge): badge is string => Boolean(badge))

  return (
    <div
      style={{
        ...indent,
        padding: '0.5rem',
        background: 'rgba(143, 148, 156, 0.1)',
        borderRadius: '4px',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.875rem',
      }}
    >
      {condition.type}: {condition.name}
      {badges.length > 0 && (
        <div style={{ fontSize: '0.75rem', opacity: 0.75, marginTop: '0.25rem' }}>
          {badges.join(' · ')}
        </div>
      )}
    </div>
  )
}
