import styles from './ConfigPageDecisionConditionsView.module.css'
import type { DecisionCondition } from './configPageSupport'

interface ConfigPageDecisionConditionsViewProps {
  conditions: DecisionCondition[]
}

const operatorLabels: Record<NonNullable<DecisionCondition['operator']>, string> = {
  AND: 'All of',
  OR: 'Any of',
  NOT: 'Not',
}

const predicateSymbols = {
  gt: '>',
  gte: '≥',
  lt: '<',
  lte: '≤',
} as const

function humanizeType(value?: string) {
  if (!value) return 'Condition'

  return value.replace(/[_-]+/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())
}

function conditionMetadata(condition: DecisionCondition) {
  const metadata: string[] = []

  if (condition.label) metadata.push(`Label ${condition.label}`)

  for (const [operator, symbol] of Object.entries(predicateSymbols)) {
    const value = condition.predicate?.[operator as keyof typeof predicateSymbols]
    if (typeof value === 'number') metadata.push(`${symbol} ${value}`)
  }

  if (condition.on_error) {
    metadata.push(condition.on_error === 'match' ? 'Match on error' : 'Skip on error')
  }

  return metadata
}

function ConditionNode({ condition, path }: { condition: DecisionCondition; path: string }) {
  const children = condition.conditions ?? []

  if (condition.operator || children.length > 0) {
    const operator = condition.operator ?? 'AND'

    return (
      <li className={styles.node}>
        <div className={styles.groupLabel}>
          <span>{operatorLabels[operator]}</span>
          <span className={styles.groupCount}>{children.length}</span>
        </div>
        {children.length > 0 ? (
          <ul className={styles.branch}>
            {children.map((child, index) => (
              <ConditionNode
                key={`${path}-${index}-${child.operator || child.type || 'condition'}-${child.name || ''}`}
                condition={child}
                path={`${path}-${index}`}
              />
            ))}
          </ul>
        ) : (
          <span className={styles.empty}>No nested conditions</span>
        )}
      </li>
    )
  }

  const metadata = conditionMetadata(condition)

  return (
    <li className={styles.node}>
      <div className={styles.leaf}>
        <span className={styles.type}>{humanizeType(condition.type)}</span>
        <span className={styles.name}>{condition.name?.trim() || 'Unnamed condition'}</span>
        {metadata.length > 0 ? (
          <span className={styles.metadata}>{metadata.join(' · ')}</span>
        ) : null}
      </div>
    </li>
  )
}

export default function ConfigPageDecisionConditionsView({
  conditions,
}: ConfigPageDecisionConditionsViewProps) {
  if (conditions.length === 0) return <span>No conditions</span>

  return (
    <ul className={styles.tree} aria-label="Decision conditions">
      {conditions.map((condition, index) => (
        <ConditionNode
          key={`${index}-${condition.operator || condition.type || 'condition'}-${condition.name || ''}`}
          condition={condition}
          path={`${index}`}
        />
      ))}
    </ul>
  )
}
