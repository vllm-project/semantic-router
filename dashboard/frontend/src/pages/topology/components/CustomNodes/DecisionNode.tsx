// CustomNodes/DecisionNode.tsx - Decision node with collapsible rules

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { DecisionConfig } from '../../types'
import { SIGNAL_ICONS, NODE_COLORS } from '../../constants'
import styles from './CustomNodes.module.css'

interface DecisionNodeData {
  decision: DecisionConfig
  rulesCollapsed?: boolean
  isHighlighted?: boolean
  isUnreachable?: boolean
  unreachableReason?: string
  onToggleRulesCollapse?: () => void
}

export const DecisionNode = memo<NodeProps<DecisionNodeData>>(({ data }) => {
  const { 
    decision, 
    rulesCollapsed = false, 
    isHighlighted, 
    isUnreachable = false,
    unreachableReason,
    onToggleRulesCollapse 
  } = data
  const { name, priority, rules, modelRefs, algorithm, plugins } = decision

  const hasReasoning = modelRefs.some(m => m.use_reasoning)
  const hasPlugins = plugins && plugins.length > 0
  const hasAlgorithm = algorithm && algorithm.type !== 'static'
  
  // Use warning colors for unreachable decisions
  const colors = isUnreachable 
    ? NODE_COLORS.decision.unreachable 
    : hasReasoning 
      ? NODE_COLORS.decision.reasoning 
      : NODE_COLORS.decision.normal

  return (
    <div
      className={`${styles.decisionNode} ${isHighlighted ? styles.highlighted : ''} ${isUnreachable ? styles.unreachable : ''}`}
      style={{
        background: colors.background,
        border: `2px solid ${colors.border}`,
      }}
      title={isUnreachable ? `⚠️ Unreachable: ${unreachableReason}` : undefined}
    >
      <Handle type="target" position={Position.Top} />

      <div className={styles.decisionHeader}>
        <span className={styles.decisionIcon}>{isUnreachable ? '⚠️' : '🔀'}</span>
        <span className={styles.decisionName} title={name}>{name}</span>
        <span className={styles.decisionPriority}>P{priority}</span>
      </div>

      {/* Unreachable Warning Banner */}
      {isUnreachable && (
        <div className={styles.unreachableBanner}>
          ⚠️ {unreachableReason || 'Unreachable'}
        </div>
      )}

      {/* Rules Section */}
      <div className={styles.rulesSection}>
        <div
          className={styles.rulesHeader}
          onClick={onToggleRulesCollapse}
        >
          <span className={styles.collapseIcon}>{rulesCollapsed ? '▶' : '▼'}</span>
          <span className={styles.rulesOperator}>{rules.operator}</span>
          <span className={styles.rulesCount}>
            {rules.conditions.length === 0 ? '0 rules ⚠️' : `${rules.conditions.length} rules`}
          </span>
        </div>

        {!rulesCollapsed && rules.conditions.length > 0 && (
          <div className={styles.conditionsList}>
            {rules.conditions.slice(0, 4).map((cond, idx) => (
              <span key={idx} className={styles.condition}>
                {SIGNAL_ICONS[cond.type] || '❓'} {cond.type}: {cond.name}
              </span>
            ))}
            {rules.conditions.length > 4 && (
              <span className={styles.condition} style={{ opacity: 0.7 }}>
                +{rules.conditions.length - 4} more
              </span>
            )}
          </div>
        )}
      </div>

      {/* Metadata Tags */}
      <div className={styles.decisionMeta}>
        {hasAlgorithm && (
          <span className={styles.metaTag} title="Multi-model algorithm">
            🔄 {algorithm!.type}
          </span>
        )}
        {hasPlugins && (
          <span className={styles.metaTag} title="Has plugins">
            🔌 {plugins!.length}
          </span>
        )}
        {hasReasoning && (
          <span className={styles.metaTag} title="Reasoning enabled">
            🧠
          </span>
        )}
      </div>

      {/* Models Preview */}
      <div className={styles.modelsList}>
        {modelRefs.slice(0, 2).map((ref, idx) => (
          <span key={idx} className={styles.modelItem}>
            {ref.model.split('/').pop()}
          </span>
        ))}
        {modelRefs.length > 2 && (
          <span className={styles.moreModels}>+{modelRefs.length - 2}</span>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
})

DecisionNode.displayName = 'DecisionNode'
