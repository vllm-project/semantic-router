// CustomNodes/DecisionNode.tsx - Decision node with collapsible rules

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { DecisionConfig } from '../../types'
import { NODE_COLORS } from '../../constants'
import { buildRulePreviewLines, summarizeRuleNode } from '../../utils/ruleTree'
import styles from './CustomNodes.module.css'

interface DecisionNodeData {
  decision: DecisionConfig
  rulesCollapsed?: boolean
  isHighlighted?: boolean
  isFocusTarget?: boolean
  focusModeEnabled?: boolean
  isUnreachable?: boolean
  unreachableReason?: string
  onToggleRulesCollapse?: () => void
  onFocusDecision?: (decisionName: string) => void
}

export const DecisionNode = memo<NodeProps<DecisionNodeData>>(({ data }) => {
  const { 
    decision, 
    rulesCollapsed = false, 
    isHighlighted, 
    isFocusTarget = false,
    focusModeEnabled = false,
    isUnreachable = false,
    unreachableReason,
    onToggleRulesCollapse,
    onFocusDecision,
  } = data
  const { name, priority, rules, modelRefs, algorithm, plugins } = decision

  const hasReasoning = modelRefs.some(m => m.use_reasoning)
  const hasPlugins = plugins && plugins.length > 0
  const hasAlgorithm = algorithm && algorithm.type !== 'static'
  const previewConditions = rules.conditions.slice(0, 4).map((condition, index) => ({
    key: `condition-${index}`,
    title: summarizeRuleNode(condition),
    lines: buildRulePreviewLines(condition, {
      includeRootOperator: true,
      maxLines: 3,
    }),
  }))
  
  // Use warning colors for unreachable decisions
  const colors = isUnreachable 
    ? NODE_COLORS.decision.unreachable 
    : hasReasoning 
      ? NODE_COLORS.decision.reasoning 
      : NODE_COLORS.decision.normal

  return (
    <div
      className={`${styles.decisionNode} ${isHighlighted ? styles.highlighted : ''} ${isUnreachable ? styles.unreachable : ''} ${isFocusTarget ? styles.focusTarget : ''}`}
      style={{
        background: colors.background,
        border: `2px solid ${colors.border}`,
        cursor: focusModeEnabled ? 'pointer' : undefined,
      }}
      title={isUnreachable ? `⚠️ Unreachable: ${unreachableReason}` : undefined}
      onClick={() => {
        if (focusModeEnabled) {
          onFocusDecision?.(name)
        }
      }}
    >
      <Handle type="target" position={Position.Left} />

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

        {!rulesCollapsed && previewConditions.length > 0 && (
          <div className={styles.conditionsList}>
            {previewConditions.map((condition) => {
              return (
                <div
                  key={condition.key}
                  className={styles.conditionTree}
                  title={condition.title}
                >
                  {condition.lines.map((line) => {
                    const rowClassName = line.kind === 'operator'
                      ? styles.conditionOperatorRow
                      : line.kind === 'more'
                        ? styles.conditionMoreRow
                        : styles.conditionLeafRow

                    return (
                      <div
                        key={line.key}
                        className={`${styles.conditionRow} ${rowClassName}`}
                        style={{ paddingInlineStart: `${Math.min(line.depth, 2) * 10}px` }}
                      >
                        <span className={line.kind === 'operator' ? styles.conditionOperatorBadge : styles.conditionText}>
                          {line.text}
                        </span>
                      </div>
                    )
                  })}
                </div>
              )
            })}
            {rules.conditions.length > 4 && (
              <div className={styles.conditionTree}>
                <div className={`${styles.conditionRow} ${styles.conditionMoreRow}`}>
                  <span className={styles.conditionText}>+{rules.conditions.length - 4} more</span>
                </div>
              </div>
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

      <Handle type="source" position={Position.Right} />
    </div>
  )
})

DecisionNode.displayName = 'DecisionNode'
