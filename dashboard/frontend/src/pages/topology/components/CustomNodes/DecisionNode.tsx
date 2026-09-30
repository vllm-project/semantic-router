// CustomNodes/DecisionNode.tsx - Decision node with collapsible rules

import { memo, useId } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { DecisionConfig } from '../../types'
import { NODE_COLORS } from '../../constants'
import { buildRulePreviewLines, summarizeRuleNode } from '../../utils/ruleTree'
import { formatRoutingMetadataValue } from '../../../../components/routingMetadataDisplay'
import styles from './CustomNodes.module.css'

interface DecisionNodeData {
  decision: DecisionConfig
  rulesCollapsed?: boolean
  isHighlighted?: boolean
  isFocusTarget?: boolean
  focusModeEnabled?: boolean
  isFallback?: boolean
  isUnreachable?: boolean
  unreachableReason?: string
  onToggleRulesCollapse?: () => void
  onFocusDecision?: (decisionName: string) => void
}

interface RulesHeaderProps {
  rulesCollapsed: boolean
  hasRuleDetail: boolean
  conditionsId: string
  label: string
  count: string
  onToggle?: () => void
}

/** Toggle for a decision node rules list. Extracted to keep the node body on its own seam. */
const RulesHeader: React.FC<RulesHeaderProps> = ({
  rulesCollapsed,
  hasRuleDetail,
  conditionsId,
  label,
  count,
  onToggle,
}) => (
  <button
    type="button"
    className={styles.rulesHeader}
    onClick={onToggle}
    aria-expanded={hasRuleDetail ? !rulesCollapsed : undefined}
    aria-controls={hasRuleDetail ? conditionsId : undefined}
  >
    <span className={styles.collapseIcon}>{rulesCollapsed ? '▶' : '▼'}</span>
    <span className={styles.rulesOperator}>{label}</span>
    <span className={styles.rulesCount}>
      {count}
    </span>
  </button>
)

interface RulesConditionListProps {
  conditionsId: string
  previewConditions: ReadonlyArray<{
    key: string
    title: string
    lines: ReturnType<typeof buildRulePreviewLines>
  }>
  totalConditions: number
}

/** Rendered rule preview for a decision node. Extracted to keep the node body readable. */
const RulesConditionList: React.FC<RulesConditionListProps> = ({
  conditionsId,
  previewConditions,
  totalConditions,
}) => (
  <div id={conditionsId} className={styles.conditionsList}>
    {previewConditions.map((condition) => {
      return (
        <div key={condition.key} className={styles.conditionTree} title={condition.title}>
          {condition.lines.map((line) => {
            const rowClassName =
              line.kind === 'operator'
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
                <span
                  className={
                    line.kind === 'operator'
                      ? styles.conditionOperatorBadge
                      : styles.conditionText
                  }
                >
                  {line.text}
                </span>
              </div>
            )
          })}
        </div>
      )
    })}
    {totalConditions > 4 && (
      <div className={styles.conditionTree}>
        <div className={`${styles.conditionRow} ${styles.conditionMoreRow}`}>
          <span className={styles.conditionText}>+{totalConditions - 4} more</span>
        </div>
      </div>
    )}
  </div>
)

export const DecisionNode = memo<NodeProps<DecisionNodeData>>(({ data }) => {
  const {
    decision,
    rulesCollapsed = false,
    isHighlighted,
    isFocusTarget = false,
    focusModeEnabled = false,
    isFallback = false,
    isUnreachable = false,
    unreachableReason,
    onToggleRulesCollapse,
    onFocusDecision,
  } = data
  const { name, priority, rules, modelRefs, algorithm, plugins } = decision
  const displayName = formatRoutingMetadataValue('x-vsr-selected-decision', name)

  const hasReasoning = modelRefs.some((m) => m.use_reasoning)
  const hasPlugins = plugins && plugins.length > 0
  const hasAlgorithm = algorithm && algorithm.type !== 'static'
  const conditionsId = useId()

  const previewConditions = rules.conditions.slice(0, 4).map((condition, index) => ({
    key: `condition-${index}`,
    title: summarizeRuleNode(condition),
    lines: buildRulePreviewLines(condition, {
      includeRootOperator: true,
      maxLines: 3,
    }),
  }))

  const hasRuleDetail = previewConditions.length > 0

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
      title={
        isUnreachable
          ? `⚠️ Unreachable: ${unreachableReason}`
          : isFallback
            ? 'Fallback route: matches when no earlier decision wins'
            : undefined
      }
      onClick={() => {
        if (focusModeEnabled) {
          onFocusDecision?.(name)
        }
      }}
    >
      <Handle type="target" position={Position.Left} />

      <div className={styles.decisionHeader}>
        <span className={styles.decisionIcon}>
          {isUnreachable ? '⚠️' : isFallback ? '↪' : '🔀'}
        </span>
        <span className={styles.decisionName} title={name}>
          {displayName}
        </span>
        <span className={styles.decisionPriority}>P{priority}</span>
      </div>

      {/* Unreachable Warning Banner */}
      {isUnreachable && (
        <div className={styles.unreachableBanner}>⚠️ {unreachableReason || 'Unreachable'}</div>
      )}

      {/* Rules Section */}
      <div className={styles.rulesSection}>
        <RulesHeader
          rulesCollapsed={rulesCollapsed}
          hasRuleDetail={hasRuleDetail}
          conditionsId={conditionsId}
          label={isFallback ? 'FALLBACK' : rules.operator}
          count={isFallback ? 'Always matches' : `${rules.conditions.length} rules`}
          onToggle={onToggleRulesCollapse}
        />

        {!rulesCollapsed && hasRuleDetail && (
          <RulesConditionList
            conditionsId={conditionsId}
            previewConditions={previewConditions}
            totalConditions={rules.conditions.length}
          />
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
        {modelRefs.length > 2 && <span className={styles.moreModels}>+{modelRefs.length - 2}</span>}
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  )
})

DecisionNode.displayName = 'DecisionNode'
