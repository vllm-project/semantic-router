// CustomNodes/ModelNode.tsx - Model node with aggregated reasoning modes display

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { ModelRefConfig } from '../../types'
import { REASONING_EFFORT_DISPLAY, NODE_COLORS } from '../../constants'
import styles from './CustomNodes.module.css'

interface ModelMode {
  decisionName: string
  hasReasoning: boolean
  reasoningEffort?: string
}

interface ModelNodeData {
  modelRef: ModelRefConfig
  decisionName: string
  fromDecisions?: string[]
  isHighlighted?: boolean
  // New: aggregated modes from multiple decisions
  modes?: ModelMode[]
  hasMultipleModes?: boolean
}

export const ModelNode = memo<NodeProps<ModelNodeData>>(({ data }) => {
  const { modelRef, decisionName, isHighlighted, modes, hasMultipleModes } = data
  const { model, lora_name, reasoning_family } = modelRef

  // Extract short model name
  const shortName = model.split('/').pop() || model

  // Analyze modes: group by reasoning status
  const reasoningModes = modes?.filter(m => m.hasReasoning) || []
  const standardModes = modes?.filter(m => !m.hasReasoning) || []
  
  // Determine primary display mode
  const hasAnyReasoning = reasoningModes.length > 0
  const hasAnyStandard = standardModes.length > 0

  // Node colors based on whether it has reasoning capability
  const colors = hasAnyReasoning
    ? NODE_COLORS.model.reasoning
    : NODE_COLORS.model.standard

  // Get unique reasoning efforts
  const reasoningEfforts = [...new Set(reasoningModes.map(m => m.reasoningEffort).filter(Boolean))]

  return (
    <div
      className={`${styles.modelNode} ${isHighlighted ? styles.highlighted : ''}`}
      style={{
        background: colors.background,
        border: `2px solid ${colors.border}`,
        minWidth: hasMultipleModes ? '180px' : '160px',
      }}
    >
      <Handle type="target" position={Position.Top} />

      {/* Model Name */}
      <div className={styles.modelHeader}>
        <span className={styles.modelIcon}>🤖</span>
        <span className={styles.modelName} title={model}>{shortName}</span>
      </div>

      {/* Modes Section - Show aggregated modes */}
      <div className={styles.modelFeatures}>
        {/* Show both modes if model is used in multiple ways */}
        {hasMultipleModes && hasAnyReasoning && hasAnyStandard ? (
          <div className={styles.modesContainer}>
            {/* Reasoning Mode */}
            <div className={styles.modeBadge} style={{ background: 'rgba(118, 185, 0, 0.2)', borderColor: '#76b900', color: 'white' }}>
              <span>🧠</span>
              <span>Reasoning</span>
              {reasoning_family && (
                <span className={styles.reasoningFamily}>({reasoning_family})</span>
              )}
              {reasoningEfforts.length > 0 && (
                <span className={styles.effortTag}>
                  {reasoningEfforts.map(e => e ? (REASONING_EFFORT_DISPLAY[e]?.icon || e) : '').join(' ')}
                </span>
              )}
            </div>
            {/* Standard Mode */}
            <div className={styles.modeBadge} style={{ background: 'rgba(100, 100, 100, 0.3)', borderColor: '#666' }}>
              <span>⚡</span>
              <span>Standard</span>
            </div>
          </div>
        ) : (
          <>
            {/* Single Reasoning Mode */}
            {hasAnyReasoning && (
              <div className={styles.reasoningBadge}>
                <span className={styles.reasoningIcon}>🧠</span>
                <span>Reasoning</span>
                {reasoning_family && (
                  <span className={styles.reasoningFamily}>({reasoning_family})</span>
                )}
              </div>
            )}

            {/* Reasoning Effort Level */}
            {reasoningEfforts.length > 0 && reasoningEfforts.map((effort, idx) => {
              if (!effort) return null
              const effortConfig = REASONING_EFFORT_DISPLAY[effort]
              return effortConfig ? (
                <div
                  key={idx}
                  className={styles.effortBadge}
                  style={{ background: effortConfig.color }}
                  title={`Reasoning Effort: ${effortConfig.label}`}
                >
                  <span>{effortConfig.icon}</span>
                  <span>{effortConfig.label}</span>
                </div>
              ) : null
            })}

            {/* Standard Mode Only */}
            {!hasAnyReasoning && (
              <div className={styles.standardBadge}>
                <span>⚡</span>
                <span>Standard</span>
              </div>
            )}
          </>
        )}

        {/* LoRA Adapter */}
        {lora_name && (
          <div className={styles.loraBadge} title={`LoRA Adapter: ${lora_name}`}>
            <span className={styles.loraIcon}>🎨</span>
            <span className={styles.loraName}>LoRA: {lora_name}</span>
          </div>
        )}
      </div>

      {/* Source Decisions */}
      <div className={styles.modelSource}>
        <span className={styles.sourceLabel}>from:</span>
        <span className={styles.sourceName} title={decisionName}>
          {decisionName.length > 30 ? decisionName.slice(0, 27) + '...' : decisionName}
        </span>
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
})

ModelNode.displayName = 'ModelNode'
