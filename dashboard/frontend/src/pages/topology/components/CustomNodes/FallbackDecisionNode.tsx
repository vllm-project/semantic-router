// CustomNodes/FallbackDecisionNode.tsx - System fallback decision node

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import styles from './CustomNodes.module.css'

interface FallbackDecisionNodeData {
  decisionName: string
  fallbackReason?: string
  defaultModel?: string
  isHighlighted?: boolean
}

// Mapping of system fallback decision names to display info
const FALLBACK_DECISION_INFO: Record<string, { icon: string; label: string; description: string }> = {
  low_confidence_general: {
    icon: '📉',
    label: 'Low Confidence',
    description: 'Classification confidence below threshold',
  },
  high_confidence_specialized: {
    icon: '📈',
    label: 'High Confidence',
    description: 'Classification confidence above threshold',
  },
}

export const FallbackDecisionNode = memo<NodeProps<FallbackDecisionNodeData>>(({ data }) => {
  const { decisionName, fallbackReason, defaultModel, isHighlighted } = data
  
  const info = FALLBACK_DECISION_INFO[decisionName] || {
    icon: '⚠️',
    label: decisionName,
    description: 'System fallback decision',
  }

  return (
    <div
      className={`${styles.fallbackDecisionNode} ${isHighlighted ? styles.highlighted : ''}`}
      title={fallbackReason || info.description}
    >
      <Handle type="target" position={Position.Top} />

      <div className={styles.fallbackDecisionHeader}>
        <span className={styles.fallbackDecisionIcon}>{info.icon}</span>
        <span className={styles.fallbackDecisionTitle}>{info.label}</span>
      </div>

      <div className={styles.fallbackDecisionInfo}>
        <span className={styles.fallbackDecisionBadge}>System Fallback</span>
      </div>

      <div className={styles.fallbackDecisionReason}>
        {fallbackReason || info.description}
      </div>

      {defaultModel && (
        <div className={styles.fallbackDecisionModel}>
          → {defaultModel.length > 20 ? defaultModel.slice(0, 20) + '...' : defaultModel}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
})

FallbackDecisionNode.displayName = 'FallbackDecisionNode'
