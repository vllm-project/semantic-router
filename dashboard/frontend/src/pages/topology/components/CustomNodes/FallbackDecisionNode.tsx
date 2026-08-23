// CustomNodes/FallbackDecisionNode.tsx - System fallback decision node

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import styles from './CustomNodes.module.css'
import ProductIcon, { type ProductIconName } from '../../../../components/ProductIcon'

interface FallbackDecisionNodeData {
  decisionName: string
  fallbackReason?: string
  defaultModel?: string
  isHighlighted?: boolean
}

// Mapping of system fallback decision names to display info
const FALLBACK_DECISION_INFO: Record<
  string,
  { icon: ProductIconName; label: string; description: string }
> = {
  low_confidence_general: {
    icon: 'chart',
    label: 'Low Confidence',
    description: 'Classification confidence below threshold',
  },
  high_confidence_specialized: {
    icon: 'activity',
    label: 'High Confidence',
    description: 'Classification confidence above threshold',
  },
}

export const FallbackDecisionNode = memo<NodeProps<FallbackDecisionNodeData>>(({ data }) => {
  const { decisionName, fallbackReason, defaultModel, isHighlighted } = data

  const info = FALLBACK_DECISION_INFO[decisionName] || {
    icon: 'alert' as ProductIconName,
    label: decisionName,
    description: 'System fallback decision',
  }

  return (
    <div
      className={`${styles.fallbackDecisionNode} ${isHighlighted ? styles.highlighted : ''}`}
      title={fallbackReason || info.description}
    >
      <Handle type="target" position={Position.Left} />

      <div className={styles.fallbackDecisionHeader}>
        <ProductIcon className={styles.fallbackDecisionIcon} name={info.icon} aria-hidden="true" />
        <span className={styles.fallbackDecisionTitle}>{info.label}</span>
      </div>

      <div className={styles.fallbackDecisionInfo}>
        <span className={styles.fallbackDecisionBadge}>System Fallback</span>
      </div>

      <div className={styles.fallbackDecisionReason}>{fallbackReason || info.description}</div>

      {defaultModel && (
        <div className={styles.fallbackDecisionModel}>
          <ProductIcon name="chevron-right" aria-hidden="true" />
          {defaultModel.length > 20 ? defaultModel.slice(0, 20) + '...' : defaultModel}
        </div>
      )}

      <Handle type="source" position={Position.Right} />
    </div>
  )
})

FallbackDecisionNode.displayName = 'FallbackDecisionNode'
