// CustomNodes/DefaultRouteNode.tsx - Default route fallback node

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import styles from './CustomNodes.module.css'

interface DefaultRouteNodeData {
  label?: string
  defaultModel?: string
  isHighlighted?: boolean
}

export const DefaultRouteNode = memo<NodeProps<DefaultRouteNodeData>>(({ data }) => {
  const { label = 'Default Route', defaultModel, isHighlighted } = data

  return (
    <div
      className={`${styles.defaultRouteNode} ${isHighlighted ? styles.highlighted : ''}`}
    >
      <Handle type="target" position={Position.Left} />

      <div className={styles.defaultRouteHeader}>
        <span className={styles.decisionIcon}>↪</span>
        <span className={styles.decisionName} title={label}>{label}</span>
        <span className={styles.decisionPriority}>Default</span>
      </div>

      <div className={styles.rulesSection}>
        <div className={styles.conditionTree}>
          <div className={`${styles.conditionRow} ${styles.conditionLeafRow}`}>
            <span className={styles.conditionText}>No decision matched</span>
          </div>
        </div>
      </div>

      <div className={styles.decisionMeta}>
        <span className={styles.metaTag}>Fallback path</span>
      </div>

      <div className={styles.modelsList}>
        {defaultModel ? (
          <span className={styles.modelItem} title={defaultModel}>
            {defaultModel.split('/').pop() || defaultModel}
          </span>
        ) : (
          <span className={styles.moreModels}>No default model</span>
        )}
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  )
})

DefaultRouteNode.displayName = 'DefaultRouteNode'
