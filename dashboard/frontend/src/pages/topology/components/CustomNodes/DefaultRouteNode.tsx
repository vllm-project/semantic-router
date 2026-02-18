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
      <div className={styles.defaultRouteHeader}>
        <span className={styles.defaultRouteIcon}>🔄</span>
        <span className={styles.defaultRouteTitle}>{label}</span>
      </div>
      <div className={styles.defaultRouteInfo}>
        <span className={styles.defaultRouteLabel}>Fallback</span>
        {defaultModel && (
          <span className={styles.defaultRouteModel} title={defaultModel}>
            → {defaultModel.length > 15 ? defaultModel.slice(0, 15) + '...' : defaultModel}
          </span>
        )}
      </div>
      <Handle type="target" position={Position.Top} />
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
})

DefaultRouteNode.displayName = 'DefaultRouteNode'
