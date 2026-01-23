// CustomNodes/ClientNode.tsx - Client entry node

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import styles from './CustomNodes.module.css'

interface ClientNodeData {
  label?: string
  isHighlighted?: boolean
}

export const ClientNode = memo<NodeProps<ClientNodeData>>(({ data }) => {
  const { label = 'User Query', isHighlighted } = data

  return (
    <div
      className={`${styles.clientNode} ${isHighlighted ? styles.highlighted : ''}`}
    >
      <span className={styles.clientIcon}>👤</span>
      <span className={styles.clientLabel}>{label}</span>
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
})

ClientNode.displayName = 'ClientNode'
