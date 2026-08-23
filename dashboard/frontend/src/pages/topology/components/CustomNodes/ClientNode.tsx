// CustomNodes/ClientNode.tsx - Client entry node

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import styles from './CustomNodes.module.css'
import ProductIcon from '../../../../components/ProductIcon'

interface ClientNodeData {
  label?: string
  isHighlighted?: boolean
}

export const ClientNode = memo<NodeProps<ClientNodeData>>(({ data }) => {
  const { label = 'User Query', isHighlighted } = data

  return (
    <div className={`${styles.clientNode} ${isHighlighted ? styles.highlighted : ''}`}>
      <ProductIcon className={styles.clientIcon} name="user" aria-hidden="true" />
      <span className={styles.clientLabel}>{label}</span>
      <Handle type="source" position={Position.Right} />
    </div>
  )
})

ClientNode.displayName = 'ClientNode'
