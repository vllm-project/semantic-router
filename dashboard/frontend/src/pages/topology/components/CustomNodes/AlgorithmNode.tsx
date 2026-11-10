// CustomNodes/AlgorithmNode.tsx - Algorithm node

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { AlgorithmConfig } from '../../types'
import { ALGORITHM_ICONS, ALGORITHM_COLORS } from '../../constants'
import styles from './CustomNodes.module.css'

interface AlgorithmNodeData {
  algorithm: AlgorithmConfig
  decisionName: string
  isHighlighted?: boolean
}

export const AlgorithmNode = memo<NodeProps<AlgorithmNodeData>>(({ data }) => {
  const { algorithm, isHighlighted } = data
  const colors = ALGORITHM_COLORS[algorithm.type] || { background: '#607D8B', border: '#455A64' }
  const icon = ALGORITHM_ICONS[algorithm.type] || '🔄'

  // Get algorithm-specific config display
  const getConfigDisplay = () => {
    const confidence = algorithm.escalate?.accept?.confidence
    if (algorithm.type === 'escalate' && confidence) {
      return `gte: ${confidence.gte}`
    }
    if (algorithm.type === 'multi_factor' && algorithm.multi_factor) {
      const latencyPercentile = algorithm.multi_factor.latency_percentile
      return typeof latencyPercentile === 'number' ? `P${latencyPercentile} latency` : null
    }
    return null
  }

  const configDisplay = getConfigDisplay()

  return (
    <div
      className={`${styles.algorithmNode} ${isHighlighted ? styles.highlighted : ''}`}
      style={{
        background: colors.background,
        border: `2px solid ${colors.border}`,
      }}
    >
      <Handle type="target" position={Position.Left} />

      <div className={styles.algorithmHeader}>
        <span className={styles.algorithmIcon}>{icon}</span>
        <span className={styles.algorithmType}>{algorithm.type}</span>
      </div>

      {configDisplay && <div className={styles.algorithmConfig}>{configDisplay}</div>}

      <Handle type="source" position={Position.Right} />
    </div>
  )
})

AlgorithmNode.displayName = 'AlgorithmNode'
