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
    if (algorithm.type === 'confidence' && algorithm.confidence) {
      return `threshold: ${algorithm.confidence.threshold}`
    }
    if (algorithm.type === 'concurrent' && algorithm.concurrent) {
      return `timeout: ${algorithm.concurrent.timeout_seconds}s`
    }
    if (algorithm.type === 'latency_aware' && algorithm.latency_aware) {
      const parts: string[] = []
      if (algorithm.latency_aware.tpot_percentile) {
        parts.push(`TPOT P${algorithm.latency_aware.tpot_percentile}`)
      }
      if (algorithm.latency_aware.ttft_percentile) {
        parts.push(`TTFT P${algorithm.latency_aware.ttft_percentile}`)
      }
      return parts.length > 0 ? parts.join(', ') : null
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
      <Handle type="target" position={Position.Top} />

      <div className={styles.algorithmHeader}>
        <span className={styles.algorithmIcon}>{icon}</span>
        <span className={styles.algorithmType}>{algorithm.type}</span>
      </div>

      {configDisplay && (
        <div className={styles.algorithmConfig}>
          {configDisplay}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
})

AlgorithmNode.displayName = 'AlgorithmNode'
