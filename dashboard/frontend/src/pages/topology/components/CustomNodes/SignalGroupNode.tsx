// CustomNodes/SignalGroupNode.tsx - Signal group node with collapse support

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { SignalType, SignalConfig } from '../../types'
import { SIGNAL_ICONS, SIGNAL_COLORS, SIGNAL_LATENCY } from '../../constants'
import styles from './CustomNodes.module.css'

interface SignalGroupNodeData {
  signalType: SignalType
  signals: SignalConfig[]
  collapsed?: boolean
  isHighlighted?: boolean
  isDynamic?: boolean // True if signals were detected dynamically (not from config)
  onToggleCollapse?: () => void
}

export const SignalGroupNode = memo<NodeProps<SignalGroupNodeData>>(({ data }) => {
  const { signalType, signals, collapsed = false, isHighlighted, isDynamic = false, onToggleCollapse } = data
  const color = SIGNAL_COLORS[signalType]
  const icon = SIGNAL_ICONS[signalType]
  const latency = SIGNAL_LATENCY[signalType]

  return (
    <div
      className={`${styles.signalGroupNode} ${isHighlighted ? styles.highlighted : ''} ${isDynamic ? styles.dynamicSignal : ''}`}
      style={{
        background: color.background,
        border: `2px ${isDynamic ? 'dashed' : 'solid'} ${color.border}`,
      }}
      onClick={onToggleCollapse}
      title={isDynamic ? 'Detected by ML model (not in config)' : undefined}
    >
      <Handle type="target" position={Position.Top} />

      <div className={styles.signalGroupHeader}>
        <span className={styles.signalGroupIcon}>{icon}</span>
        <span className={styles.signalGroupTitle}>
          {signalType.replace('_', ' ')}
        </span>
        <span className={styles.signalGroupBadge}>{signals.length}</span>
        {isDynamic && <span className={styles.dynamicBadge}>ML</span>}
      </div>

      <div className={styles.signalGroupContent}>
        <div className={styles.signalLatency}>
          <span>⏱️</span>
          <span>{latency}</span>
          <span className={styles.collapseIcon}>
            {collapsed ? '▶' : '▼'}
          </span>
        </div>

        {!collapsed && signals.length > 0 && (
          <div className={styles.signalList}>
            {signals.slice(0, 5).map(signal => (
              <div key={signal.name} className={styles.signalItem}>
                {signal.name}
                {(signal as any).isDynamic && <span className={styles.mlTag}>🤖</span>}
              </div>
            ))}
            {signals.length > 5 && (
              <div className={styles.signalItem} style={{ opacity: 0.7 }}>
                +{signals.length - 5} more
              </div>
            )}
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
})

SignalGroupNode.displayName = 'SignalGroupNode'
