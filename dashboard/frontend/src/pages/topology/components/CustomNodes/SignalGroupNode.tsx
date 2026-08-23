// CustomNodes/SignalGroupNode.tsx - Signal group node with collapse support

import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { SignalType, SignalConfig } from '../../types'
import { SIGNAL_COLORS, SIGNAL_LATENCY } from '../../constants'
import ProductIcon from '../../../../components/ProductIcon'
import styles from './CustomNodes.module.css'

interface SignalGroupNodeData {
  signalType: SignalType
  signals: SignalConfig[]
  collapsed?: boolean
  isHighlighted?: boolean
  isDynamic?: boolean // True if signals were detected dynamically (not from config)
  title?: string
  subtitle?: string
  latencyLabel?: string
  onToggleCollapse?: () => void
}

export const SignalGroupNode = memo<NodeProps<SignalGroupNodeData>>(({ data }) => {
  const {
    signalType,
    signals,
    collapsed = false,
    isHighlighted,
    isDynamic = false,
    title,
    subtitle,
    latencyLabel,
    onToggleCollapse,
  } = data
  const color = SIGNAL_COLORS[signalType]
  const latency = latencyLabel || SIGNAL_LATENCY[signalType]

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
      <Handle type="target" position={Position.Left} />

      <div className={styles.signalGroupHeader}>
        <ProductIcon className={styles.signalGroupIcon} name="signal" aria-hidden="true" />
        <span className={styles.signalGroupTitle}>{title || signalType.replace('_', ' ')}</span>
        <span className={styles.signalGroupBadge}>{signals.length}</span>
        {isDynamic && <span className={styles.dynamicBadge}>ML</span>}
      </div>

      <div className={styles.signalGroupContent}>
        {subtitle ? <div className={styles.signalGroupSubtitle}>{subtitle}</div> : null}
        <div className={styles.signalLatency}>
          <ProductIcon name="activity" aria-hidden="true" />
          <span>{latency}</span>
          <ProductIcon
            className={`${styles.collapseIcon} ${collapsed ? '' : styles.collapseIconOpen}`}
            name="chevron-right"
            aria-hidden="true"
          />
        </div>

        {!collapsed && signals.length > 0 && (
          <div className={styles.signalList}>
            {signals.slice(0, 5).map((signal) => (
              <div key={signal.name} className={styles.signalItem}>
                {signal.name}
                {(signal as SignalConfig & { isDynamic?: boolean }).isDynamic && (
                  <ProductIcon className={styles.mlTag} name="compute" aria-label="ML detected" />
                )}
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

      <Handle type="source" position={Position.Right} />
    </div>
  )
})

SignalGroupNode.displayName = 'SignalGroupNode'
