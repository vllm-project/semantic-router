// ResultCard.tsx - Floating result card for routing results

import React from 'react'
import { TestQueryResult, SignalType } from '../../types'
import { SIGNAL_COLORS, SIGNAL_ICONS } from '../../constants'
import styles from './ResultCard.module.css'

interface ResultCardProps {
  result: TestQueryResult | null
  onClose: () => void
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, onClose }) => {
  if (!result) return null

  const matchedSignals = result.matchedSignals.filter(signal => signal.matched)

  const getSignalColor = (type: SignalType): string => {
    return SIGNAL_COLORS[type]?.background || '#607D8B'
  }

  const getSignalIcon = (type: SignalType): string => {
    return SIGNAL_ICONS[type] || '❓'
  }

  const formatValue = (value: number): string => {
    if (Number.isInteger(value)) return `${value}`
    if (Math.abs(value) >= 1) return value.toFixed(2)
    return value.toFixed(3)
  }

  const formatScore = (score: number): string => `${Math.round(score * 100)}%`

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.card} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className={styles.header}>
          <span className={styles.title}>📊 Routing</span>
          {result.routingLatency !== undefined && (
            <span className={styles.latencyBadge}>{result.routingLatency}ms</span>
          )}
          <button className={styles.closeBtn} onClick={onClose}>✕</button>
        </div>

        {/* Warning Banner */}
        {result.warning && (
          <div className={styles.warningBanner}>
            <span>⚠️ {result.warning}</span>
          </div>
        )}

        {/* Content */}
        <div className={styles.content}>
          {/* Decision & Model in one row */}
          <div className={styles.compactRow}>
            <div className={styles.compactItem}>
              <span className={styles.label}>Decision:</span>
              <span className={styles.value}>
                {result.matchedDecision || 'Default'}
              </span>
            </div>
            <div className={styles.compactItem}>
              <span className={styles.label}>Model:</span>
              <span className={styles.value}>
                {result.matchedModels[0]?.split('/').pop() || 'N/A'}
              </span>
            </div>
          </div>

          {/* Matched Signals */}
          {matchedSignals.length > 0 && (
            <div className={styles.section}>
              <span className={styles.sectionTitle}>Signals:</span>
              <div className={styles.signalList}>
                {matchedSignals.map(signal => (
                  <div
                    key={`${signal.type}-${signal.name}`}
                    className={styles.signalCard}
                  >
                    <div className={styles.signalCardHeader}>
                      <span
                        className={styles.signalTag}
                        style={{ background: getSignalColor(signal.type) }}
                      >
                        {getSignalIcon(signal.type)} {signal.name}
                      </span>
                      <span className={styles.signalType}>{signal.type}</span>
                    </div>
                    <div className={styles.signalMeta}>
                      {signal.value !== undefined && (
                        <span className={styles.signalMetric}>Value {formatValue(signal.value)}</span>
                      )}
                      {(signal.score ?? signal.confidence) !== undefined && (
                        <span className={styles.signalMetric}>
                          Score {formatScore(signal.score ?? signal.confidence ?? 0)}
                        </span>
                      )}
                    </div>
                    {signal.reason && <div className={styles.signalReason}>{signal.reason}</div>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Fallback Reason */}
          {result.isFallbackDecision && result.fallbackReason && (
            <div className={styles.fallbackReason}>
              💡 {result.fallbackReason}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
