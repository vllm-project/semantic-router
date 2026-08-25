// ResultCard.tsx - Centered result dialog for routing previews

import React, { useId } from 'react'
import { TestQueryResult, SignalType } from '../../types'
import { SIGNAL_COLORS } from '../../constants'
import ProductIcon from '../../../../components/ProductIcon'
import useAccessibleDialog from '../../../../hooks/useAccessibleDialog'
import styles from './ResultCard.module.css'

interface ResultCardProps {
  result: TestQueryResult | null
  onClose: () => void
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, onClose }) => {
  const titleId = useId()
  const dialogRef = useAccessibleDialog<HTMLElement>({ isOpen: Boolean(result), onClose })
  if (!result) return null

  const matchedSignals = result.matchedSignals.filter((signal) => signal.matched)

  const getSignalColor = (type: SignalType): string => {
    return SIGNAL_COLORS[type]?.background || '#607D8B'
  }

  const formatValue = (value: number): string => {
    if (Number.isInteger(value)) return `${value}`
    if (Math.abs(value) >= 1) return value.toFixed(2)
    return value.toFixed(3)
  }

  const formatScore = (score: number): string => `${Math.round(score * 100)}%`

  return (
    <div className={styles.overlay} role="presentation" onMouseDown={onClose}>
      <section
        ref={dialogRef}
        className={styles.card}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        data-testid="topology-result-dialog"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className={styles.header} data-testid="topology-result-header">
          <span className={styles.headerIcon} aria-hidden="true">
            <ProductIcon name="topology" />
          </span>
          <span className={styles.title}>
            <small>Path preview</small>
            <strong id={titleId}>Routing result</strong>
          </span>
          {result.routingLatency !== undefined && (
            <span className={styles.latencyBadge}>{result.routingLatency}ms</span>
          )}
          <button
            type="button"
            className={styles.closeBtn}
            onClick={onClose}
            aria-label="Close routing result"
            data-dialog-initial-focus
          >
            <ProductIcon name="close" aria-hidden="true" />
          </button>
        </header>

        <div className={styles.scrollBody} data-testid="topology-result-scroll">
          {result.warning && (
            <div className={styles.warningBanner}>
              <ProductIcon name="info" aria-hidden="true" />
              <span>{result.warning}</span>
            </div>
          )}

          <div className={styles.content}>
            <div className={styles.compactRow}>
              <div className={styles.compactItem}>
                <span className={styles.label}>Decision</span>
                <span className={styles.value}>{result.matchedDecision || 'Not resolved'}</span>
              </div>
              <div className={styles.compactItem}>
                <span className={styles.label}>Model</span>
                <span className={styles.value}>
                  {result.matchedModels[0]?.split('/').pop() || 'N/A'}
                </span>
              </div>
            </div>

            {matchedSignals.length > 0 && (
              <div className={styles.section}>
                <span className={styles.sectionTitle}>Matched signals</span>
                <div className={styles.signalList}>
                  {matchedSignals.map((signal) => (
                    <div key={`${signal.type}-${signal.name}`} className={styles.signalCard}>
                      <div className={styles.signalCardHeader}>
                        <span
                          className={styles.signalTag}
                          style={{ borderColor: getSignalColor(signal.type) }}
                        >
                          <ProductIcon name="signal" aria-hidden="true" />
                          {signal.name}
                        </span>
                        <span className={styles.signalType}>{signal.type}</span>
                      </div>
                      <div className={styles.signalMeta}>
                        {signal.value !== undefined && (
                          <span className={styles.signalMetric}>
                            Value {formatValue(signal.value)}
                          </span>
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

            {result.isFallbackDecision && result.fallbackReason && (
              <div className={styles.fallbackReason}>
                <ProductIcon name="info" aria-hidden="true" />
                {result.fallbackReason}
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
