// ControlPanel/TestQueryInput.tsx - Test query input (always uses backend verification)

import React from 'react'
import { TestQueryResult, SignalType } from '../../types'
import { SIGNAL_COLORS, SIGNAL_ICONS } from '../../constants'
import styles from './ControlPanel.module.css'

interface TestQueryInputProps {
  value: string
  onChange: (value: string) => void
  onTest: () => void
  isLoading: boolean
  result: TestQueryResult | null
}

export const TestQueryInput: React.FC<TestQueryInputProps> = ({
  value,
  onChange,
  onTest,
  isLoading,
  result,
}) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.metaKey) {
      onTest()
    }
  }

  const getSignalColor = (type: SignalType): string => {
    return SIGNAL_COLORS[type]?.background || '#607D8B'
  }

  const getSignalIcon = (type: SignalType): string => {
    return SIGNAL_ICONS[type] || '❓'
  }

  return (
    <div className={styles.section}>
      <div className={styles.testQueryHeader}>
        <span className={styles.testQueryTitle}>Send Query</span>
      </div>

      <div className={styles.inputGroup}>
        <textarea
          className={styles.queryInput}
          placeholder="Enter a test query to verify routing... (⌘+Enter to test)"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={3}
        />
        <button
          className={styles.testBtn}
          onClick={onTest}
          disabled={isLoading || !value.trim()}
        >
          {isLoading ? 'Testing...' : 'Verify'}
        </button>
      </div>

      {/* Result Display */}
      {result && (
        <div className={styles.testResult}>
          <div className={styles.resultHeader}>
            <span className={styles.resultTitle}>📊 Routing Result</span>
            <span className={styles.accurateBadge}>
              ✅ Verified
            </span>
            {result.routingLatency !== undefined && (
              <span className={styles.latencyBadge}>{result.routingLatency}ms</span>
            )}
          </div>

          {/* Warning Banner */}
          {result.warning && (
            <div className={styles.warningBanner}>
              <span>⚠️</span>
              <span>{result.warning}</span>
            </div>
          )}

          {/* Matched Signals */}
          <div className={styles.resultSection}>
            <span className={styles.resultSectionTitle}>Matched Signals:</span>
            <div className={styles.signalTags}>
              {result.matchedSignals.filter(s => s.matched).map(signal => (
                <span
                  key={`${signal.type}-${signal.name}`}
                  className={styles.signalTag}
                  style={{ background: getSignalColor(signal.type) }}
                >
                  {getSignalIcon(signal.type)} {signal.name}
                  {signal.score !== undefined && (
                    <span className={styles.signalScore}>
                      {(signal.score * 100).toFixed(0)}%
                    </span>
                  )}
                </span>
              ))}

              {result.matchedSignals.filter(s => s.matched).length === 0 && (
                <span className={styles.noMatch}>No signals matched</span>
              )}
            </div>
          </div>

          {/* Selected Decision */}
          <div className={styles.resultSection}>
            <span className={styles.resultSectionTitle}>Selected Decision: </span>
            <span className={`${styles.decisionResult} ${result.isFallbackDecision ? styles.fallbackDecision : ''}`}>
              {result.isFallbackDecision && <span className={styles.fallbackBadge}>⚠️ System Fallback</span>}
              {result.matchedDecision || 'Default (no match)'}
            </span>
            {result.isFallbackDecision && result.fallbackReason && (
              <div className={styles.fallbackReason}>
                💡 {result.fallbackReason}
              </div>
            )}
          </div>

          {/* Target Models */}
          <div className={styles.resultSection}>
            <span className={styles.resultSectionTitle}>Target Model(s):</span>
            <div className={styles.modelTags}>
              {result.matchedModels.map(model => (
                <span key={model} className={styles.modelTag}>
                  🤖 {model.split('/').pop()}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
