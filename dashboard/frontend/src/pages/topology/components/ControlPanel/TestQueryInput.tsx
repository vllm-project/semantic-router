// ControlPanel/TestQueryInput.tsx - Test query input (always uses backend verification)

import React from 'react'
import styles from './ControlPanel.module.css'

interface TestQueryInputProps {
  value: string
  onChange: (value: string) => void
  onTest: () => void
  isLoading: boolean
  /** When set, the selected scope cannot be run (e.g. a recipe with no
   * entrypoint) — the send action stays disabled and this explains why. */
  disabledReason?: string
}

export const TestQueryInput: React.FC<TestQueryInputProps> = ({
  value,
  onChange,
  onTest,
  isLoading,
  disabledReason,
}) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.metaKey) {
      onTest()
    }
  }

  return (
    <div className={styles.section}>
      <div className={styles.testQueryHeader}>
        <span className={styles.testQueryTitle}>Send Query</span>
      </div>

      <div className={styles.inputGroup}>
        <textarea
          className={styles.queryInput}
          placeholder="Message..."
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <button
          className={styles.testBtn}
          onClick={onTest}
          disabled={isLoading || !value.trim() || Boolean(disabledReason)}
          title={disabledReason}
        >
          {isLoading ? 'Testing...' : 'Send'}
        </button>
      </div>
      {disabledReason && (
        <p className={styles.testQueryDisabledNote} role="note">
          {disabledReason}
        </p>
      )}
    </div>
  )
}
