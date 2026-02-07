// ControlPanel/TestQueryInput.tsx - Test query input (always uses backend verification)

import React from 'react'
import styles from './ControlPanel.module.css'

interface TestQueryInputProps {
  value: string
  onChange: (value: string) => void
  onTest: () => void
  isLoading: boolean
}

export const TestQueryInput: React.FC<TestQueryInputProps> = ({
  value,
  onChange,
  onTest,
  isLoading,
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
          disabled={isLoading || !value.trim()}
        >
          {isLoading ? 'Testing...' : 'Send'}
        </button>
      </div>
    </div>
  )
}
