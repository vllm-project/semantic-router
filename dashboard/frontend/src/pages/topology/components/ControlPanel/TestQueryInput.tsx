// ControlPanel/TestQueryInput.tsx - Test query input (always uses backend verification)

import React from 'react'
import styles from './ControlPanel.module.css'

interface TestQueryInputProps {
  requestContext: string
  onRequestContextChange: (value: string) => void
  disabledReason?: string
  hasInput: boolean
  value: string
  onChange: (value: string) => void
  onTest: () => void
  isLoading: boolean
}

export const TestQueryInput: React.FC<TestQueryInputProps> = ({
  requestContext,
  onRequestContextChange,
  disabledReason,
  hasInput,
  value,
  onChange,
  onTest,
  isLoading,
}) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.metaKey && !isLoading && !disabledReason && hasInput) {
      onTest()
    }
  }

  return (
    <div className={styles.section}>
      <div className={styles.testQueryHeader}>
        <span className={styles.testQueryTitle}>Send Query</span>
      </div>

      {disabledReason && <p role="status">{disabledReason}</p>}
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
          disabled={isLoading || Boolean(disabledReason) || !hasInput}
        >
          {isLoading ? 'Testing...' : 'Send'}
        </button>
      </div>
      <details className={styles.requestContext}>
        <summary>Request context (optional)</summary>
        <label htmlFor="topology-request-context">Messages, metadata and tools (JSON)</label>
        <textarea
          id="topology-request-context"
          className={styles.contextInput}
          value={requestContext}
          onChange={(event) => onRequestContextChange(event.target.value)}
          rows={4}
          placeholder='{"metadata":{"tenant":"example"}}'
        />
      </details>
    </div>
  )
}
