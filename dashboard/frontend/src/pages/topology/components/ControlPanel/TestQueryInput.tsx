// ControlPanel/TestQueryInput.tsx - Local topology preview input

import React, { type Ref } from 'react'
import styles from './ControlPanel.module.css'

interface TestQueryInputProps {
  value: string
  onChange: (value: string) => void
  onTest: () => void
  isLoading: boolean
  previewButtonRef?: Ref<HTMLButtonElement>
}

export const TestQueryInput: React.FC<TestQueryInputProps> = ({
  value,
  onChange,
  onTest,
  isLoading,
  previewButtonRef,
}) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.metaKey) {
      onTest()
    }
  }

  return (
    <div className={styles.section}>
      <div className={styles.testQueryHeader}>
        <span className={styles.testQueryTitle}>Preview path</span>
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
          ref={previewButtonRef}
          className={styles.testBtn}
          onClick={onTest}
          disabled={isLoading || !value.trim()}
        >
          {isLoading ? 'Previewing…' : 'Preview'}
        </button>
      </div>
    </div>
  )
}
