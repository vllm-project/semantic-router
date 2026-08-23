import React, { useCallback, useState } from 'react'

import type { DSLFieldObject } from '@/types/dsl'
import styles from './BuilderPage.module.css'

// ===================================================================
// Manual Plugin Adder (for inline plugins not in templates)
// ===================================================================

const ManualPluginAdder: React.FC<{
  existingNames: Set<string>
  onAdd: (name: string, fields?: DSLFieldObject) => void
}> = ({ existingNames, onAdd }) => {
  const [name, setName] = useState('')

  const handleAdd = useCallback(() => {
    const n = name.trim()
    if (!n || existingNames.has(n)) return
    onAdd(n)
    setName('')
  }, [name, existingNames, onAdd])

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--spacing-sm)',
        marginTop: 'var(--spacing-xs)',
      }}
    >
      <input
        className={styles.fieldInput}
        style={{ flex: 1, fontSize: 'var(--text-xs)' }}
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Add inline plugin by name..."
        onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
      />
      <button
        className={styles.toolbarBtn}
        onClick={handleAdd}
        disabled={!name.trim() || existingNames.has(name.trim())}
        style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}
      >
        + Add
      </button>
    </div>
  )
}

export { ManualPluginAdder }
