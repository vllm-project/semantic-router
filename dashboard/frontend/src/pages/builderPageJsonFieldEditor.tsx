import { useEffect, useMemo, useState } from 'react'

import type { FieldSchema } from '@/lib/dslMutations'

import styles from './BuilderPage.module.css'

interface JsonFieldEditorProps {
  schema: FieldSchema
  value: unknown
  onChange: (value: unknown) => void
}

export function JsonFieldEditor({ schema, value, onChange }: JsonFieldEditorProps) {
  const serialized = useMemo(
    () => (value === undefined ? '' : JSON.stringify(value, null, 2)),
    [value],
  )
  const [draft, setDraft] = useState(serialized)
  const [error, setError] = useState('')

  useEffect(() => {
    setDraft(serialized)
    setError('')
  }, [serialized])

  const commit = () => {
    if (!draft.trim()) {
      setError('')
      onChange(undefined)
      return
    }
    try {
      onChange(JSON.parse(draft))
      setError('')
    } catch (parseError) {
      setError(parseError instanceof Error ? parseError.message : 'Invalid JSON')
    }
  }

  return (
    <div className={styles.fieldGroup}>
      <label className={styles.fieldLabel}>
        {schema.label}{' '}
        {schema.required ? <span style={{ color: 'var(--color-danger)' }}>*</span> : null}
      </label>
      <textarea
        className={styles.fieldInput}
        style={{ minHeight: '8rem', resize: 'vertical', fontFamily: 'var(--font-mono)' }}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        aria-invalid={Boolean(error)}
        placeholder={schema.placeholder || '{}'}
      />
      {schema.description ? (
        <span style={{ fontSize: '0.625rem', color: 'var(--color-text-muted)' }}>
          {schema.description}
        </span>
      ) : null}
      {error ? (
        <span style={{ color: 'var(--color-danger)', fontSize: '0.625rem' }}>{error}</span>
      ) : null}
    </div>
  )
}
