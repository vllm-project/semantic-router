import { useId } from 'react'

import styles from './StructuredFieldEditors.module.css'

interface StructuredEditorStateProps {
  disabled?: boolean
  readOnly?: boolean
}

export interface StringListEditorProps extends StructuredEditorStateProps {
  value: readonly string[]
  onChange: (value: string[]) => void
  addLabel?: string
  emptyLabel?: string
  itemLabel?: string
  maxItems?: number
  placeholder?: string
  validateItem?: (value: string, index: number) => string | null
}

export function StringListEditor({
  value,
  onChange,
  addLabel = 'Add value',
  emptyLabel = 'No values configured.',
  itemLabel = 'Value',
  maxItems,
  placeholder,
  validateItem,
  disabled = false,
  readOnly = false,
}: StringListEditorProps) {
  const editorId = useId()
  const normalizedValue = value.filter((item): item is string => typeof item === 'string')
  const normalizedCounts = normalizedValue.reduce<Map<string, number>>((counts, item) => {
    const normalized = item.trim().toLocaleLowerCase()
    if (normalized) counts.set(normalized, (counts.get(normalized) ?? 0) + 1)
    return counts
  }, new Map())
  const canAdd =
    !disabled && !readOnly && (maxItems === undefined || normalizedValue.length < maxItems)

  if (readOnly) {
    return normalizedValue.length > 0 ? (
      <div className={styles.chipList} aria-label={`${itemLabel} values`}>
        {normalizedValue.map((item, index) => (
          <span key={`${item}-${index}`} className={styles.chip}>
            {item}
          </span>
        ))}
      </div>
    ) : (
      <p className={styles.emptyCompact}>{emptyLabel}</p>
    )
  }

  return (
    <div className={styles.editor}>
      <div className={styles.rows}>
        {normalizedValue.map((item, index) => {
          const normalized = item.trim().toLocaleLowerCase()
          const isEmpty = !item.trim()
          const isDuplicate = Boolean(normalized && (normalizedCounts.get(normalized) ?? 0) > 1)
          const structuralError = isEmpty
            ? `${itemLabel} cannot be empty.`
            : isDuplicate
              ? `${itemLabel} must be unique.`
              : null
          const error = structuralError ?? validateItem?.(item, index) ?? null
          const inputId = `${editorId}-${index}`
          const errorId = `${inputId}-error`

          return (
            <div key={index} className={styles.stringRow}>
              <div className={styles.inputStack}>
                <label className={styles.visuallyHidden} htmlFor={inputId}>
                  {itemLabel} {index + 1}
                </label>
                <input
                  id={inputId}
                  className={styles.input}
                  value={item}
                  onChange={(event) => {
                    const next = [...normalizedValue]
                    next[index] = event.target.value
                    onChange(next)
                  }}
                  placeholder={placeholder}
                  disabled={disabled}
                  aria-invalid={error ? 'true' : undefined}
                  aria-describedby={error ? errorId : undefined}
                />
                {error ? (
                  <span id={errorId} className={styles.inlineError}>
                    {error}
                  </span>
                ) : null}
              </div>
              <button
                type="button"
                className={styles.removeButton}
                onClick={() =>
                  onChange(normalizedValue.filter((_, itemIndex) => itemIndex !== index))
                }
                disabled={disabled}
                aria-label={`Remove ${itemLabel.toLocaleLowerCase()} ${index + 1}`}
              >
                Remove
              </button>
            </div>
          )
        })}
      </div>

      {normalizedValue.length === 0 ? <p className={styles.empty}>{emptyLabel}</p> : null}
      <button
        type="button"
        className={styles.addButton}
        onClick={() => onChange([...normalizedValue, ''])}
        disabled={!canAdd}
      >
        <span aria-hidden="true">+</span>
        {addLabel}
      </button>
    </div>
  )
}
