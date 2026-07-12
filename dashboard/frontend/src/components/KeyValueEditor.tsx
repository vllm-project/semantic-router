import { useId, useState } from 'react'

import styles from './StructuredFieldEditors.module.css'

interface StructuredEditorStateProps {
  disabled?: boolean
  readOnly?: boolean
}

export interface KeyValueEditorProps extends StructuredEditorStateProps {
  value: Readonly<Record<string, string>>
  onChange: (value: Record<string, string>) => void
  addLabel?: string
  emptyLabel?: string
  keyLabel?: string
  keyPlaceholder?: string
  valueLabel?: string
  valuePlaceholder?: string
}

function nextAvailableKey(value: Readonly<Record<string, string>>): string {
  let index = 1
  while (value[`key_${index}`] !== undefined) index += 1
  return `key_${index}`
}

export function KeyValueEditor({
  value,
  onChange,
  addLabel = 'Add entry',
  emptyLabel = 'No entries configured.',
  keyLabel = 'Key',
  keyPlaceholder,
  valueLabel = 'Value',
  valuePlaceholder,
  disabled = false,
  readOnly = false,
}: KeyValueEditorProps) {
  const editorId = useId()
  const [keyErrors, setKeyErrors] = useState<Record<string, string>>({})
  const entries = Object.entries(value).filter(
    (entry): entry is [string, string] => typeof entry[1] === 'string',
  )

  const renameKey = (currentKey: string, nextKey: string) => {
    const trimmedKey = nextKey.trim()
    if (!trimmedKey) {
      setKeyErrors((current) => ({ ...current, [currentKey]: `${keyLabel} cannot be empty.` }))
      return
    }
    if (trimmedKey !== currentKey && value[trimmedKey] !== undefined) {
      setKeyErrors((current) => ({ ...current, [currentKey]: `${keyLabel} must be unique.` }))
      return
    }

    const nextEntries = entries.map(([key, entryValue]) =>
      key === currentKey ? [trimmedKey, entryValue] : [key, entryValue],
    )
    setKeyErrors((current) => {
      const next = { ...current }
      delete next[currentKey]
      return next
    })
    onChange(Object.fromEntries(nextEntries))
  }

  if (readOnly) {
    return entries.length > 0 ? (
      <dl className={styles.keyValueReadOnly}>
        {entries.map(([key, entryValue]) => (
          <div key={key} className={styles.keyValueReadOnlyRow}>
            <dt>{key}</dt>
            <dd>{entryValue}</dd>
          </div>
        ))}
      </dl>
    ) : (
      <p className={styles.emptyCompact}>{emptyLabel}</p>
    )
  }

  return (
    <div className={styles.nestedEditor}>
      <div className={styles.rows}>
        {entries.map(([key, entryValue], index) => {
          const keyId = `${editorId}-key-${index}`
          const valueId = `${editorId}-value-${index}`
          const errorId = `${keyId}-error`
          const keyError = keyErrors[key]

          return (
            <div key={index} className={styles.keyValueRow}>
              <div className={styles.inputStack}>
                <label className={styles.miniLabel} htmlFor={keyId}>
                  {keyLabel}
                </label>
                <input
                  id={keyId}
                  className={styles.input}
                  value={key}
                  onChange={(event) => renameKey(key, event.target.value)}
                  placeholder={keyPlaceholder}
                  disabled={disabled}
                  aria-invalid={keyError ? 'true' : undefined}
                  aria-describedby={keyError ? errorId : undefined}
                />
                {keyError ? (
                  <span id={errorId} className={styles.inlineError}>
                    {keyError}
                  </span>
                ) : null}
              </div>
              <div className={styles.inputStack}>
                <label className={styles.miniLabel} htmlFor={valueId}>
                  {valueLabel}
                </label>
                <input
                  id={valueId}
                  className={styles.input}
                  value={entryValue}
                  onChange={(event) => onChange({ ...value, [key]: event.target.value })}
                  placeholder={valuePlaceholder}
                  disabled={disabled}
                />
              </div>
              <button
                type="button"
                className={styles.iconRemoveButton}
                onClick={() =>
                  onChange(Object.fromEntries(entries.filter(([entryKey]) => entryKey !== key)))
                }
                disabled={disabled}
                aria-label={`Remove ${keyLabel.toLocaleLowerCase()} ${key}`}
              >
                ×
              </button>
            </div>
          )
        })}
      </div>

      {entries.length === 0 ? <p className={styles.emptyCompact}>{emptyLabel}</p> : null}
      <button
        type="button"
        className={styles.addButtonSmall}
        onClick={() => onChange({ ...value, [nextAvailableKey(value)]: '' })}
        disabled={disabled}
      >
        <span aria-hidden="true">+</span>
        {addLabel}
      </button>
    </div>
  )
}
