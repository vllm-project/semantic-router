import { useId, useState } from 'react'

import { KeyValueEditor } from './KeyValueEditor'
import styles from './StructuredFieldEditors.module.css'

interface StructuredEditorStateProps {
  disabled?: boolean
  readOnly?: boolean
}

export type ObjectEditorFieldType = 'text' | 'number' | 'select' | 'password' | 'key-value'

export interface ObjectEditorField<TItem extends object> {
  key: Extract<keyof TItem, string>
  label: string
  type?: ObjectEditorFieldType
  placeholder?: string
  options?: readonly string[]
  required?: boolean
  min?: number
  max?: number
  step?: number
  fullWidth?: boolean
  helpText?: string
  emptyValueLabel?: string
  shouldHide?: (item: TItem) => boolean
}

export interface ObjectListEditorProps<TItem extends object> extends StructuredEditorStateProps {
  value: readonly TItem[]
  onChange: (value: TItem[]) => void
  fields: readonly ObjectEditorField<TItem>[]
  createItem: (index: number) => TItem
  addLabel?: string
  emptyLabel?: string
  itemLabel?: (item: TItem, index: number) => string
  itemDescription?: (item: TItem, index: number) => string | undefined
  validateItem?: (item: TItem, index: number) => string[]
  minItems?: number
  maxItems?: number
}

function itemRecord<TItem extends object>(item: TItem): Record<string, unknown> {
  return item as Record<string, unknown>
}

function fieldStringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function fieldNumberValue(value: unknown): number | '' {
  return typeof value === 'number' && Number.isFinite(value) ? value : ''
}

export function ObjectListEditor<TItem extends object>({
  value,
  onChange,
  fields,
  createItem,
  addLabel = 'Add item',
  emptyLabel = 'No items configured.',
  itemLabel = (_item, index) => `Item ${index + 1}`,
  itemDescription,
  validateItem,
  minItems = 0,
  maxItems,
  disabled = false,
  readOnly = false,
}: ObjectListEditorProps<TItem>) {
  const editorId = useId()
  const [editingIndex, setEditingIndex] = useState<number | null>(value.length === 1 ? 0 : null)

  const updateItem = (index: number, field: ObjectEditorField<TItem>, nextValue: unknown) => {
    const nextItems = [...value]
    const nextItem = { ...nextItems[index] } as TItem
    const record = itemRecord(nextItem)
    if (nextValue === undefined || nextValue === '') delete record[field.key]
    else record[field.key] = nextValue
    nextItems[index] = nextItem
    onChange(nextItems)
  }

  const removeItem = (index: number) => {
    if (value.length <= minItems) return
    onChange(value.filter((_, itemIndex) => itemIndex !== index))
    setEditingIndex((current) => {
      if (current === null) return null
      if (current === index) return null
      return current > index ? current - 1 : current
    })
  }

  return (
    <div className={styles.editor}>
      <div className={styles.objectList}>
        {value.map((item, index) => {
          const record = itemRecord(item)
          const errors = validateItem?.(item, index) ?? []
          const isExpanded = readOnly || editingIndex === index
          const description = itemDescription?.(item, index)

          return (
            <section
              key={index}
              className={`${styles.objectCard} ${errors.length > 0 ? styles.objectCardInvalid : ''}`}
              aria-labelledby={`${editorId}-item-${index}`}
            >
              <div className={styles.objectCardHeader}>
                <div className={styles.objectCardHeading}>
                  <span className={styles.itemIndex}>{String(index + 1).padStart(2, '0')}</span>
                  <div>
                    <h4 id={`${editorId}-item-${index}`}>{itemLabel(item, index)}</h4>
                    {description ? <p>{description}</p> : null}
                  </div>
                </div>
                {!readOnly ? (
                  <div className={styles.cardActions}>
                    <button
                      type="button"
                      className={styles.secondaryButton}
                      onClick={() => setEditingIndex(isExpanded ? null : index)}
                      disabled={disabled}
                      aria-expanded={isExpanded}
                    >
                      {isExpanded ? 'Done' : 'Edit'}
                    </button>
                    {value.length > minItems ? (
                      <button
                        type="button"
                        className={styles.removeButton}
                        onClick={() => removeItem(index)}
                        disabled={disabled}
                      >
                        Remove
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </div>

              {errors.length > 0 ? (
                <ul className={styles.validationList} aria-live="polite">
                  {errors.map((error) => (
                    <li key={error}>{error}</li>
                  ))}
                </ul>
              ) : null}

              {isExpanded ? (
                <div className={styles.objectGrid}>
                  {fields.map((field) => {
                    if (field.shouldHide?.(item)) return null
                    const fieldValue = record[field.key]
                    if (
                      readOnly &&
                      (fieldValue === undefined || fieldValue === '' || fieldValue === null)
                    )
                      return null
                    const fieldId = `${editorId}-${index}-${field.key}`
                    const fieldClassName = field.fullWidth
                      ? styles.objectFieldWide
                      : styles.objectField
                    const requiredError =
                      field.required &&
                      (fieldValue === undefined || fieldValue === null || fieldValue === '')

                    return (
                      <div key={field.key} className={fieldClassName}>
                        <label
                          className={styles.miniLabel}
                          htmlFor={field.type === 'key-value' ? undefined : fieldId}
                        >
                          {field.label}
                          {field.required ? <span className={styles.required}> *</span> : null}
                        </label>
                        {field.helpText ? (
                          <p className={styles.helpText}>{field.helpText}</p>
                        ) : null}

                        {field.type === 'key-value' ? (
                          <KeyValueEditor
                            value={
                              fieldValue &&
                              typeof fieldValue === 'object' &&
                              !Array.isArray(fieldValue)
                                ? (fieldValue as Record<string, string>)
                                : {}
                            }
                            onChange={(nextValue) => updateItem(index, field, nextValue)}
                            emptyLabel={field.emptyValueLabel}
                            keyLabel="Header"
                            keyPlaceholder="X-Header"
                            valueLabel="Value"
                            valuePlaceholder="Header value"
                            disabled={disabled}
                            readOnly={readOnly}
                          />
                        ) : field.type === 'select' ? (
                          <select
                            id={fieldId}
                            className={styles.input}
                            value={fieldStringValue(fieldValue)}
                            onChange={(event) =>
                              updateItem(index, field, event.target.value || undefined)
                            }
                            disabled={disabled || readOnly}
                            aria-invalid={requiredError ? 'true' : undefined}
                          >
                            <option value="">Not set</option>
                            {field.options?.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </select>
                        ) : field.type === 'number' ? (
                          <input
                            id={fieldId}
                            className={styles.input}
                            type="number"
                            value={fieldNumberValue(fieldValue)}
                            onChange={(event) =>
                              updateItem(
                                index,
                                field,
                                event.target.value === '' ? undefined : Number(event.target.value),
                              )
                            }
                            placeholder={field.placeholder}
                            min={field.min}
                            max={field.max}
                            step={field.step ?? 'any'}
                            disabled={disabled}
                            readOnly={readOnly}
                          />
                        ) : (
                          <input
                            id={fieldId}
                            className={styles.input}
                            type={field.type === 'password' ? 'password' : 'text'}
                            value={
                              readOnly && field.type === 'password' && fieldValue
                                ? '••••••••'
                                : fieldStringValue(fieldValue)
                            }
                            onChange={(event) =>
                              updateItem(index, field, event.target.value || undefined)
                            }
                            placeholder={field.placeholder}
                            disabled={disabled}
                            readOnly={readOnly}
                            aria-invalid={requiredError ? 'true' : undefined}
                          />
                        )}
                      </div>
                    )
                  })}
                </div>
              ) : null}
            </section>
          )
        })}
      </div>

      {value.length === 0 ? <p className={styles.empty}>{emptyLabel}</p> : null}
      {!readOnly && (maxItems === undefined || value.length < maxItems) ? (
        <button
          type="button"
          className={styles.addButton}
          onClick={() => {
            const nextIndex = value.length
            onChange([...value, createItem(nextIndex)])
            setEditingIndex(nextIndex)
          }}
          disabled={disabled}
        >
          <span aria-hidden="true">+</span>
          {addLabel}
        </button>
      ) : null}
    </div>
  )
}
