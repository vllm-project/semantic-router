import { KeyValueEditor } from './KeyValueEditor'
import type { ObjectEditorField } from './ObjectListEditorTypes'
import styles from './StructuredFieldEditors.module.css'

interface ObjectEditorFieldControlProps<TItem extends object> {
  field: ObjectEditorField<TItem>
  fieldId: string
  value: unknown
  disabled: boolean
  readOnly: boolean
  onChange: (value: unknown) => void
}

const stringValue = (value: unknown): string => (typeof value === 'string' ? value : '')

const numberValue = (value: unknown): number | '' =>
  typeof value === 'number' && Number.isFinite(value) ? value : ''

const isMissing = (value: unknown): boolean => value === undefined || value === null || value === ''

function ObjectEditorFieldControl<TItem extends object>({
  field,
  fieldId,
  value,
  disabled,
  readOnly,
  onChange,
}: ObjectEditorFieldControlProps<TItem>) {
  if (field.type === 'key-value') {
    return (
      <KeyValueEditor
        value={
          value && typeof value === 'object' && !Array.isArray(value)
            ? (value as Record<string, string>)
            : {}
        }
        onChange={onChange}
        emptyLabel={field.emptyValueLabel}
        keyLabel={field.keyLabel ?? 'Key'}
        keyPlaceholder={field.keyPlaceholder}
        valueLabel={field.valueLabel ?? 'Value'}
        valuePlaceholder={field.valuePlaceholder}
        disabled={disabled}
        readOnly={readOnly}
      />
    )
  }
  if (field.type === 'select') {
    return (
      <select
        id={fieldId}
        className={styles.input}
        value={stringValue(value)}
        onChange={(event) => onChange(event.target.value || undefined)}
        disabled={disabled || readOnly}
        aria-invalid={field.required && isMissing(value) ? 'true' : undefined}
      >
        <option value="">Not set</option>
        {field.options?.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    )
  }
  if (field.type === 'number') {
    return (
      <input
        id={fieldId}
        className={styles.input}
        type="number"
        value={numberValue(value)}
        onChange={(event) =>
          onChange(event.target.value === '' ? undefined : Number(event.target.value))
        }
        placeholder={field.placeholder}
        min={field.min}
        max={field.max}
        step={field.step ?? 'any'}
        disabled={disabled}
        readOnly={readOnly}
      />
    )
  }
  return (
    <input
      id={fieldId}
      className={styles.input}
      type={field.type === 'password' ? 'password' : 'text'}
      value={readOnly && field.type === 'password' && value ? '••••••••' : stringValue(value)}
      onChange={(event) => onChange(event.target.value || (field.preserveEmpty ? '' : undefined))}
      placeholder={field.placeholder}
      disabled={disabled}
      readOnly={readOnly}
      aria-invalid={field.required && isMissing(value) ? 'true' : undefined}
    />
  )
}

interface ObjectEditorFieldsProps<TItem extends object> {
  editorId: string
  item: TItem
  index: number
  fields: readonly ObjectEditorField<TItem>[]
  disabled: boolean
  readOnly: boolean
  onChange: (field: ObjectEditorField<TItem>, value: unknown) => void
}

function ObjectEditorFields<TItem extends object>({
  editorId,
  item,
  index,
  fields,
  disabled,
  readOnly,
  onChange,
}: ObjectEditorFieldsProps<TItem>) {
  const record = item as Record<string, unknown>
  return (
    <div className={styles.objectGrid}>
      {fields.map((field) => {
        if (field.shouldHide?.(item)) return null
        const value = record[field.key]
        if (readOnly && (value === undefined || value === '' || value === null)) return null
        const fieldId = `${editorId}-${index}-${field.key}`
        return (
          <div
            key={field.key}
            className={field.fullWidth ? styles.objectFieldWide : styles.objectField}
          >
            <label
              className={styles.miniLabel}
              htmlFor={field.type === 'key-value' ? undefined : fieldId}
            >
              {field.label}
              {field.required ? <span className={styles.required}> *</span> : null}
            </label>
            {field.helpText ? <p className={styles.helpText}>{field.helpText}</p> : null}
            <ObjectEditorFieldControl
              field={field}
              fieldId={fieldId}
              value={value}
              disabled={disabled}
              readOnly={readOnly}
              onChange={(nextValue) => onChange(field, nextValue)}
            />
          </div>
        )
      })}
    </div>
  )
}

interface ObjectListEditorItemProps<TItem extends object> {
  editorId: string
  item: TItem
  index: number
  fields: readonly ObjectEditorField<TItem>[]
  label: string
  description?: string
  errors: string[]
  expanded: boolean
  removable: boolean
  disabled: boolean
  readOnly: boolean
  onToggle: () => void
  onRemove: () => void
  onChange: (field: ObjectEditorField<TItem>, value: unknown) => void
}

export function ObjectListEditorItem<TItem extends object>({
  editorId,
  item,
  index,
  fields,
  label,
  description,
  errors,
  expanded,
  removable,
  disabled,
  readOnly,
  onToggle,
  onRemove,
  onChange,
}: ObjectListEditorItemProps<TItem>) {
  return (
    <section
      className={`${styles.objectCard} ${errors.length > 0 ? styles.objectCardInvalid : ''}`}
      aria-labelledby={`${editorId}-item-${index}`}
    >
      <div className={styles.objectCardHeader}>
        <div className={styles.objectCardHeading}>
          <span className={styles.itemIndex}>{String(index + 1).padStart(2, '0')}</span>
          <div>
            <h4 id={`${editorId}-item-${index}`}>{label}</h4>
            {description ? <p>{description}</p> : null}
          </div>
        </div>
        {!readOnly ? (
          <div className={styles.cardActions}>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={onToggle}
              disabled={disabled}
              aria-expanded={expanded}
            >
              {expanded ? 'Done' : 'Edit'}
            </button>
            {removable ? (
              <button
                type="button"
                className={styles.removeButton}
                onClick={onRemove}
                disabled={disabled}
              >
                Remove
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
      {errors.length ? (
        <ul className={styles.validationList} aria-live="polite">
          {errors.map((error) => (
            <li key={error}>{error}</li>
          ))}
        </ul>
      ) : null}
      {expanded ? (
        <ObjectEditorFields
          editorId={editorId}
          item={item}
          index={index}
          fields={fields}
          disabled={disabled}
          readOnly={readOnly}
          onChange={onChange}
        />
      ) : null}
    </section>
  )
}
