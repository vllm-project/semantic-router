import { KeyValueEditor } from '../components/KeyValueEditor'
import { StringListEditor } from '../components/StringListEditor'
import {
  createRouterStructuredValue,
  type RouterStructuredSchema,
} from './configPageRouterStructuredSchema'
import styles from './ConfigPageRouterStructuredEditor.module.css'

interface ConfigPageRouterStructuredEditorProps {
  schema: RouterStructuredSchema
  value: unknown
  onChange: (value: unknown) => void
  readOnly?: boolean
  depth?: number
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function asStringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : []
}

function asNumberList(value: unknown): number[] {
  return Array.isArray(value)
    ? value.filter((item): item is number => typeof item === 'number' && Number.isFinite(item))
    : []
}

function scalarValue(value: unknown): string | number {
  return typeof value === 'string' || typeof value === 'number' ? value : ''
}

function isCompound(schema: RouterStructuredSchema): boolean {
  return ['object', 'object-list', 'string-list', 'number-list', 'string-map'].includes(schema.kind)
}

function ReadOnlyScalar({ schema, value }: { schema: RouterStructuredSchema; value: unknown }) {
  const display = schema.kind === 'password' && value ? '••••••••' : value
  return (
    <span className={styles.readonlyValue}>
      {display === undefined || display === '' ? 'Not set' : String(display)}
    </span>
  )
}

function NumberListEditor({
  schema,
  value,
  onChange,
  readOnly,
}: ConfigPageRouterStructuredEditorProps) {
  const values = asNumberList(value)
  if (readOnly) {
    return values.length > 0 ? (
      <span className={styles.readonlyValue}>{values.join(', ')}</span>
    ) : (
      <p className={styles.empty}>No values configured.</p>
    )
  }
  return (
    <div className={styles.numberList}>
      {values.map((item, index) => (
        <div key={index} className={styles.numberRow}>
          <input
            className={styles.input}
            type="number"
            step={schema.step ?? 'any'}
            min={schema.min}
            max={schema.max}
            value={item}
            onChange={(event) => {
              const next = [...values]
              next[index] = Number(event.target.value)
              onChange(next)
            }}
            aria-label={`${schema.label} ${index + 1}`}
          />
          <button
            type="button"
            className={styles.removeButton}
            onClick={() => onChange(values.filter((_, itemIndex) => itemIndex !== index))}
          >
            Remove
          </button>
        </div>
      ))}
      <button type="button" className={styles.addButton} onClick={() => onChange([...values, 0])}>
        + Add value
      </button>
    </div>
  )
}

function ObjectEditor({
  schema,
  value,
  onChange,
  readOnly = false,
  depth = 0,
}: ConfigPageRouterStructuredEditorProps) {
  const record = asRecord(value)
  const fields = Object.entries(schema.fields ?? {})
  const knownKeys = new Set(fields.map(([key]) => key))
  const unknownCount = Object.keys(record).filter((key) => !knownKeys.has(key)).length
  const content = (
    <div className={styles.objectFields}>
      <div className={styles.objectGrid}>
        {fields.map(([key, fieldSchema]) => {
          const wide = isCompound(fieldSchema)
          return (
            <div key={key} className={wide ? styles.fieldWide : styles.field}>
              <span className={styles.label}>
                {fieldSchema.label}
                {fieldSchema.required ? ' *' : ''}
              </span>
              {fieldSchema.description ? (
                <p className={styles.description}>{fieldSchema.description}</p>
              ) : null}
              <ConfigPageRouterStructuredEditor
                schema={fieldSchema}
                value={record[key]}
                onChange={(nextValue) => {
                  const next = { ...record }
                  if (nextValue === undefined) delete next[key]
                  else next[key] = nextValue
                  onChange(next)
                }}
                readOnly={readOnly}
                depth={depth + 1}
              />
            </div>
          )
        })}
      </div>
      {unknownCount > 0 ? (
        <p className={styles.preservedNote}>
          {unknownCount} additional advanced field{unknownCount === 1 ? '' : 's'} will be preserved
          unchanged.
        </p>
      ) : null}
    </div>
  )

  if (depth === 0) return content
  return (
    <details className={styles.nestedObject} open={readOnly}>
      <summary>{schema.label}</summary>
      <div className={styles.nestedContent}>{content}</div>
    </details>
  )
}

function ObjectListEditor({
  schema,
  value,
  onChange,
  readOnly = false,
}: ConfigPageRouterStructuredEditorProps) {
  const items = Array.isArray(value) ? value : []
  const itemSchema = schema.item
  if (!itemSchema) return <p className={styles.empty}>No item schema configured.</p>
  return (
    <div className={styles.objectList}>
      {items.map((item, index) => {
        const record = asRecord(item)
        const labelValue = schema.itemLabelKey ? record[schema.itemLabelKey] : undefined
        const title =
          typeof labelValue === 'string' && labelValue.trim()
            ? labelValue
            : `${schema.label} ${index + 1}`
        return (
          <section key={index} className={styles.objectCard}>
            <div className={styles.objectCardHeader}>
              <h4 className={styles.objectCardTitle}>{title}</h4>
              {!readOnly ? (
                <button
                  type="button"
                  className={styles.removeButton}
                  onClick={() => onChange(items.filter((_, itemIndex) => itemIndex !== index))}
                >
                  Remove
                </button>
              ) : null}
            </div>
            <ConfigPageRouterStructuredEditor
              schema={itemSchema}
              value={item}
              onChange={(nextItem) =>
                onChange(items.map((entry, itemIndex) => (itemIndex === index ? nextItem : entry)))
              }
              readOnly={readOnly}
              depth={0}
            />
          </section>
        )
      })}
      {items.length === 0 ? (
        <p className={styles.empty}>{schema.emptyLabel ?? 'No items configured.'}</p>
      ) : null}
      {!readOnly ? (
        <button
          type="button"
          className={styles.addButton}
          onClick={() => onChange([...items, createRouterStructuredValue(itemSchema)])}
        >
          + {schema.addLabel ?? 'Add item'}
        </button>
      ) : null}
    </div>
  )
}

export default function ConfigPageRouterStructuredEditor({
  schema,
  value,
  onChange,
  readOnly = false,
  depth = 0,
}: ConfigPageRouterStructuredEditorProps) {
  if (schema.kind === 'object') {
    return (
      <ObjectEditor
        schema={schema}
        value={value}
        onChange={onChange}
        readOnly={readOnly}
        depth={depth}
      />
    )
  }
  if (schema.kind === 'object-list') {
    return (
      <ObjectListEditor
        schema={schema}
        value={value}
        onChange={onChange}
        readOnly={readOnly}
        depth={depth}
      />
    )
  }
  if (schema.kind === 'string-list') {
    return (
      <StringListEditor
        value={asStringList(value)}
        onChange={onChange}
        addLabel={`Add ${schema.label.toLocaleLowerCase()}`}
        emptyLabel={`No ${schema.label.toLocaleLowerCase()} configured.`}
        itemLabel={schema.label}
        placeholder={schema.placeholder}
        readOnly={readOnly}
      />
    )
  }
  if (schema.kind === 'number-list') {
    return (
      <NumberListEditor
        schema={schema}
        value={value}
        onChange={onChange}
        readOnly={readOnly}
        depth={depth}
      />
    )
  }
  if (schema.kind === 'string-map') {
    return (
      <KeyValueEditor
        value={asRecord(value) as Record<string, string>}
        onChange={onChange}
        keyLabel="Key"
        valueLabel="Value"
        emptyLabel="No entries configured."
        readOnly={readOnly}
      />
    )
  }
  if (readOnly) return <ReadOnlyScalar schema={schema} value={value} />
  if (schema.kind === 'boolean') {
    return (
      <label className={styles.checkField}>
        <input
          type="checkbox"
          aria-label={schema.label}
          checked={value === true}
          onChange={(event) => onChange(event.target.checked)}
        />
        {value === true ? 'Enabled' : 'Disabled'}
      </label>
    )
  }
  if (schema.kind === 'select') {
    const current = typeof value === 'string' ? value : ''
    const options =
      current && !schema.options?.includes(current)
        ? [current, ...(schema.options ?? [])]
        : (schema.options ?? [])
    return (
      <select
        className={styles.select}
        aria-label={schema.label}
        value={current}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">Not set</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    )
  }
  return (
    <input
      className={styles.input}
      aria-label={schema.label}
      type={schema.kind === 'number' ? 'number' : schema.kind === 'password' ? 'password' : 'text'}
      value={scalarValue(value)}
      min={schema.min}
      max={schema.max}
      step={schema.kind === 'number' ? (schema.step ?? 'any') : undefined}
      placeholder={schema.placeholder}
      onChange={(event) => {
        if (schema.kind === 'number') {
          onChange(event.target.value === '' ? undefined : Number(event.target.value))
        } else {
          onChange(event.target.value)
        }
      }}
    />
  )
}
