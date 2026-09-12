import type { FieldSchema } from '../lib/dslSchemas'
import type { DSLFieldValue } from '../types/dsl'
import { FieldEditor } from './builderPageFormPrimitives'
import styles from './ConfigPageSchemaFieldsEditor.module.css'

interface ConfigPageSchemaFieldsEditorProps {
  schema: FieldSchema[]
  value: Record<string, unknown>
  onChange?: (value: Record<string, unknown>) => void
  readOnly?: boolean
}

const COMPOUND_TYPES = new Set<FieldSchema['type']>([
  'string[]',
  'number[]',
  'string[][]',
  'object',
  'object[]',
  'key-value',
  'json',
  'rule',
])

function displayValue(value: unknown): string {
  if (value === undefined || value === null || value === '') return 'Not set'
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value, null, 2)
}

export default function ConfigPageSchemaFieldsEditor({
  schema,
  value,
  onChange,
  readOnly = false,
}: ConfigPageSchemaFieldsEditorProps) {
  const knownKeys = new Set(schema.map((field) => field.key))
  const visibleSchema = readOnly
    ? schema.filter((field) => Object.prototype.hasOwnProperty.call(value, field.key))
    : schema
  const unknownEntries = Object.entries(value).filter(([key]) => !knownKeys.has(key))
  const isEmpty = visibleSchema.length === 0 && unknownEntries.length === 0

  return (
    <fieldset className={styles.fieldset} disabled={readOnly}>
      <div className={styles.fields}>
        {visibleSchema.map((field) => (
          <div
            key={field.key}
            className={COMPOUND_TYPES.has(field.type) ? styles.wide : styles.field}
          >
            {readOnly ? (
              <div className={styles.readonlyField}>
                <span className={styles.readonlyLabel}>{field.label}</span>
                <pre className={styles.value}>{displayValue(value[field.key])}</pre>
              </div>
            ) : (
              <FieldEditor
                schema={field}
                value={value[field.key]}
                onChange={(nextValue) => {
                  const next = { ...value }
                  if (nextValue === undefined) delete next[field.key]
                  else next[field.key] = nextValue as DSLFieldValue
                  onChange?.(next)
                }}
              />
            )}
          </div>
        ))}
        {unknownEntries.length > 0 ? (
          <section className={styles.preserved} aria-label="Additional configuration fields">
            <h4 className={styles.preservedTitle}>Additional configuration</h4>
            {unknownEntries.map(([key, entryValue]) => (
              <div key={key} className={styles.preservedRow}>
                <span className={styles.key}>{key}</span>
                <pre className={styles.value}>{displayValue(entryValue)}</pre>
              </div>
            ))}
          </section>
        ) : null}
        {isEmpty ? <p className={styles.empty}>No fields configured.</p> : null}
      </div>
    </fieldset>
  )
}
