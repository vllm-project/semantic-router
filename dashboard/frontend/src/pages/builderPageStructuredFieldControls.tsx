import type { ReactNode } from 'react'

import { KeyValueEditor } from '@/components/KeyValueEditor'
import { ObjectListEditor, type ObjectEditorField } from '@/components/ObjectListEditor'
import { StringListEditor } from '@/components/StringListEditor'
import type { FieldSchema } from '@/lib/dslMutations'
import type { DSLFieldObject, DSLFieldValue } from '@/types/dsl'

import styles from './BuilderPage.module.css'
import {
  normalizeStructuredObject,
  normalizeStructuredObjectList,
  normalizeStructuredStringList,
  normalizeStructuredStringMatrix,
  removeStructuredObjectListItem,
  requiredStructuredFieldErrors,
  structuredItemLabel,
  supportsSharedObjectList,
  updateStructuredObjectField,
  updateStructuredObjectListItem,
} from './builderPageStructuredFieldSupport'

type RenderNestedField = (
  schema: FieldSchema,
  value: unknown,
  onChange: (value: unknown) => void,
) => ReactNode

interface StructuredFieldEditorProps {
  schema: FieldSchema
  value: unknown
  onChange: (value: unknown) => void
  renderField: RenderNestedField
}

function fieldItemLabel(label: string): string {
  return label.endsWith('s') ? label.slice(0, -1) : label
}

function FieldShell({
  schema,
  children,
  onClear,
}: {
  schema: FieldSchema
  children: ReactNode
  onClear?: () => void
}) {
  return (
    <div className={styles.fieldGroup} role="group" aria-label={schema.label}>
      <div className={styles.structuredFieldHeader}>
        <span className={styles.fieldLabel}>
          {schema.label}{' '}
          {schema.required ? <span className={styles.structuredRequired}>*</span> : null}
        </span>
        {onClear ? (
          <button type="button" className={styles.structuredClearButton} onClick={onClear}>
            Clear
          </button>
        ) : null}
      </div>
      {schema.description ? (
        <p className={styles.structuredDescription}>{schema.description}</p>
      ) : null}
      {children}
    </div>
  )
}

function StringListField({
  schema,
  value,
  onChange,
}: Omit<StructuredFieldEditorProps, 'renderField'>) {
  const values = normalizeStructuredStringList(value)
  return (
    <FieldShell schema={schema}>
      <StringListEditor
        value={values}
        onChange={onChange}
        addLabel={schema.addLabel || `Add ${fieldItemLabel(schema.label).toLocaleLowerCase()}`}
        emptyLabel={schema.emptyLabel || `No ${schema.label.toLocaleLowerCase()} configured.`}
        itemLabel={schema.itemLabel || fieldItemLabel(schema.label)}
        placeholder={schema.placeholder}
      />
    </FieldShell>
  )
}

function KeyValueField({
  schema,
  value,
  onChange,
}: Omit<StructuredFieldEditorProps, 'renderField'>) {
  const objectValue = normalizeStructuredObject(value)
  const entries = Object.entries(objectValue).filter(
    (entry): entry is [string, string] => typeof entry[1] === 'string',
  )
  return (
    <FieldShell schema={schema} onClear={value === undefined ? undefined : () => onChange(undefined)}>
      <KeyValueEditor
        value={Object.fromEntries(entries)}
        onChange={onChange}
        addLabel={schema.addLabel}
        emptyLabel={schema.emptyLabel}
        keyLabel={schema.keyLabel}
        valueLabel={schema.valueLabel}
      />
    </FieldShell>
  )
}

function ObjectField({ schema, value, onChange, renderField }: StructuredFieldEditorProps) {
  const objectValue = normalizeStructuredObject(value)
  const fields = schema.fields || []
  return (
    <FieldShell
      schema={schema}
      onClear={!schema.required && value !== undefined ? () => onChange(undefined) : undefined}
    >
      <div className={styles.structuredSurface}>
        {fields.length > 0 ? (
          fields.map((field) =>
            renderField(field, objectValue[field.key], (nextValue) =>
              onChange(
                updateStructuredObjectField(
                  objectValue,
                  field.key,
                  nextValue as DSLFieldValue | undefined,
                ),
              ),
            ),
          )
        ) : (
          <p className={styles.structuredEmpty}>No fields are available for this object.</p>
        )}
      </div>
    </FieldShell>
  )
}

function sharedObjectFieldType(field: FieldSchema): ObjectEditorField<DSLFieldObject>['type'] {
  if (field.type === 'number' || field.type === 'select' || field.type === 'key-value') {
    return field.type
  }
  return 'text'
}

function SharedObjectListField({
  schema,
  value,
  onChange,
}: Omit<StructuredFieldEditorProps, 'renderField'>) {
  const items = normalizeStructuredObjectList(value)
  const fields = schema.fields || []
  const objectFields: ObjectEditorField<DSLFieldObject>[] = fields.map((field) => ({
    key: field.key,
    label: field.label,
    type: sharedObjectFieldType(field),
    placeholder: field.placeholder,
    options: field.options,
    required: field.required,
    helpText: field.description,
    fullWidth: field.type === 'key-value',
  }))

  return (
    <FieldShell schema={schema}>
      <ObjectListEditor<DSLFieldObject>
        value={items}
        onChange={onChange}
        fields={objectFields}
        createItem={() => ({})}
        addLabel={schema.addLabel}
        emptyLabel={schema.emptyLabel}
        itemLabel={(item, index) => structuredItemLabel(schema, item, index)}
        validateItem={(item) => requiredStructuredFieldErrors(fields, item)}
        minItems={schema.required ? 1 : 0}
      />
    </FieldShell>
  )
}

function RecursiveObjectListField({
  schema,
  value,
  onChange,
  renderField,
}: StructuredFieldEditorProps) {
  const items = normalizeStructuredObjectList(value)
  const fields = schema.fields || []

  return (
    <FieldShell schema={schema}>
      <div className={styles.structuredList}>
        {items.map((item, index) => {
          const errors = requiredStructuredFieldErrors(fields, item)
          return (
            <section key={index} className={styles.structuredCard}>
              <div className={styles.structuredCardHeader}>
                <div>
                  <span className={styles.structuredIndex}>{String(index + 1).padStart(2, '0')}</span>
                  <h4>{structuredItemLabel(schema, item, index)}</h4>
                </div>
                <button
                  type="button"
                  className={styles.structuredRemoveButton}
                  onClick={() => onChange(removeStructuredObjectListItem(items, index))}
                  disabled={Boolean(schema.required && items.length <= 1)}
                >
                  Remove
                </button>
              </div>
              {errors.length > 0 ? (
                <ul className={styles.structuredErrors} aria-live="polite">
                  {errors.map((error) => <li key={error}>{error}</li>)}
                </ul>
              ) : null}
              <div className={styles.structuredCardBody}>
                {fields.map((field) =>
                  renderField(field, item[field.key], (nextValue) => {
                    const nextItem = updateStructuredObjectField(
                      item,
                      field.key,
                      nextValue as DSLFieldValue | undefined,
                    )
                    onChange(updateStructuredObjectListItem(items, index, nextItem))
                  }),
                )}
              </div>
            </section>
          )
        })}
      </div>
      {items.length === 0 ? (
        <p className={styles.structuredEmpty}>{schema.emptyLabel || 'No items configured.'}</p>
      ) : null}
      <button
        type="button"
        className={styles.structuredAddButton}
        onClick={() => onChange([...items, {}])}
      >
        <span aria-hidden="true">+</span>
        {schema.addLabel || 'Add item'}
      </button>
    </FieldShell>
  )
}

function StringMatrixField({
  schema,
  value,
  onChange,
}: Omit<StructuredFieldEditorProps, 'renderField'>) {
  const rows = normalizeStructuredStringMatrix(value)
  return (
    <FieldShell schema={schema}>
      <div className={styles.structuredList}>
        {rows.map((row, index) => (
          <section key={index} className={styles.structuredCard}>
            <div className={styles.structuredCardHeader}>
              <div>
                <span className={styles.structuredIndex}>{String(index + 1).padStart(2, '0')}</span>
                <h4>Sequence {index + 1}</h4>
              </div>
              <button
                type="button"
                className={styles.structuredRemoveButton}
                onClick={() => onChange(rows.filter((_, rowIndex) => rowIndex !== index))}
              >
                Remove
              </button>
            </div>
            {row.length < 2 ? (
              <p className={styles.structuredInlineError}>Add at least two markers.</p>
            ) : null}
            <StringListEditor
              value={row}
              onChange={(nextRow) =>
                onChange(rows.map((current, rowIndex) => (rowIndex === index ? nextRow : current)))
              }
              addLabel="Add marker"
              emptyLabel="No markers configured."
              itemLabel="Marker"
              placeholder="Sequence marker"
            />
          </section>
        ))}
      </div>
      {rows.length === 0 ? (
        <p className={styles.structuredEmpty}>{schema.emptyLabel || 'No sequences configured.'}</p>
      ) : null}
      <button
        type="button"
        className={styles.structuredAddButton}
        onClick={() => onChange([...rows, ['', '']])}
      >
        <span aria-hidden="true">+</span>
        {schema.addLabel || 'Add sequence'}
      </button>
    </FieldShell>
  )
}

function RuleNodeEditor({
  value,
  onChange,
  depth = 0,
}: {
  value: unknown
  onChange: (value: DSLFieldObject) => void
  depth?: number
}) {
  const node = normalizeStructuredObject(value)
  const operator = typeof node.operator === 'string' ? node.operator.toUpperCase() : ''
  const mode = ['AND', 'OR', 'NOT'].includes(operator) ? operator : 'signal'
  const conditions = normalizeStructuredObjectList(node.conditions)

  const setMode = (nextMode: string) => {
    if (nextMode === 'signal') {
      onChange({})
      return
    }
    onChange({ operator: nextMode, conditions: nextMode === 'NOT' ? [{}] : [] })
  }

  return (
    <div className={styles.ruleNode} data-depth={depth}>
      <label className={styles.structuredMiniLabel}>
        Node Type
        <select
          className={styles.fieldInput}
          value={mode}
          onChange={(event) => setMode(event.target.value)}
        >
          <option value="signal">Signal condition</option>
          <option value="AND">AND group</option>
          <option value="OR">OR group</option>
          <option value="NOT">NOT group</option>
        </select>
      </label>

      {mode === 'signal' ? (
        <div className={styles.ruleLeafGrid}>
          <label className={styles.structuredMiniLabel}>
            Signal Type
            <input
              className={styles.fieldInput}
              value={typeof node.type === 'string' ? node.type : ''}
              onChange={(event) =>
                onChange(updateStructuredObjectField(node, 'type', event.target.value || undefined))
              }
              placeholder="domain"
            />
          </label>
          <label className={styles.structuredMiniLabel}>
            Signal Name
            <input
              className={styles.fieldInput}
              value={typeof node.name === 'string' ? node.name : ''}
              onChange={(event) =>
                onChange(updateStructuredObjectField(node, 'name', event.target.value || undefined))
              }
              placeholder="technical"
            />
          </label>
        </div>
      ) : (
        <div className={styles.ruleConditions}>
          {conditions.map((condition, index) => (
            <div key={index} className={styles.ruleCondition}>
              <div className={styles.ruleConditionHeader}>
                <span>Condition {index + 1}</span>
                <button
                  type="button"
                  className={styles.structuredRemoveButton}
                  onClick={() =>
                    onChange({
                      ...node,
                      conditions: conditions.filter((_, conditionIndex) => conditionIndex !== index),
                    })
                  }
                  disabled={mode === 'NOT' && conditions.length <= 1}
                >
                  Remove
                </button>
              </div>
              <RuleNodeEditor
                value={condition}
                depth={depth + 1}
                onChange={(nextCondition) =>
                  onChange({
                    ...node,
                    conditions: updateStructuredObjectListItem(conditions, index, nextCondition),
                  })
                }
              />
            </div>
          ))}
          {mode !== 'NOT' || conditions.length === 0 ? (
            <button
              type="button"
              className={styles.structuredAddButton}
              onClick={() => onChange({ ...node, conditions: [...conditions, {}] })}
            >
              <span aria-hidden="true">+</span>
              Add condition
            </button>
          ) : null}
        </div>
      )}
    </div>
  )
}

function RuleField({
  schema,
  value,
  onChange,
}: Omit<StructuredFieldEditorProps, 'renderField'>) {
  return (
    <FieldShell schema={schema} onClear={value === undefined ? undefined : () => onChange(undefined)}>
      <RuleNodeEditor value={value} onChange={onChange} />
    </FieldShell>
  )
}

export function StructuredFieldEditor({
  schema,
  value,
  onChange,
  renderField,
}: StructuredFieldEditorProps) {
  if (schema.type === 'string[]') {
    return <StringListField schema={schema} value={value} onChange={onChange} />
  }
  if (schema.type === 'key-value') {
    return <KeyValueField schema={schema} value={value} onChange={onChange} />
  }
  if (schema.type === 'object') {
    return (
      <ObjectField schema={schema} value={value} onChange={onChange} renderField={renderField} />
    )
  }
  if (schema.type === 'object[]') {
    return supportsSharedObjectList(schema.fields || []) ? (
      <SharedObjectListField schema={schema} value={value} onChange={onChange} />
    ) : (
      <RecursiveObjectListField
        schema={schema}
        value={value}
        onChange={onChange}
        renderField={renderField}
      />
    )
  }
  if (schema.type === 'string[][]') {
    return <StringMatrixField schema={schema} value={value} onChange={onChange} />
  }
  if (schema.type === 'rule') {
    return <RuleField schema={schema} value={value} onChange={onChange} />
  }
  return null
}
