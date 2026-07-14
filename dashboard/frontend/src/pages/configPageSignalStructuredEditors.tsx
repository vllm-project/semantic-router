import { ObjectListEditor, type ObjectEditorField } from '../components/ObjectListEditor'
import { StringListEditor } from '../components/StringListEditor'
import type {
  DecisionCondition,
  NumericPredicate,
  StructureFeature,
  Subject,
} from './configPageSupport'
import {
  readConditions,
  readStringList,
  readStructureFeature,
  readStructurePredicate,
  readSubjects,
  STRUCTURE_FEATURE_TYPES,
  STRUCTURE_SOURCE_TYPES,
} from './configPageSignalFormSupport'
import styles from './configPageSignalStructuredEditors.module.css'

interface UnknownEditorProps {
  value: unknown
  onChange: (value: unknown) => void
  readOnly?: boolean
}

interface SignalStringListEditorProps extends UnknownEditorProps {
  addLabel: string
  emptyLabel: string
  itemLabel: string
  placeholder?: string
}

export function SignalStringListEditor({
  value,
  onChange,
  addLabel,
  emptyLabel,
  itemLabel,
  placeholder,
  readOnly = false,
}: SignalStringListEditorProps) {
  return (
    <StringListEditor
      value={readStringList(value)}
      onChange={onChange}
      addLabel={addLabel}
      emptyLabel={emptyLabel}
      itemLabel={itemLabel}
      placeholder={placeholder}
      readOnly={readOnly}
    />
  )
}

const conditionFields: ObjectEditorField<DecisionCondition>[] = [
  { key: 'type', label: 'Signal type', required: true, placeholder: 'domain' },
  { key: 'name', label: 'Signal name', required: true, placeholder: 'computer_science' },
]

export function SignalConditionsEditor({ value, onChange, readOnly = false }: UnknownEditorProps) {
  return (
    <ObjectListEditor<DecisionCondition>
      value={readConditions(value)}
      onChange={onChange}
      fields={conditionFields}
      createItem={() => ({ type: '', name: '' })}
      addLabel="Add condition"
      emptyLabel="No composer conditions."
      itemLabel={(condition) => condition.name || 'New condition'}
      itemDescription={(condition) => condition.type || 'Choose a signal type and name.'}
      validateItem={(condition) => [
        ...(!condition.type.trim() ? ['Signal type is required.'] : []),
        ...(!condition.name.trim() ? ['Signal name is required.'] : []),
      ]}
      readOnly={readOnly}
    />
  )
}

const subjectFields: ObjectEditorField<Subject>[] = [
  {
    key: 'kind',
    label: 'Subject kind',
    type: 'select',
    options: ['User', 'Group'],
    required: true,
  },
  { key: 'name', label: 'Subject name', required: true, placeholder: 'alice or admins' },
]

export function SignalSubjectsEditor({ value, onChange, readOnly = false }: UnknownEditorProps) {
  return (
    <ObjectListEditor<Subject>
      value={readSubjects(value)}
      onChange={onChange}
      fields={subjectFields}
      createItem={() => ({ kind: 'User', name: '' })}
      addLabel="Add subject"
      emptyLabel="No users or groups configured."
      itemLabel={(subject) => subject.name || 'New subject'}
      itemDescription={(subject) => subject.kind}
      validateItem={(subject) => (!subject.name.trim() ? ['Subject name is required.'] : [])}
      readOnly={readOnly}
    />
  )
}

interface SequenceListEditorProps {
  value: readonly string[][]
  onChange: (value: string[][]) => void
  readOnly?: boolean
}

function SequenceListEditor({ value, onChange, readOnly = false }: SequenceListEditorProps) {
  return (
    <div className={styles.sequenceList}>
      {value.map((sequence, index) => (
        <section key={index} className={styles.sequenceCard}>
          <div className={styles.sequenceHeader}>
            <strong>Sequence {index + 1}</strong>
            {!readOnly ? (
              <button
                type="button"
                className={styles.removeButton}
                onClick={() =>
                  onChange(value.filter((_, sequenceIndex) => sequenceIndex !== index))
                }
              >
                Remove
              </button>
            ) : null}
          </div>
          <StringListEditor
            value={sequence}
            onChange={(nextSequence) =>
              onChange(
                value.map((entry, sequenceIndex) =>
                  sequenceIndex === index ? nextSequence : entry,
                ),
              )
            }
            addLabel="Add token"
            emptyLabel="This sequence has no tokens."
            itemLabel="Token"
            placeholder="first"
            readOnly={readOnly}
          />
        </section>
      ))}
      {!readOnly ? (
        <button
          type="button"
          className={styles.addButton}
          onClick={() => onChange([...value, ['']])}
        >
          + Add sequence
        </button>
      ) : null}
    </div>
  )
}

export function SignalStructureFeatureEditor({
  value,
  onChange,
  readOnly = false,
}: UnknownEditorProps) {
  const feature = readStructureFeature(value)
  const updateFeature = (next: StructureFeature) => onChange(next)
  const updateSource = (nextSource: StructureFeature['source']) =>
    updateFeature({ ...feature, source: nextSource })

  if (readOnly) {
    return (
      <div className={styles.fieldGrid}>
        <div className={styles.field}>
          <span className={styles.label}>Feature</span>
          <span className={styles.readonlyValue}>{feature.type}</span>
        </div>
        <div className={styles.field}>
          <span className={styles.label}>Source</span>
          <span className={styles.readonlyValue}>{feature.source.type}</span>
        </div>
        {feature.source.pattern ? (
          <div className={styles.wideField}>
            <span className={styles.label}>Pattern</span>
            <span className={styles.readonlyValue}>{feature.source.pattern}</span>
          </div>
        ) : null}
        {feature.source.keywords ? (
          <div className={styles.wideField}>
            <span className={styles.label}>Keywords</span>
            <StringListEditor
              value={feature.source.keywords}
              onChange={() => undefined}
              readOnly
              itemLabel="Keyword"
            />
          </div>
        ) : null}
        {feature.source.sequences ? (
          <div className={styles.wideField}>
            <span className={styles.label}>Sequences</span>
            <SequenceListEditor
              value={feature.source.sequences}
              onChange={() => undefined}
              readOnly
            />
          </div>
        ) : null}
      </div>
    )
  }

  return (
    <div className={styles.fieldGrid}>
      <label className={styles.field}>
        <span className={styles.label}>Feature type</span>
        <select
          className={styles.select}
          value={feature.type}
          onChange={(event) => {
            const nextType = event.target.value
            if (nextType === 'sequence') {
              updateFeature({
                type: nextType,
                source: { type: 'sequence', case_sensitive: false, sequences: [['first', 'then']] },
              })
            } else {
              updateFeature({ ...feature, type: nextType })
            }
          }}
        >
          {STRUCTURE_FEATURE_TYPES.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
      </label>
      <label className={styles.field}>
        <span className={styles.label}>Source type</span>
        <select
          className={styles.select}
          value={feature.source.type}
          onChange={(event) => {
            const nextType = event.target.value
            if (nextType === 'regex') updateSource({ type: nextType, pattern: '' })
            else if (nextType === 'keyword_set')
              updateSource({ type: nextType, keywords: [''], case_sensitive: false })
            else
              updateFeature({
                type: 'sequence',
                source: { type: nextType, sequences: [['']], case_sensitive: false },
              })
          }}
        >
          {STRUCTURE_SOURCE_TYPES.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
      </label>
      {feature.source.type === 'regex' ? (
        <label className={styles.wideField}>
          <span className={styles.label}>Regex pattern</span>
          <input
            className={styles.input}
            value={feature.source.pattern ?? ''}
            onChange={(event) => updateSource({ ...feature.source, pattern: event.target.value })}
            placeholder="[?？]"
          />
        </label>
      ) : null}
      {feature.source.type === 'keyword_set' ? (
        <div className={styles.wideField}>
          <span className={styles.label}>Keywords</span>
          <StringListEditor
            value={feature.source.keywords ?? []}
            onChange={(keywords) => updateSource({ ...feature.source, keywords })}
            addLabel="Add keyword"
            emptyLabel="No structure keywords."
            itemLabel="Keyword"
            placeholder="at least"
          />
        </div>
      ) : null}
      {feature.source.type === 'sequence' ? (
        <div className={styles.wideField}>
          <span className={styles.label}>Token sequences</span>
          <SequenceListEditor
            value={feature.source.sequences ?? []}
            onChange={(sequences) => updateSource({ ...feature.source, sequences })}
          />
        </div>
      ) : null}
      <label className={`${styles.checkField} ${styles.wideField}`}>
        <input
          type="checkbox"
          checked={feature.source.case_sensitive ?? false}
          onChange={(event) =>
            updateSource({ ...feature.source, case_sensitive: event.target.checked })
          }
        />
        Case-sensitive matching
      </label>
    </div>
  )
}

export function SignalStructurePredicateEditor({
  value,
  onChange,
  readOnly = false,
}: UnknownEditorProps) {
  const predicate = readStructurePredicate(value)
  const bounds: Array<{ key: keyof NumericPredicate; label: string }> = [
    { key: 'gt', label: 'Greater than' },
    { key: 'gte', label: 'Greater than or equal' },
    { key: 'lt', label: 'Less than' },
    { key: 'lte', label: 'Less than or equal' },
  ]
  return (
    <div className={styles.fieldGrid}>
      {bounds.map(({ key, label }) => (
        <label key={key} className={styles.field}>
          <span className={styles.label}>{label}</span>
          {readOnly ? (
            <span className={styles.readonlyValue}>{predicate[key] ?? 'Not set'}</span>
          ) : (
            <input
              className={styles.input}
              type="number"
              step="any"
              value={predicate[key] ?? ''}
              onChange={(event) => {
                const next = { ...predicate }
                if (event.target.value === '') delete next[key]
                else next[key] = Number(event.target.value)
                onChange(next)
              }}
            />
          )}
        </label>
      ))}
    </div>
  )
}
