import {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
  type AlgorithmType,
} from '../lib/dslAlgorithmSchemas'
import { getAlgorithmFieldSchema } from '../lib/dslAlgorithmSchemas'
import ConfigPageSchemaFieldsEditor from './ConfigPageSchemaFieldsEditor'
import {
  algorithmFields,
  algorithmType,
  mergeAlgorithmFields,
} from './configPageDecisionAlgorithmSupport'
import styles from './ConfigPageDecisionsSection.module.css'

interface ConfigPageDecisionAlgorithmEditorProps {
  value: unknown
  onChange?: (value: Record<string, unknown> | undefined) => void
  readOnly?: boolean
}

export default function ConfigPageDecisionAlgorithmEditor({
  value,
  onChange,
  readOnly = false,
}: ConfigPageDecisionAlgorithmEditorProps) {
  const configured = Boolean(value && typeof value === 'object')
  const type = algorithmType(value)
  const fields = algorithmFields(value)

  if (!configured && readOnly) return <span>Not configured</span>

  return (
    <div className={styles.advancedEditor}>
      <label className={styles.advancedTypeControl}>
        <span>Algorithm type</span>
        <select
          value={configured ? type : ''}
          disabled={readOnly}
          onChange={(event) => {
            if (!event.target.value) {
              onChange?.(undefined)
              return
            }
            const nextType = event.target.value as AlgorithmType
            onChange?.(mergeAlgorithmFields(undefined, nextType, {}))
          }}
        >
          <option value="">No algorithm</option>
          {ALGORITHM_TYPES.map((candidate) => (
            <option key={candidate} value={candidate}>
              {candidate}
            </option>
          ))}
        </select>
      </label>
      {configured ? (
        <>
          <p className={styles.editorHelp}>{ALGORITHM_DESCRIPTIONS[type]}</p>
          <ConfigPageSchemaFieldsEditor
            schema={getAlgorithmFieldSchema(type)}
            value={fields}
            readOnly={readOnly}
            onChange={(nextFields) => onChange?.(mergeAlgorithmFields(value, type, nextFields))}
          />
        </>
      ) : null}
    </div>
  )
}
