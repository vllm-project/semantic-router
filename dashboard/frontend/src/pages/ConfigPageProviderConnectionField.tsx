import type { ProviderConnectionField } from '../utils/providerCatalogApi'
import type { EditableConnectionValue } from './configPageModelOnboardingSupport'
import styles from './ConfigPageAddModelsDialog.module.css'

interface Props {
  field: ProviderConnectionField
  value: EditableConnectionValue
  onChange: (value: EditableConnectionValue) => void
}

export default function ConfigPageProviderConnectionField({ field, value, onChange }: Props) {
  if (field.kind === 'boolean') {
    return (
      <label className={`${styles.field} ${styles.booleanField}`}>
        <span>{field.label}</span>
        <input
          type="checkbox"
          checked={value === true}
          onChange={(event) => onChange(event.target.checked)}
        />
        {field.hint ? <small>{field.hint}</small> : null}
      </label>
    )
  }
  return (
    <label className={styles.field}>
      <span>
        {field.label} {!field.required ? <small>Optional</small> : null}
      </span>
      {field.kind === 'select' ? (
        <select
          value={typeof value === 'string' ? value : ''}
          onChange={(event) => onChange(event.target.value)}
        >
          {!field.required ? <option value="">Default</option> : null}
          {(field.options ?? []).map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          type={field.kind === 'integer' ? 'number' : 'text'}
          step={field.kind === 'integer' ? '1' : undefined}
          value={typeof value === 'string' ? value : ''}
          placeholder={field.placeholder}
          onChange={(event) => onChange(event.target.value)}
        />
      )}
      {field.hint ? <small>{field.hint}</small> : null}
    </label>
  )
}
