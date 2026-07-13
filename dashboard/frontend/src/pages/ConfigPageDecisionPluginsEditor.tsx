import { getPluginFieldSchema, PLUGIN_DESCRIPTIONS, PLUGIN_TYPES } from '../lib/dslMutations'
import type { FieldSchema } from '../lib/dslSchemas'
import type { DSLFieldObject, DSLFieldValue } from '../types/dsl'
import type { DecisionFormState, DecisionPluginConfiguration } from './configPageSupport'
import decisionStyles from './ConfigPageDecisionsSection.module.css'
import { FieldEditor } from './builderPageFormPrimitives'

interface ConfigPageDecisionPluginsEditorProps {
  value: DecisionFormState['plugins']
  onChange: (value: DecisionFormState['plugins']) => void
}

function normalizeConfiguration(value: unknown): DSLFieldObject {
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value)
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
        ? (parsed as DSLFieldObject)
        : {}
    } catch {
      return {}
    }
  }
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as DSLFieldObject)
    : {}
}

function updateConfigurationField(
  configuration: DSLFieldObject,
  field: FieldSchema,
  value: unknown,
): DecisionPluginConfiguration {
  const next = { ...configuration }
  if (value === undefined || value === '') delete next[field.key]
  else next[field.key] = value as DSLFieldValue
  return next as DecisionPluginConfiguration
}

export default function ConfigPageDecisionPluginsEditor({
  value,
  onChange,
}: ConfigPageDecisionPluginsEditorProps) {
  const rows = Array.isArray(value) ? value : []

  const updateRow = (
    index: number,
    patch: Partial<NonNullable<DecisionFormState['plugins']>[number]>,
  ) => {
    onChange(rows.map((row, rowIndex) => (rowIndex === index ? { ...row, ...patch } : row)))
  }

  return (
    <div className={decisionStyles.editorList}>
      {rows.map((plugin, index) => {
        const type = plugin?.type || ''
        const schema = getPluginFieldSchema(type)
        const configuration = normalizeConfiguration(plugin?.configuration)
        const knownKeys = new Set(schema.map((field) => field.key))
        const preservedFields = Object.keys(configuration).filter(
          (key) => !knownKeys.has(key),
        ).length
        const typeOptions =
          type && !PLUGIN_TYPES.includes(type as (typeof PLUGIN_TYPES)[number])
            ? [type, ...PLUGIN_TYPES]
            : [...PLUGIN_TYPES]

        return (
          <section key={`${type || 'plugin'}-${index}`} className={decisionStyles.editorCard}>
            <div className={decisionStyles.editorMetaRow}>
              <label className={decisionStyles.editorControlLabel} style={{ flex: 1 }}>
                <span className={decisionStyles.editorControlLabelText}>Plugin type</span>
                <select
                  value={type}
                  onChange={(event) => updateRow(index, { type: event.target.value })}
                  className={decisionStyles.editorInput}
                >
                  <option value="">Select a plugin</option>
                  {typeOptions.map((option) => (
                    <option key={option} value={option}>
                      {option.replace(/_/g, ' ')}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                onClick={() => onChange(rows.filter((_, rowIndex) => rowIndex !== index))}
                className={decisionStyles.editorButtonDanger}
              >
                Remove
              </button>
            </div>

            {type && PLUGIN_DESCRIPTIONS[type] ? (
              <p className={decisionStyles.editorHelp}>{PLUGIN_DESCRIPTIONS[type]}</p>
            ) : null}

            {schema.length > 0 ? (
              <div className={decisionStyles.pluginConfigurationGrid}>
                {schema.map((field) => (
                  <FieldEditor
                    key={field.key}
                    schema={field}
                    value={configuration[field.key]}
                    onChange={(nextValue) =>
                      updateRow(index, {
                        configuration: updateConfigurationField(configuration, field, nextValue),
                      })
                    }
                  />
                ))}
              </div>
            ) : type ? (
              <p className={decisionStyles.editorHelp} role="note">
                This custom plugin has no registered form schema. Its existing configuration is
                preserved; use DSL mode to edit extension fields.
              </p>
            ) : (
              <p className={decisionStyles.editorHelp}>Choose a plugin to configure its fields.</p>
            )}

            {preservedFields > 0 ? (
              <p className={decisionStyles.editorHelp} role="note">
                {preservedFields} extension field{preservedFields === 1 ? '' : 's'} will be
                preserved unchanged.
              </p>
            ) : null}
          </section>
        )
      })}

      <button
        type="button"
        onClick={() => onChange([...rows, { type: 'semantic_cache', configuration: {} }])}
        className={decisionStyles.editorButtonSecondary}
      >
        Add Plugin
      </button>
    </div>
  )
}
