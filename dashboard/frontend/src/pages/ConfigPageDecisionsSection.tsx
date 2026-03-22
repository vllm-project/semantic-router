import styles from './ConfigPage.module.css'
import decisionStyles from './ConfigPageDecisionsSection.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'
import type { ViewSection } from '../components/ViewModal'
import type {
  ConfigData,
  ConfigDecisionConditionType,
  DecisionConfig,
  DecisionFormState,
  DecisionPluginConfiguration,
  NormalizedModel,
} from './configPageSupport'
import { TABLE_COLUMN_WIDTH } from './configPageSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { cloneConfigData } from './configPageCanonicalization'

interface ConfigPageDecisionsSectionProps {
  config: ConfigData | null
  isPythonCLI: boolean
  isReadonly: boolean
  decisionsSearch: string
  onDecisionsSearchChange: (value: string) => void
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
  removeDecisionByName: (cfg: ConfigData, targetName: string) => void
  models: NormalizedModel[]
}

export default function ConfigPageDecisionsSection({
  config,
  isPythonCLI,
  isReadonly,
  decisionsSearch,
  onDecisionsSearchChange,
  saveConfig,
  openEditModal,
  openViewModal,
  removeDecisionByName,
  models,
}: ConfigPageDecisionsSectionProps) {
  const decisions = config?.decisions || []

  const filteredDecisions = decisions.filter(decision =>
    decision.name.toLowerCase().includes(decisionsSearch.toLowerCase()) ||
    decision.description?.toLowerCase().includes(decisionsSearch.toLowerCase())
  )

  type DecisionRow = NonNullable<ConfigData['decisions']>[number]
  const decisionsColumns: Column<DecisionRow>[] = [
    {
      key: 'name',
      header: 'Name',
      sortable: true,
      render: (row) => <span style={{ fontWeight: 600 }}>{row.name}</span>
    },
    {
      key: 'priority',
      header: 'Priority',
      width: TABLE_COLUMN_WIDTH.compact,
      align: 'center',
      sortable: true,
      render: (row) => (
        <span className={`${styles.tableMetaBadge} ${styles.tableMetaBadgeMono}`}>
          P{row.priority}
        </span>
      )
    },
    {
      key: 'conditions',
      header: 'Conditions',
      width: TABLE_COLUMN_WIDTH.medium,
      render: (row) => {
        const count = row.rules?.conditions?.length || 0
        return <span>{count} {count === 1 ? 'condition' : 'conditions'}</span>
      }
    },
    {
      key: 'models',
      header: 'Models',
      width: TABLE_COLUMN_WIDTH.medium,
      render: (row) => {
        const count = row.modelRefs?.length || 0
        return <span>{count} {count === 1 ? 'model' : 'models'}</span>
      }
    }
  ]

  const renderDecisionModelRefSummary = (ref: DecisionConfig['modelRefs'][number], index: number) => {
    const badges = [
      ref.use_reasoning ? 'Reasoning enabled' : 'Standard inference',
      ref.reasoning_effort ? `Effort: ${ref.reasoning_effort}` : null,
      ref.lora_name ? `LoRA: ${ref.lora_name}` : null,
      typeof ref.weight === 'number' ? `Weight: ${ref.weight}` : null,
    ].filter((value): value is string => Boolean(value))

    const details = [
      ref.reasoning_description
        ? { label: 'Reasoning description', value: ref.reasoning_description }
        : null,
    ].filter((value): value is { label: string; value: string } => Boolean(value))

    return (
      <div key={`${ref.model}-${index}`} className={decisionStyles.viewCard}>
        <div className={decisionStyles.viewHeading}>
          <span className={decisionStyles.viewTitle}>{ref.model}</span>
          {badges.length > 0 ? (
            <div className={decisionStyles.viewBadgeRow}>
              {badges.map((badge) => (
                <span key={badge} className={decisionStyles.viewBadge}>
                  {badge}
                </span>
              ))}
            </div>
          ) : null}
        </div>
        {details.length > 0 ? (
          <div className={decisionStyles.viewMeta}>
            {details.map((detail) => (
              <div key={detail.label} className={decisionStyles.viewMetaRow}>
                <span className={decisionStyles.viewMetaLabel}>{detail.label}</span>
                <span className={decisionStyles.viewMetaValue}>{detail.value}</span>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    )
  }

  const handleViewDecision = (decision: DecisionRow) => {
    const sections: ViewSection[] = [
      {
        title: 'Basic Information',
        fields: [
          { label: 'Name', value: decision.name },
          { label: 'Priority', value: `P${decision.priority}` },
          { label: 'Description', value: decision.description || 'N/A', fullWidth: true }
        ]
      },
      {
        title: 'Rules',
        fields: [
          { label: 'Operator', value: decision.rules?.operator || 'N/A' },
          {
            label: 'Conditions',
            value: decision.rules?.conditions?.length ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {decision.rules.conditions.map((cond, i) => (
                  <div key={i} style={{
                    padding: '0.5rem',
                    background: 'rgba(118, 185, 0, 0.1)',
                    borderRadius: '4px',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.875rem'
                  }}>
                    {cond.type}: {cond.name}
                  </div>
                ))}
              </div>
            ) : 'No conditions',
            fullWidth: true
          }
        ]
      },
      {
        title: 'Models',
        fields: [
          {
            label: 'Model References',
            value: decision.modelRefs?.length ? (
              <div className={decisionStyles.viewStack}>
                {decision.modelRefs.map((ref, i) => renderDecisionModelRefSummary(ref, i))}
              </div>
            ) : 'No models',
            fullWidth: true
          }
        ]
      }
    ]

    if (decision.plugins && decision.plugins.length > 0) {
      sections.push({
        title: 'Plugins',
        fields: [
          {
            label: 'Configured Plugins',
            value: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {decision.plugins.map((plugin, i) => (
                  <div key={i} style={{
                    padding: '0.5rem',
                    background: 'rgba(147, 51, 234, 0.1)',
                    borderRadius: '4px',
                    fontSize: '0.875rem'
                  }}>
                    <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{plugin.type}</div>
                    <pre style={{
                      margin: '0.5rem 0 0',
                      padding: '0.5rem',
                      borderRadius: '4px',
                      background: 'rgba(0, 0, 0, 0.18)',
                      overflowX: 'auto',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.8rem',
                      whiteSpace: 'pre-wrap',
                    }}>
                      {JSON.stringify(plugin.configuration || {}, null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            ),
            fullWidth: true
          }
        ]
      })
    }

    openViewModal(`Decision: ${decision.name}`, sections, () => handleEditDecision(decision))
  }

  const openDecisionEditor = (mode: 'add' | 'edit', decision?: DecisionRow) => {
    const conditionTypeOptions = ['keyword', 'domain', 'preference', 'user_feedback', 'embedding', 'fact_check', 'language', 'context', 'complexity', 'modality', 'authz', 'jailbreak', 'pii', 'projection'] as const
    const projectionOutputs = (config?.projections?.mappings || []).flatMap((mapping) =>
      (mapping.outputs || []).map((output) => output.name)
    )

    const getConditionNameOptions = (type?: ConfigDecisionConditionType) => {
      switch (type) {
        case 'keyword':
          return config?.signals?.keywords?.map((k) => k.name) || []
        case 'domain':
          return config?.signals?.domains?.map((d) => d.name) || []
        case 'preference':
          return config?.signals?.preferences?.map((p) => p.name) || []
        case 'user_feedback':
          return config?.signals?.user_feedbacks?.map((u) => u.name) || []
        case 'embedding':
          return config?.signals?.embeddings?.map((e) => e.name) || []
        case 'fact_check':
          return config?.signals?.fact_check?.map((f) => f.name) || []
        case 'language':
          return config?.signals?.language?.map((l) => l.name) || []
        case 'context':
          return config?.signals?.context?.map((c) => c.name) || []
        case 'complexity':
          return (config?.signals?.complexity || []).flatMap((signal) => [
            `${signal.name}:easy`,
            `${signal.name}:medium`,
            `${signal.name}:hard`,
          ])
        case 'modality':
          return config?.signals?.modality?.map((m) => m.name) || []
        case 'authz':
          return config?.signals?.role_bindings?.map((binding) => binding.name) || []
        case 'jailbreak':
          return config?.signals?.jailbreak?.map((rule) => rule.name) || []
        case 'pii':
          return config?.signals?.pii?.map((rule) => rule.name) || []
        case 'projection':
          return projectionOutputs
        default:
          return []
      }
    }

    const defaultForm: DecisionFormState = {
      name: '',
      description: '',
      priority: 1,
      operator: 'AND',
      conditions: [{ type: 'keyword', name: '' }],
      modelRefs: [
        {
          model: '',
          use_reasoning: false,
          reasoning_description: '',
          reasoning_effort: '',
          lora_name: '',
        },
      ],
      plugins: []
    }

    const initialData: DecisionFormState = mode === 'edit' && decision ? {
      name: decision.name,
      description: decision.description || '',
      priority: decision.priority ?? 1,
      operator: decision.rules?.operator || 'AND',
      conditions: (decision.rules?.conditions || []).map((cond) => ({
        type: cond.type,
        name: cond.name
      })),
      modelRefs: (decision.modelRefs || []).map((ref) => ({
        model: ref.model,
        use_reasoning: !!ref.use_reasoning,
        reasoning_description: ref.reasoning_description || '',
        reasoning_effort: ref.reasoning_effort || '',
        lora_name: ref.lora_name || '',
        weight: typeof ref.weight === 'number' ? ref.weight : undefined,
      })),
      plugins: (decision.plugins || []).map((plugin) => ({
        type: plugin.type,
        configuration: JSON.stringify(plugin.configuration || {}, null, 2)
      }))
    } : defaultForm

    const renderConditionsEditor = (
      value: DecisionFormState['conditions'],
      onChange: (value: DecisionFormState['conditions']) => void
    ) => {
      const rows = (Array.isArray(value) ? value : []).length ? value : [{ type: 'keyword', name: '' }]

      const updateItem = (index: number, key: 'type' | 'name', val: string) => {
        const next = rows.map((item, idx) => {
          if (idx !== index) return item
          if (key === 'type') {
            return { type: val, name: '' }
          }
          return { ...item, [key]: val }
        })
        onChange(next)
      }

      const removeItem = (index: number) => {
        const next = rows.filter((_, idx) => idx !== index)
        onChange(next.length ? next : [{ type: 'keyword', name: '' }])
      }

      const addItem = () => onChange([...rows, { type: 'keyword', name: '' }])

      return (
        <div className={decisionStyles.editorList}>
          {rows.map((cond, idx) => (
            <div
              key={idx}
              className={decisionStyles.editorGridConditions}
            >
              <label className={decisionStyles.editorControlLabel}>
                <span className={decisionStyles.editorControlLabelText}>Signal type</span>
                <select
                  value={cond?.type || conditionTypeOptions[0]}
                  onChange={(e) => updateItem(idx, 'type', e.target.value)}
                  className={decisionStyles.editorSelect}
                >
                  {conditionTypeOptions.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </label>
              <label className={decisionStyles.editorControlLabel}>
                <span className={decisionStyles.editorControlLabelText}>Signal name</span>
                <select
                  value={cond?.name || ''}
                  onChange={(e) => updateItem(idx, 'name', e.target.value)}
                  className={decisionStyles.editorSelect}
                >
                  <option value="" disabled>Select name</option>
                  {getConditionNameOptions(cond?.type as ConfigDecisionConditionType).map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                  {getConditionNameOptions(cond?.type as ConfigDecisionConditionType).length === 0 && (
                    <option value="" disabled>No matching signals</option>
                  )}
                </select>
              </label>
              <button
                type="button"
                onClick={() => removeItem(idx)}
                className={decisionStyles.editorButtonSecondary}
              >
                Remove
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={addItem}
            className={decisionStyles.editorButtonSecondary}
          >
            Add Condition
          </button>
        </div>
      )
    }

    const renderModelRefsEditor = (
      value: DecisionFormState['modelRefs'],
      onChange: (value: DecisionFormState['modelRefs']) => void
    ) => {
      const modelOptions = models.map((model) => model.name)
      const rows = (Array.isArray(value) ? value : []).length
        ? value
        : [{ model: '', use_reasoning: false, reasoning_description: '', reasoning_effort: '', lora_name: '' }]

      const updateItem = (
        index: number,
        key: 'model' | 'use_reasoning' | 'reasoning_description' | 'reasoning_effort' | 'lora_name' | 'weight',
        val: string | boolean | number | undefined
      ) => {
        const next = rows.map((item, idx) => idx === index ? { ...item, [key]: val } : item)
        onChange(next)
      }

      const removeItem = (index: number) => {
        const next = rows.filter((_, idx) => idx !== index)
        onChange(next.length ? next : [{ model: '', use_reasoning: false, reasoning_description: '', reasoning_effort: '', lora_name: '' }])
      }

      const addItem = () =>
        onChange([
          ...rows,
          { model: '', use_reasoning: false, reasoning_description: '', reasoning_effort: '', lora_name: '' },
        ])

      return (
        <div className={decisionStyles.editorList}>
          {rows.map((ref, idx) => (
            <div key={idx} className={decisionStyles.editorCard}>
              <div className={decisionStyles.editorGridTwo}>
                <label className={decisionStyles.editorControlLabel}>
                  <span className={decisionStyles.editorControlLabelText}>Model</span>
                  <select
                    value={ref?.model || ''}
                    onChange={(e) => updateItem(idx, 'model', e.target.value)}
                    className={decisionStyles.editorSelect}
                  >
                    <option value="">Select model</option>
                    {ref?.model && !modelOptions.includes(ref.model) ? (
                      <option value={ref.model}>{ref.model}</option>
                    ) : null}
                    {modelOptions.map((option) => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                </label>
                <label className={decisionStyles.editorControlLabel}>
                  <span className={decisionStyles.editorControlLabelText}>Reasoning effort</span>
                  <select
                    value={ref?.reasoning_effort || ''}
                    onChange={(e) => updateItem(idx, 'reasoning_effort', e.target.value)}
                    className={decisionStyles.editorSelect}
                  >
                    <option value="">Default effort</option>
                    <option value="low">low</option>
                    <option value="medium">medium</option>
                    <option value="high">high</option>
                  </select>
                </label>
              </div>

              <div className={decisionStyles.editorMetaRow}>
                <label className={decisionStyles.editorCheckbox}>
                  <input
                    type="checkbox"
                    checked={!!ref?.use_reasoning}
                    onChange={(e) => updateItem(idx, 'use_reasoning', e.target.checked)}
                  />
                  Use reasoning
                </label>
                <button
                  type="button"
                  onClick={() => removeItem(idx)}
                  className={decisionStyles.editorButtonDanger}
                >
                  Remove model reference
                </button>
              </div>

              <div className={decisionStyles.editorGridTwo}>
                <label className={decisionStyles.editorControlLabel}>
                  <span className={decisionStyles.editorControlLabelText}>LoRA adapter</span>
                  <input
                    type="text"
                    value={ref?.lora_name || ''}
                    onChange={(e) => updateItem(idx, 'lora_name', e.target.value)}
                    placeholder="Optional adapter name"
                    className={decisionStyles.editorInput}
                  />
                </label>
                <label className={decisionStyles.editorControlLabel}>
                  <span className={decisionStyles.editorControlLabelText}>Weight</span>
                  <input
                    type="number"
                    value={typeof ref?.weight === 'number' ? ref.weight : ''}
                    onChange={(e) => updateItem(idx, 'weight', e.target.value === '' ? undefined : Number(e.target.value))}
                    placeholder="Optional weight"
                    step="0.1"
                    min="0"
                    className={decisionStyles.editorInput}
                  />
                </label>
              </div>

              <label className={decisionStyles.editorControlLabel}>
                <span className={decisionStyles.editorControlLabelText}>Reasoning description</span>
                <input
                  type="text"
                  value={ref?.reasoning_description || ''}
                  onChange={(e) => updateItem(idx, 'reasoning_description', e.target.value)}
                  placeholder="Optional operator note or reasoning hint"
                  className={decisionStyles.editorInput}
                />
              </label>
            </div>
          ))}
          <button
            type="button"
            onClick={addItem}
            className={decisionStyles.editorButtonSecondary}
          >
            Add Model Reference
          </button>
        </div>
      )
    }

    const renderPluginsEditor = (
      value: DecisionFormState['plugins'],
      onChange: (value: DecisionFormState['plugins']) => void
    ) => {
      const rows = Array.isArray(value) ? value : []

      const updateItem = (index: number, key: 'type' | 'configuration', val: string | DecisionPluginConfiguration) => {
        const next = rows.map((item, idx) => idx === index ? { ...item, [key]: val } : item)
        onChange(next)
      }

      const removeItem = (index: number) => {
        const next = rows.filter((_, idx) => idx !== index)
        onChange(next)
      }

      const addItem = () => onChange([...rows, { type: '', configuration: '' }])

      return (
        <div className={decisionStyles.editorList}>
          {rows.map((plugin, idx) => (
            <div key={idx} className={decisionStyles.editorCard}>
              <div className={decisionStyles.editorMetaRow}>
                <label className={decisionStyles.editorControlLabel} style={{ flex: 1 }}>
                  <span className={decisionStyles.editorControlLabelText}>Plugin type</span>
                  <input
                    type="text"
                    value={plugin?.type || ''}
                    onChange={(e) => updateItem(idx, 'type', e.target.value)}
                    placeholder="Plugin type (e.g. logging)"
                    className={decisionStyles.editorInput}
                  />
                </label>
                <button
                  type="button"
                  onClick={() => removeItem(idx)}
                  className={decisionStyles.editorButtonDanger}
                >
                  Remove
                </button>
              </div>
              <label className={decisionStyles.editorControlLabel}>
                <span className={decisionStyles.editorControlLabelText}>Configuration JSON</span>
                <textarea
                  value={typeof plugin?.configuration === 'string' ? plugin.configuration : JSON.stringify(plugin?.configuration || {}, null, 2)}
                  onChange={(e) => updateItem(idx, 'configuration', e.target.value)}
                  placeholder='Configuration JSON (optional)'
                  rows={4}
                  className={decisionStyles.editorTextarea}
                />
              </label>
            </div>
          ))}
          <button
            type="button"
            onClick={addItem}
            className={decisionStyles.editorButtonSecondary}
          >
            Add Plugin
          </button>
        </div>
      )
    }

    const fields: FieldConfig<DecisionFormState>[] = [
      {
        name: 'name',
        label: 'Name',
        type: 'text',
        required: true,
        placeholder: 'Enter a unique decision name'
      },
      {
        name: 'description',
        label: 'Description',
        type: 'textarea',
        placeholder: 'What does this decision route?'
      },
      {
        name: 'priority',
        label: 'Priority',
        type: 'number',
        min: 0,
        placeholder: '1'
      },
      {
        name: 'operator',
        label: 'Rules Operator',
        type: 'select',
        options: ['AND', 'OR', 'NOT'],
        description: 'AND: all conditions must match. OR: any condition matches. NOT: none of the conditions must match (exclusion routing).',
        required: true
      },
      {
        name: 'conditions',
        label: 'Conditions',
        type: 'custom',
        description: 'Add routing conditions (type and name).',
        customRender: (value, onChange) => renderConditionsEditor(
          Array.isArray(value) ? value as DecisionFormState['conditions'] : [],
          (nextValue) => onChange(nextValue)
        )
      },
      {
        name: 'modelRefs',
        label: 'Model References',
        type: 'custom',
        description: 'Set target models and whether to enable reasoning.',
        customRender: (value, onChange) => renderModelRefsEditor(
          Array.isArray(value) ? value as DecisionFormState['modelRefs'] : [],
          (nextValue) => onChange(nextValue)
        )
      },
      {
        name: 'plugins',
        label: 'Plugins',
        type: 'custom',
        description: 'Optional plugins applied to this decision.',
        customRender: (value, onChange) => renderPluginsEditor(
          Array.isArray(value) ? value as DecisionFormState['plugins'] : [],
          (nextValue) => onChange(nextValue)
        )
      }
    ]

    const saveDecision = async (formData: DecisionFormState) => {
      if (!config) {
        throw new Error('Configuration not loaded yet.')
      }

      if (!isPythonCLI) {
        throw new Error('Decisions are only supported for Python CLI configs.')
      }

      const name = (formData.name || '').trim()
      if (!name) {
        throw new Error('Name is required.')
      }

      const priority = Number.isFinite(formData.priority) ? formData.priority : 0

      const normalizedConditions = (formData.conditions || []).filter((c) => (c?.type || '').trim() || (c?.name || '').trim())
      const conditions = normalizedConditions.map((condition, idx) => {
        const type = (condition?.type || '').trim()
        const conditionName = (condition?.name || '').trim()
        if (!type || !conditionName) {
          throw new Error(`Condition #${idx + 1} needs both type and name.`)
        }
        return { type, name: conditionName }
      })

      const normalizedModelRefs = (formData.modelRefs || []).filter((m) => (m?.model || '').trim())
      const modelRefs = normalizedModelRefs.map((modelRefValue, idx) => {
        const model = (modelRefValue?.model || '').trim()
        if (!model) {
          throw new Error(`Model reference #${idx + 1} is missing a model name.`)
        }
        const modelRef: DecisionConfig['modelRefs'][number] = {
          model,
          use_reasoning: !!modelRefValue?.use_reasoning,
        }
        const reasoningDescription = (modelRefValue?.reasoning_description || '').trim()
        if (reasoningDescription) {
          modelRef.reasoning_description = reasoningDescription
        }
        const reasoningEffort = (modelRefValue?.reasoning_effort || '').trim()
        if (reasoningEffort) {
          modelRef.reasoning_effort = reasoningEffort
        }
        const loraName = (modelRefValue?.lora_name || '').trim()
        if (loraName) {
          modelRef.lora_name = loraName
        }
        if (typeof modelRefValue?.weight === 'number' && Number.isFinite(modelRefValue.weight)) {
          modelRef.weight = modelRefValue.weight
        }
        return modelRef
      })

      const normalizedPlugins = (formData.plugins || []).filter((p) => {
        const hasType = (p?.type || '').trim()
        const hasConfigString = typeof p?.configuration === 'string' && (p.configuration as string).trim()
        const hasConfigObject = p?.configuration && typeof p.configuration === 'object'
        return hasType || hasConfigString || hasConfigObject
      })

      const plugins = normalizedPlugins.map((pluginValue, idx) => {
        const type = (pluginValue?.type || '').trim()
        if (!type) {
          throw new Error(`Plugin #${idx + 1} must include a type.`)
        }

        let configuration: DecisionPluginConfiguration = {}
        if (typeof pluginValue?.configuration === 'string') {
          const trimmed = pluginValue.configuration.trim()
          if (trimmed) {
            try {
              configuration = JSON.parse(trimmed)
            } catch {
              throw new Error(`Plugin #${idx + 1} configuration must be valid JSON.`)
            }
          }
        } else if (pluginValue?.configuration && typeof pluginValue.configuration === 'object') {
          configuration = pluginValue.configuration as DecisionPluginConfiguration
        }

        return { type, configuration }
      })

      const newDecision: DecisionConfig = {
        name,
        description: formData.description,
        priority: priority || 0,
        rules: {
          operator: formData.operator,
          conditions
        },
        modelRefs,
        plugins
      }

      const newConfig: ConfigData = cloneConfigData(config)
      newConfig.decisions = [...(newConfig.decisions || [])]

      if (mode === 'edit' && decision) {
        removeDecisionByName(newConfig, decision.name)
      }

      newConfig.decisions.push(newDecision)
      await saveConfig(newConfig)
    }

    openEditModal<DecisionFormState>(
      mode === 'add' ? 'Add Decision' : `Edit Decision: ${decision?.name}`,
      initialData,
      fields,
      saveDecision,
      mode
    )
  }

  const handleEditDecision = (decision: DecisionRow) => {
    openDecisionEditor('edit', decision)
  }

  const handleDeleteDecision = async (decision: DecisionConfig) => {
    if (!confirm(`Are you sure you want to delete decision "${decision.name}"?`)) {
      return
    }

    if (!config || !isPythonCLI) {
      alert('Deleting decisions is only supported for Python CLI configs.')
      return
    }

    const newConfig: ConfigData = cloneConfigData(config)
    removeDecisionByName(newConfig, decision.name)
    await saveConfig(newConfig)
  }

  return (
    <ConfigPageManagerLayout title="Decisions" description="Shape routing outcomes with ordered rules and plugins that map signals to concrete model behavior.">
      <div className={styles.sectionPanel}>
        <div className={styles.sectionTableBlock}>
          <TableHeader
            title="Routing Decisions"
            count={decisions.length}
            searchPlaceholder="Search decisions..."
            searchValue={decisionsSearch}
            onSearchChange={onDecisionsSearchChange}
            onAdd={() => openDecisionEditor('add')}
            addButtonText="Add Decision" disabled={isReadonly} variant="embedded"
          />
          <DataTable
            columns={decisionsColumns}
            data={filteredDecisions}
            keyExtractor={(row) => row.name}
            onView={handleViewDecision}
            onEdit={handleEditDecision}
            onDelete={handleDeleteDecision}
            emptyMessage={decisionsSearch ? 'No decisions match your search' : 'No routing decisions configured'}
            className={styles.managerTable}
            readonly={isReadonly}
          />
        </div>
      </div>
    </ConfigPageManagerLayout>
  )
}
