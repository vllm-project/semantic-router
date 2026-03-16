import styles from './ConfigPage.module.css'
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
  defaultModel: string
  onOpenPresetModal: () => void
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
  defaultModel,
  onOpenPresetModal,
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
        <span className={styles.badge} style={{ background: 'rgba(0, 212, 255, 0.15)', color: 'var(--color-accent-cyan)' }}>
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
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {decision.modelRefs.map((ref, i) => (
                  <div key={i} style={{
                    padding: '0.5rem',
                    background: 'rgba(0, 212, 255, 0.1)',
                    borderRadius: '4px',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.875rem'
                  }}>
                    {ref.model} {ref.use_reasoning && '(with reasoning)'}
                  </div>
                ))}
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
    const conditionTypeOptions = ['keyword', 'domain', 'preference', 'user_feedback', 'embedding', 'fact_check', 'language', 'context', 'complexity', 'modality', 'authz', 'jailbreak', 'pii'] as const

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
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {rows.map((cond, idx) => (
            <div
              key={idx}
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr auto',
                gap: '0.5rem',
                alignItems: 'center'
              }}
            >
              <select
                value={cond?.type || conditionTypeOptions[0]}
                onChange={(e) => updateItem(idx, 'type', e.target.value)}
                style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
              >
                {conditionTypeOptions.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
              <select
                value={cond?.name || ''}
                onChange={(e) => updateItem(idx, 'name', e.target.value)}
                style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
              >
                <option value="" disabled>Select name</option>
                {getConditionNameOptions(cond?.type as ConfigDecisionConditionType).map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
                {getConditionNameOptions(cond?.type as ConfigDecisionConditionType).length === 0 && (
                  <option value="" disabled>No matching signals</option>
                )}
              </select>
              <button
                type="button"
                onClick={() => removeItem(idx)}
                style={{
                  padding: '0.5rem 0.75rem',
                  borderRadius: 6,
                  border: '1px solid var(--color-border)',
                  background: 'transparent',
                  color: 'var(--color-text)'
                }}
              >
                Remove
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={addItem}
            style={{
              width: 'fit-content',
              padding: '0.5rem 0.75rem',
              borderRadius: 6,
              border: '1px solid var(--color-border)',
              background: 'transparent',
              color: 'var(--color-text)'
            }}
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
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {rows.map((ref, idx) => (
            <div
              key={idx}
              style={{
                display: 'grid',
                gap: '0.75rem',
                padding: '0.75rem',
                borderRadius: 8,
                border: '1px solid var(--color-border)'
              }}
            >
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'minmax(220px, 1fr) minmax(180px, 220px) auto auto',
                  gap: '0.5rem',
                  alignItems: 'center'
                }}
              >
                <select
                  value={ref?.model || ''}
                  onChange={(e) => updateItem(idx, 'model', e.target.value)}
                  style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                >
                  <option value="">Select model</option>
                  {ref?.model && !modelOptions.includes(ref.model) ? (
                    <option value={ref.model}>{ref.model}</option>
                  ) : null}
                  {modelOptions.map((option) => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
                <select
                  value={ref?.reasoning_effort || ''}
                  onChange={(e) => updateItem(idx, 'reasoning_effort', e.target.value)}
                  style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                >
                  <option value="">Default effort</option>
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high</option>
                </select>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--color-text-secondary)' }}>
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
                  style={{
                    padding: '0.5rem 0.75rem',
                    borderRadius: 6,
                    border: '1px solid var(--color-border)',
                    background: 'transparent',
                    color: 'var(--color-text)'
                  }}
                >
                  Remove
                </button>
              </div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'minmax(220px, 1fr) minmax(140px, 180px)',
                  gap: '0.5rem',
                  alignItems: 'center'
                }}
              >
                <input
                  type="text"
                  value={ref?.lora_name || ''}
                  onChange={(e) => updateItem(idx, 'lora_name', e.target.value)}
                  placeholder="LoRA adapter name (optional)"
                  style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                />
                <input
                  type="number"
                  value={typeof ref?.weight === 'number' ? ref.weight : ''}
                  onChange={(e) => updateItem(idx, 'weight', e.target.value === '' ? undefined : Number(e.target.value))}
                  placeholder="Weight"
                  step="0.1"
                  min="0"
                  style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                />
              </div>
              <input
                type="text"
                value={ref?.reasoning_description || ''}
                onChange={(e) => updateItem(idx, 'reasoning_description', e.target.value)}
                placeholder="Reasoning description (optional)"
                style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
              />
            </div>
          ))}
          <button
            type="button"
            onClick={addItem}
            style={{
              width: 'fit-content',
              padding: '0.5rem 0.75rem',
              borderRadius: 6,
              border: '1px solid var(--color-border)',
              background: 'transparent',
              color: 'var(--color-text)'
            }}
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

      const updateItem = (index: number, key: 'type' | 'configuration', val: string | Record<string, unknown>) => {
        const next = rows.map((item, idx) => idx === index ? { ...item, [key]: val } : item)
        onChange(next)
      }

      const removeItem = (index: number) => {
        const next = rows.filter((_, idx) => idx !== index)
        onChange(next)
      }

      const addItem = () => onChange([...rows, { type: '', configuration: '' }])

      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {rows.map((plugin, idx) => (
            <div
              key={idx}
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr',
                gap: '0.5rem',
                padding: '0.75rem',
                borderRadius: 8,
                border: '1px solid var(--color-border)'
              }}
            >
              <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '0.5rem', alignItems: 'center' }}>
                <input
                  type="text"
                  value={plugin?.type || ''}
                  onChange={(e) => updateItem(idx, 'type', e.target.value)}
                  placeholder="Plugin type (e.g. logging)"
                  style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                />
                <button
                  type="button"
                  onClick={() => removeItem(idx)}
                  style={{
                    padding: '0.5rem 0.75rem',
                    borderRadius: 6,
                    border: '1px solid var(--color-border)',
                    background: 'transparent',
                    color: 'var(--color-text)'
                  }}
                >
                  Remove
                </button>
              </div>
              <textarea
                value={typeof plugin?.configuration === 'string' ? plugin.configuration : JSON.stringify(plugin?.configuration || {}, null, 2)}
                onChange={(e) => updateItem(idx, 'configuration', e.target.value)}
                placeholder='Configuration JSON (optional)'
                rows={4}
                style={{
                  padding: '0.55rem 0.75rem',
                  borderRadius: 6,
                  border: '1px solid var(--color-border)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.9rem'
                }}
              />
            </div>
          ))}
          <button
            type="button"
            onClick={addItem}
            style={{
              width: 'fit-content',
              padding: '0.5rem 0.75rem',
              borderRadius: 6,
              border: '1px solid var(--color-border)',
              background: 'transparent',
              color: 'var(--color-text)'
            }}
          >
            Add Plugin
          </button>
        </div>
      )
    }

    const fields: FieldConfig[] = [
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
        customRender: renderConditionsEditor
      },
      {
        name: 'modelRefs',
        label: 'Model References',
        type: 'custom',
        description: 'Set target models and whether to enable reasoning.',
        customRender: renderModelRefsEditor
      },
      {
        name: 'plugins',
        label: 'Plugins',
        type: 'custom',
        description: 'Optional plugins applied to this decision.',
        customRender: renderPluginsEditor
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

        let configuration: Record<string, unknown> = {}
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
          configuration = pluginValue.configuration as Record<string, unknown>
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

    openEditModal(
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
    <ConfigPageManagerLayout title="Decisions" description="Shape routing outcomes with ordered rules, plugins, and presets that map signals to concrete model behavior.">
      <div className={styles.sectionPanel}>
        <div className={styles.sectionTableBlock}>
          <TableHeader
            title="Routing Decisions"
            count={decisions.length}
            searchPlaceholder="Search decisions..."
            searchValue={decisionsSearch}
            onSearchChange={onDecisionsSearchChange}
            onSecondaryAction={
              isPythonCLI && defaultModel
                ? onOpenPresetModal
                : undefined
            }
            secondaryActionText="Apply preset"
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
