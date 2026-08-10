import { useEffect, useState } from 'react'

import styles from './ConfigPage.module.css'
import decisionStyles from './ConfigPageDecisionsSection.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfirmDialog from '../components/ConfirmDialog'
import RoutingScopeSelector from '../components/RoutingScopeSelector'
import TableHeader from '../components/TableHeader'
import { DataTable } from '../components/DataTable'
import DecisionRuleEditor from '../components/DecisionRuleEditor'
import type { FieldConfig } from '../components/EditModal'
import type { ViewSection } from '../components/ViewModal'
import type {
  ConfigData,
  DecisionConfig,
  DecisionFormState,
  DecisionPluginConfiguration,
  NormalizedModel,
} from './configPageSupport'
import { cloneDecisionRuleSet, mergeDecisionForSave } from './configPageSupport'
import { buildAvailableSignals } from './configPageDecisionSignalCatalog'
import { validateDecisionRules } from './configPageDecisionRuleBridge'
import { DecisionConditionView } from './configPageDecisionRuleView'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { cloneConfigData } from './configPageCanonicalization'
import ConfigPageDecisionPluginsEditor from './ConfigPageDecisionPluginsEditor'
import { useRoutingScopeManager } from './configPageRoutingScopeSupport'
import { decisionColumns } from './configPageDecisionTable'

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

type DecisionRow = DecisionConfig

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
  const [decisionPendingDelete, setDecisionPendingDelete] = useState<DecisionConfig | null>(null)
  const [decisionDeletePending, setDecisionDeletePending] = useState(false)
  const [decisionDeleteError, setDecisionDeleteError] = useState<string | null>(null)
  const {
    applyScopedConfig,
    routingScopes,
    scopedConfig,
    selectedScope,
    selectedScopeId,
    setSelectedScopeId,
  } = useRoutingScopeManager(config)
  useEffect(() => {
    setDecisionPendingDelete(null)
    setDecisionDeleteError(null)
  }, [selectedScopeId])
  const decisions = scopedConfig?.decisions || []

  const filteredDecisions = decisions.filter(
    (decision) =>
      decision.name.toLowerCase().includes(decisionsSearch.toLowerCase()) ||
      decision.description?.toLowerCase().includes(decisionsSearch.toLowerCase()),
  )

  const renderDecisionModelRefSummary = (
    ref: DecisionConfig['modelRefs'][number],
    index: number,
  ) => {
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
          { label: 'Description', value: decision.description || 'N/A', fullWidth: true },
        ],
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
                  <DecisionConditionView key={i} condition={cond} depth={0} />
                ))}
              </div>
            ) : (
              'No conditions'
            ),
            fullWidth: true,
          },
        ],
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
            ) : (
              'No models'
            ),
            fullWidth: true,
          },
        ],
      },
    ]

    if (decision.plugins && decision.plugins.length > 0) {
      sections.push({
        title: 'Plugins',
        fields: [
          {
            label: 'Configured Plugins',
            value: (
              <div className={decisionStyles.viewStack}>
                {decision.plugins.map((plugin, i) => (
                  <article key={`${plugin.type}-${i}`} className={decisionStyles.viewCard}>
                    <div className={decisionStyles.viewHeading}>
                      <span className={decisionStyles.viewTitle}>{plugin.type}</span>
                      <span className={decisionStyles.viewBadge}>
                        {Object.keys(plugin.configuration || {}).length} configured fields
                      </span>
                    </div>
                    <div className={decisionStyles.viewMeta}>
                      {Object.entries(plugin.configuration || {}).map(([key, value]) => (
                        <div key={key} className={decisionStyles.viewMetaRow}>
                          <span className={decisionStyles.viewMetaLabel}>
                            {key.replace(/_/g, ' ')}
                          </span>
                          <span className={decisionStyles.viewMetaValue}>
                            {Array.isArray(value)
                              ? `${value.length} items`
                              : value && typeof value === 'object'
                                ? `${Object.keys(value).length} fields`
                                : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </article>
                ))}
              </div>
            ),
            fullWidth: true,
          },
        ],
      })
    }

    openViewModal(`Decision: ${decision.name}`, sections, () => handleEditDecision(decision))
  }

  const openDecisionEditor = (mode: 'add' | 'edit', decision?: DecisionRow) => {
    const availableSignals = buildAvailableSignals(config?.signals, config?.projections)

    const defaultForm: DecisionFormState = {
      name: '',
      description: '',
      priority: 1,
      rules: { operator: 'AND', conditions: [] },
      modelRefs: [
        {
          model: '',
          use_reasoning: false,
          reasoning_description: '',
          reasoning_effort: '',
          lora_name: '',
        },
      ],
      plugins: [],
    }

    const initialData: DecisionFormState =
      mode === 'edit' && decision
        ? {
            name: decision.name,
            description: decision.description || '',
            priority: decision.priority ?? 1,
            rules: cloneDecisionRuleSet(decision.rules),
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
              configuration: JSON.stringify(plugin.configuration || {}, null, 2),
            })),
          }
        : defaultForm

    const renderModelRefsEditor = (
      value: DecisionFormState['modelRefs'],
      onChange: (value: DecisionFormState['modelRefs']) => void,
    ) => {
      const modelOptions = models.map((model) => model.name)
      const rows = (Array.isArray(value) ? value : []).length
        ? value
        : [
            {
              model: '',
              use_reasoning: false,
              reasoning_description: '',
              reasoning_effort: '',
              lora_name: '',
            },
          ]

      const updateItem = (
        index: number,
        key:
          | 'model'
          | 'use_reasoning'
          | 'reasoning_description'
          | 'reasoning_effort'
          | 'lora_name'
          | 'weight',
        val: string | boolean | number | undefined,
      ) => {
        const next = rows.map((item, idx) => (idx === index ? { ...item, [key]: val } : item))
        onChange(next)
      }

      const removeItem = (index: number) => {
        const next = rows.filter((_, idx) => idx !== index)
        onChange(
          next.length
            ? next
            : [
                {
                  model: '',
                  use_reasoning: false,
                  reasoning_description: '',
                  reasoning_effort: '',
                  lora_name: '',
                },
              ],
        )
      }

      const addItem = () =>
        onChange([
          ...rows,
          {
            model: '',
            use_reasoning: false,
            reasoning_description: '',
            reasoning_effort: '',
            lora_name: '',
          },
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
                      <option key={option} value={option}>
                        {option}
                      </option>
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
                    onChange={(e) =>
                      updateItem(
                        idx,
                        'weight',
                        e.target.value === '' ? undefined : Number(e.target.value),
                      )
                    }
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
          <button type="button" onClick={addItem} className={decisionStyles.editorButtonSecondary}>
            Add Model Reference
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
        placeholder: 'Enter a unique decision name',
      },
      {
        name: 'description',
        label: 'Description',
        type: 'textarea',
        placeholder: 'What does this decision route?',
      },
      {
        name: 'priority',
        label: 'Priority',
        type: 'number',
        min: 0,
        placeholder: '1',
      },
      {
        name: 'rules',
        label: 'Conditions',
        type: 'custom',
        description:
          'Build the routing rule with AND / OR / NOT groups. This is the same expression builder used for route conditions.',
        customRender: (value, onChange) => (
          <DecisionRuleEditor
            value={value as DecisionFormState['rules']}
            onChange={(nextValue) => onChange(nextValue)}
            availableSignals={availableSignals}
          />
        ),
      },
      {
        name: 'modelRefs',
        label: 'Model References',
        type: 'custom',
        description: 'Set target models and whether to enable reasoning.',
        customRender: (value, onChange) =>
          renderModelRefsEditor(
            Array.isArray(value) ? (value as DecisionFormState['modelRefs']) : [],
            (nextValue) => onChange(nextValue),
          ),
      },
      {
        name: 'plugins',
        label: 'Plugins',
        type: 'custom',
        description: 'Optional plugins applied to this decision.',
        customRender: (value, onChange) => (
          <ConfigPageDecisionPluginsEditor
            value={Array.isArray(value) ? (value as DecisionFormState['plugins']) : []}
            onChange={(nextValue) => onChange(nextValue)}
          />
        ),
      },
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

      const ruleWarnings = validateDecisionRules(formData.rules, availableSignals)
      if (ruleWarnings.length > 0) {
        throw new Error(ruleWarnings.join(' '))
      }

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
        const hasConfigString =
          typeof p?.configuration === 'string' && (p.configuration as string).trim()
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

      const newDecision = mergeDecisionForSave(mode === 'edit' ? decision : undefined, {
        name,
        description: formData.description,
        priority: priority || 0,
        rules: formData.rules,
        modelRefs,
        plugins,
      })

      if (!scopedConfig) {
        throw new Error('Routing profile not loaded yet.')
      }
      const newConfig: ConfigData = cloneConfigData(scopedConfig)
      newConfig.decisions = [...(newConfig.decisions || [])]

      if (mode === 'edit' && decision) {
        removeDecisionByName(newConfig, decision.name)
      }

      newConfig.decisions.push(newDecision)
      await saveConfig(applyScopedConfig(newConfig))
    }

    openEditModal<DecisionFormState>(
      mode === 'add' ? 'Add Decision' : `Edit Decision: ${decision?.name}`,
      initialData,
      fields,
      saveDecision,
      mode,
    )
  }

  const handleEditDecision = (decision: DecisionRow) => {
    openDecisionEditor('edit', decision)
  }

  const handleDeleteDecision = (decision: DecisionConfig) => {
    setDecisionDeleteError(null)
    setDecisionPendingDelete(decision)
  }

  const confirmDeleteDecision = async () => {
    if (!decisionPendingDelete) return
    if (!config || !isPythonCLI) {
      setDecisionDeleteError('Deleting decisions is only supported for Python CLI configs.')
      return
    }

    setDecisionDeletePending(true)
    setDecisionDeleteError(null)
    try {
      if (!scopedConfig) {
        throw new Error('Routing profile not loaded yet.')
      }
      const newConfig: ConfigData = cloneConfigData(scopedConfig)
      removeDecisionByName(newConfig, decisionPendingDelete.name)
      await saveConfig(applyScopedConfig(newConfig))
      setDecisionPendingDelete(null)
    } catch (error) {
      setDecisionDeleteError(error instanceof Error ? error.message : 'Failed to delete decision.')
    } finally {
      setDecisionDeletePending(false)
    }
  }

  return (
    <ConfigPageManagerLayout
      title="Decisions"
      description="Shape routing outcomes with ordered rules and plugins that map signals to concrete model behavior."
      scope={selectedScope?.label ?? 'Routing profile'}
    >
      <div className={styles.sectionPanel}>
        <div className={styles.sectionTableBlock}>
          <RoutingScopeSelector
            scopes={routingScopes}
            value={selectedScopeId}
            onChange={setSelectedScopeId}
          />
          <TableHeader
            title="Routing Decisions"
            count={decisions.length}
            searchPlaceholder="Search decisions..."
            searchValue={decisionsSearch}
            onSearchChange={onDecisionsSearchChange}
            onAdd={() => openDecisionEditor('add')}
            addButtonText="Add Decision"
            disabled={isReadonly}
            variant="embedded"
          />
          <DataTable
            columns={decisionColumns}
            data={filteredDecisions}
            keyExtractor={(row) => row.name}
            onView={handleViewDecision}
            onEdit={handleEditDecision}
            onDelete={handleDeleteDecision}
            emptyMessage={
              decisionsSearch ? 'No decisions match your search' : 'No routing decisions configured'
            }
            className={styles.managerTable}
            readonly={isReadonly}
          />
        </div>
      </div>

      <ConfirmDialog
        isOpen={decisionPendingDelete !== null}
        title={`Delete decision “${decisionPendingDelete?.name || ''}”?`}
        description="Remove this decision from the active routing configuration. This change cannot be undone from the dashboard."
        eyebrow="Destructive configuration change"
        confirmLabel="Delete decision"
        pending={decisionDeletePending}
        details={decisionDeleteError ? <span role="alert">{decisionDeleteError}</span> : undefined}
        onCancel={() => {
          if (decisionDeletePending) return
          setDecisionPendingDelete(null)
          setDecisionDeleteError(null)
        }}
        onConfirm={confirmDeleteDecision}
      />
    </ConfigPageManagerLayout>
  )
}
