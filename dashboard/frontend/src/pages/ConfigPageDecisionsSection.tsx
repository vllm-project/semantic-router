import { useEffect, useState } from 'react'

import styles from './ConfigPage.module.css'
import decisionStyles from './ConfigPageDecisionsSection.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfirmDialog from '../components/ConfirmDialog'
import RoutingScopeSelector from '../components/RoutingScopeSelector'
import TableHeader from '../components/TableHeader'
import { DataTable } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'
import type { ViewSection } from '../components/ViewModal'
import type {
  ConfigData,
  ConfigDecisionConditionType,
  DecisionConfig,
  DecisionFormState,
  DecisionPluginConfiguration,
} from './configPageSupport'
import {
  cloneDecisionConditions,
  conditionHasNestedRules,
  decisionRulesForSave,
  mergeDecisionForSave,
} from './configPageSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { cloneConfigData } from './configPageCanonicalization'
import ConfigPageDecisionPluginsEditor from './ConfigPageDecisionPluginsEditor'
import ConfigPageDecisionConditionsView from './ConfigPageDecisionConditionsView'
import { useRoutingScopeManager } from './configPageRoutingScopeSupport'
import { decisionColumns } from './configPageDecisionTable'
import ConfigPageRoutingScopeState from './ConfigPageRoutingScopeState'
import ProductIcon from '../components/ProductIcon'

interface ConfigPageDecisionsSectionProps {
  isReadonly: boolean
  decisionsSearch: string
  onDecisionsSearchChange: (value: string) => void
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
  removeDecisionByName: (cfg: ConfigData, targetName: string) => void
}

type DecisionRow = DecisionConfig

export default function ConfigPageDecisionsSection({
  isReadonly,
  decisionsSearch,
  onDecisionsSearchChange,
  openEditModal,
  openViewModal,
  removeDecisionByName,
}: ConfigPageDecisionsSectionProps) {
  const [decisionPendingDelete, setDecisionPendingDelete] = useState<DecisionConfig | null>(null)
  const [decisionDeletePending, setDecisionDeletePending] = useState(false)
  const [decisionDeleteError, setDecisionDeleteError] = useState<string | null>(null)
  const {
    error: routingScopeError,
    loading: routingScopeLoading,
    reload: reloadRoutingScopes,
    saveScopedConfig,
    routingScopes,
    scopedConfig,
    selectedScope,
    selectedScopeId,
    setSelectedScopeId,
    selectedRecipe,
  } = useRoutingScopeManager()
  const scopeReadonly = isReadonly || Boolean(selectedRecipe?.immutable)
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
            value: (
              <ConfigPageDecisionConditionsView conditions={decision.rules?.conditions ?? []} />
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

    openViewModal(
      `Decision: ${decision.name}`,
      sections,
      () => handleEditDecision(decision),
      scopeReadonly
        ? []
        : [
            {
              label: 'Delete decision',
              tone: 'destructive',
              onClick: () => handleDeleteDecision(decision),
            },
          ],
    )
  }

  const openDecisionEditor = (mode: 'add' | 'edit', decision?: DecisionRow) => {
    const conditionTypeOptions = [
      'keyword',
      'domain',
      'preference',
      'user_feedback',
      'reask',
      'embedding',
      'fact_check',
      'language',
      'context',
      'structure',
      'complexity',
      'modality',
      'authz',
      'jailbreak',
      'pii',
      'projection',
    ] as const
    const projectionOutputs = (scopedConfig?.projections?.mappings || []).flatMap((mapping) =>
      (mapping.outputs || []).map((output) => output.name),
    )

    const getConditionNameOptions = (type?: ConfigDecisionConditionType) => {
      switch (type) {
        case 'keyword':
          return scopedConfig?.signals?.keywords?.map((k) => k.name) || []
        case 'domain':
          return scopedConfig?.signals?.domains?.map((d) => d.name) || []
        case 'preference':
          return scopedConfig?.signals?.preferences?.map((p) => p.name) || []
        case 'user_feedback':
          return scopedConfig?.signals?.user_feedbacks?.map((u) => u.name) || []
        case 'reask':
          return scopedConfig?.signals?.reasks?.map((r) => r.name) || []
        case 'embedding':
          return scopedConfig?.signals?.embeddings?.map((e) => e.name) || []
        case 'fact_check':
          return scopedConfig?.signals?.fact_check?.map((f) => f.name) || []
        case 'language':
          return scopedConfig?.signals?.language?.map((l) => l.name) || []
        case 'context':
          return scopedConfig?.signals?.context?.map((c) => c.name) || []
        case 'structure':
          return scopedConfig?.signals?.structure?.map((s) => s.name) || []
        case 'complexity':
          return (scopedConfig?.signals?.complexity || []).flatMap((signal) => [
            `${signal.name}:easy`,
            `${signal.name}:medium`,
            `${signal.name}:hard`,
          ])
        case 'modality':
          return scopedConfig?.signals?.modality?.map((m) => m.name) || []
        case 'authz':
          return scopedConfig?.signals?.role_bindings?.map((binding) => binding.name) || []
        case 'jailbreak':
          return scopedConfig?.signals?.jailbreak?.map((rule) => rule.name) || []
        case 'pii':
          return scopedConfig?.signals?.pii?.map((rule) => rule.name) || []
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
      plugins: [],
    }

    const initialData: DecisionFormState =
      mode === 'edit' && decision
        ? {
            name: decision.name,
            description: decision.description || '',
            priority: decision.priority ?? 1,
            operator: decision.rules?.operator || 'AND',
            conditions: cloneDecisionConditions(decision.rules?.conditions),
            plugins: (decision.plugins || []).map((plugin) => ({
              type: plugin.type,
              configuration: JSON.stringify(plugin.configuration || {}, null, 2),
            })),
          }
        : defaultForm

    const renderConditionsEditor = (
      value: DecisionFormState['conditions'],
      onChange: (value: DecisionFormState['conditions']) => void,
    ) => {
      const rows = (Array.isArray(value) ? value : []).length
        ? value
        : [{ type: 'keyword', name: '' }]
      if (rows.some(conditionHasNestedRules)) {
        return (
          <p className={decisionStyles.editorHelp} role="note">
            Nested boolean rules are preserved unchanged. Use DSL mode to edit this rule tree.
          </p>
        )
      }

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
            <div key={idx} className={decisionStyles.editorGridConditions}>
              <label className={decisionStyles.editorControlLabel}>
                <span className={decisionStyles.editorControlLabelText}>Signal type</span>
                <select
                  value={cond?.type || conditionTypeOptions[0]}
                  onChange={(e) => updateItem(idx, 'type', e.target.value)}
                  className={decisionStyles.editorSelect}
                >
                  {conditionTypeOptions.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
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
                  <option value="" disabled>
                    Select name
                  </option>
                  {getConditionNameOptions(cond?.type as ConfigDecisionConditionType).map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                  {getConditionNameOptions(cond?.type as ConfigDecisionConditionType).length ===
                    0 && (
                    <option value="" disabled>
                      No matching signals
                    </option>
                  )}
                </select>
              </label>
              <button
                type="button"
                onClick={() => removeItem(idx)}
                className={decisionStyles.editorButtonSecondary}
              >
                <ProductIcon name="trash" aria-hidden="true" />
                Remove
              </button>
            </div>
          ))}
          <button type="button" onClick={addItem} className={decisionStyles.editorButtonSecondary}>
            <ProductIcon name="plus" aria-hidden="true" />
            Add condition
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
        name: 'operator',
        label: 'Rules Operator',
        type: 'select',
        options: ['AND', 'OR', 'NOT'],
        description:
          'AND: all conditions must match. OR: any condition matches. NOT: none of the conditions must match (exclusion routing).',
        required: true,
      },
      {
        name: 'conditions',
        label: 'Conditions',
        type: 'custom',
        description: 'Add routing conditions (type and name).',
        customRender: (value, onChange) =>
          renderConditionsEditor(
            Array.isArray(value) ? (value as DecisionFormState['conditions']) : [],
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
      const name = (formData.name || '').trim()
      if (!name) {
        throw new Error('Name is required.')
      }

      const priority = Number.isFinite(formData.priority) ? formData.priority : 0

      const normalizedConditions = (formData.conditions || []).filter(
        (c) => (c?.type || '').trim() || (c?.name || '').trim(),
      )
      const conditions = normalizedConditions.map((condition, idx) => {
        const type = (condition?.type || '').trim()
        const conditionName = (condition?.name || '').trim()
        if (!type || !conditionName) {
          throw new Error(`Condition #${idx + 1} needs both type and name.`)
        }
        return {
          type,
          name: conditionName,
          ...(condition.label ? { label: condition.label } : {}),
          ...(condition.predicate ? { predicate: condition.predicate } : {}),
          ...(condition.on_error ? { on_error: condition.on_error } : {}),
        }
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
        rules: decisionRulesForSave(decision?.rules, {
          operator: formData.operator,
          conditions,
        }),
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
      await saveScopedConfig(newConfig)
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
    if (!scopedConfig) {
      setDecisionDeleteError('Recipe configuration is not available.')
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
      await saveScopedConfig(newConfig)
      setDecisionPendingDelete(null)
    } catch (error) {
      setDecisionDeleteError(error instanceof Error ? error.message : 'Failed to delete decision.')
    } finally {
      setDecisionDeletePending(false)
    }
  }

  if (routingScopeLoading || routingScopeError || !selectedRecipe) {
    return (
      <ConfigPageManagerLayout
        title="Decisions"
        description="Turn request signals into clear routing intent."
      >
        <ConfigPageRoutingScopeState
          error={routingScopeError}
          loading={routingScopeLoading}
          onRetry={() => void reloadRoutingScopes()}
        />
      </ConfigPageManagerLayout>
    )
  }

  return (
    <ConfigPageManagerLayout
      title="Decisions"
      description="Turn request signals into clear routing intent."
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
            disabled={scopeReadonly}
            variant="embedded"
          />
          <DataTable
            columns={decisionColumns}
            data={filteredDecisions}
            keyExtractor={(row) => row.name}
            onView={handleViewDecision}
            openOnRowClick
            emptyMessage={
              decisionsSearch ? 'No decisions match your search' : 'No routing decisions configured'
            }
            className={styles.managerTable}
            readonly={scopeReadonly}
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
