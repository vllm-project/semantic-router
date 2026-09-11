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
import { getAlgorithmFieldSchema, getPluginFieldSchema } from '../lib/dslSchemas'
import type {
  ConfigData,
  DecisionConfig,
  DecisionFormState,
  NormalizedModel,
} from './configPageSupport'
import { getReasoningFamiliesMap, mergeDecisionForSave } from './configPageSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { cloneConfigData } from './configPageCanonicalization'
import ConfigPageDecisionPluginsEditor from './ConfigPageDecisionPluginsEditor'
import ConfigPageDecisionModelRefsEditor from './ConfigPageDecisionModelRefsEditor'
import ConfigPageDecisionAlgorithmEditor from './ConfigPageDecisionAlgorithmEditor'
import ConfigPageDecisionRulesEditor from './ConfigPageDecisionRulesEditor'
import { algorithmFields, algorithmType } from './configPageDecisionAlgorithmSupport'
import ConfigPageSchemaFieldsEditor from './ConfigPageSchemaFieldsEditor'
import { requiredSchemaFieldErrors } from './configPageSchemaValidation'
import {
  DECISION_ACTION_SCHEMA,
  DECISION_ADAPTATIONS_SCHEMA,
  DECISION_DECLARATIVE_SCHEMA,
  DECISION_OUTPUT_CONTRACT_SCHEMA,
} from './configPageDecisionAdvancedSchemas'
import {
  decisionModelRefsForForm,
  decisionModelRefsForSave,
  decisionPluginsForSave,
  decisionRulesForSave,
} from './configPageDecisionFormSupport'
import { useRoutingScopeManager } from './configPageRoutingScopeSupport'
import { decisionColumns } from './configPageDecisionTable'
import {
  reasoningFamilyForModel,
  reasoningFamilyIsAlwaysOn,
} from './configPageReasoningControlSupport'

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

function configuredObject(value: Record<string, unknown>): Record<string, unknown> | undefined {
  return Object.keys(value).length > 0 ? value : undefined
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
  const [decisionPendingDelete, setDecisionPendingDelete] = useState<DecisionConfig | null>(null)
  const [decisionDeletePending, setDecisionDeletePending] = useState(false)
  const [decisionDeleteError, setDecisionDeleteError] = useState<string | null>(null)
  const { applyScopedConfig, routingScopes, scopedConfig, selectedScopeId, setSelectedScopeId } =
    useRoutingScopeManager(config)
  const reasoningFamilies = getReasoningFamiliesMap(config, isPythonCLI)
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
      ref.reasoning_mode ? `Mode: ${ref.reasoning_mode}` : null,
      ref.reasoning_effort ? `Effort: ${ref.reasoning_effort}` : null,
      ref.lora_name ? `LoRA: ${ref.lora_name}` : null,
      typeof ref.weight === 'number' ? `Weight: ${ref.weight}` : null,
      typeof ref.max_completion_tokens === 'number'
        ? `Max completion tokens: ${ref.max_completion_tokens}`
        : null,
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
        title: 'Identity',
        fields: [
          { label: 'Name', value: decision.name },
          { label: 'Priority', value: `P${decision.priority}` },
          { label: 'Tier', value: decision.tier ?? 'Not set' },
          { label: 'Description', value: decision.description || 'N/A', fullWidth: true },
        ],
      },
      {
        title: 'Routing policy',
        fields: [
          {
            label: 'Rule Tree',
            value: <ConfigPageDecisionRulesEditor value={decision.rules || {}} readOnly />,
            fullWidth: true,
          },
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
          {
            label: 'Direct Action',
            value: (
              <ConfigPageSchemaFieldsEditor
                schema={DECISION_ACTION_SCHEMA}
                value={decision.action || {}}
                readOnly
              />
            ),
            fullWidth: true,
          },
        ],
      },
    ]

    sections.push({
      title: 'Selection & runtime',
      fields: [
        {
          label: 'Algorithm',
          value: <ConfigPageDecisionAlgorithmEditor value={decision.algorithm} readOnly />,
          fullWidth: true,
        },
        {
          label: 'Adaptation & Protection',
          value: (
            <ConfigPageSchemaFieldsEditor
              schema={DECISION_ADAPTATIONS_SCHEMA}
              value={decision.adaptations || {}}
              readOnly
            />
          ),
          fullWidth: true,
        },
        ...(decision.plugins?.length
          ? [
              {
                label: 'Plugins',
                value: (
                  <div className={decisionStyles.viewStack}>
                    {decision.plugins.map((plugin, i) => (
                      <article key={`${plugin.type}-${i}`} className={decisionStyles.viewCard}>
                        <div className={decisionStyles.viewHeading}>
                          <span className={decisionStyles.viewTitle}>{plugin.type}</span>
                        </div>
                        <ConfigPageSchemaFieldsEditor
                          schema={getPluginFieldSchema(plugin.type)}
                          value={plugin.configuration || {}}
                          readOnly
                        />
                      </article>
                    ))}
                  </div>
                ),
                fullWidth: true,
              },
            ]
          : []),
      ],
    })

    sections.push({
      title: 'Output behavior',
      fields: [
        { label: 'Output Contract', value: decision.output_contract || 'Not set', fullWidth: true },
        {
          label: 'Output Contract Spec',
          value: (
            <ConfigPageSchemaFieldsEditor
              schema={DECISION_OUTPUT_CONTRACT_SCHEMA}
              value={decision.output_contract_spec || {}}
              readOnly
            />
          ),
          fullWidth: true,
        },
        {
          label: 'Iterations, Emits & Annotations',
          value: (
            <ConfigPageSchemaFieldsEditor
              schema={DECISION_DECLARATIVE_SCHEMA}
              value={{
                candidateIterations: decision.candidateIterations || [],
                emits: decision.emits || [],
                annotations: decision.annotations || {},
              }}
              readOnly
            />
          ),
          fullWidth: true,
        },
      ],
    })

    openViewModal(`Decision: ${decision.name}`, sections, () => handleEditDecision(decision))
  }

  const openDecisionEditor = (mode: 'add' | 'edit', decision?: DecisionRow) => {
    const defaultForm: DecisionFormState = {
      name: '',
      description: '',
      priority: 1,
      tier: undefined,
      output_contract: '',
      output_contract_spec: {},
      action: {},
      algorithm: undefined,
      adaptations: {},
      declarative: {},
      rules: {},
      modelRefs: [
        {
          model: '',
          use_reasoning: false,
          reasoning_description: '',
          reasoning_mode: '',
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
            tier: decision.tier,
            output_contract: decision.output_contract || '',
            output_contract_spec: decision.output_contract_spec || {},
            action: decision.action || {},
            algorithm: decision.algorithm,
            adaptations: decision.adaptations || {},
            declarative: {
              candidateIterations: decision.candidateIterations || [],
              emits: decision.emits || [],
              annotations: decision.annotations || {},
            },
            rules: JSON.parse(JSON.stringify(decision.rules || {})),
            modelRefs: decisionModelRefsForForm(decision.modelRefs).map((ref) => ({
              ...ref,
              use_reasoning:
                ref.use_reasoning ||
                reasoningFamilyIsAlwaysOn(
                  reasoningFamilyForModel(
                    models.find((model) => model.name === ref.model),
                    reasoningFamilies,
                  ),
                ),
            })),
            plugins: (decision.plugins || []).map((plugin) => ({
              type: plugin.type,
              configuration: { ...(plugin.configuration || {}) },
            })),
          }
        : defaultForm

    const fields: FieldConfig<DecisionFormState>[] = [
      {
        name: 'name',
        label: 'Name',
        section: 'Identity',
        fullWidth: true,
        type: 'text',
        required: true,
        placeholder: 'Enter a unique decision name',
      },
      {
        name: 'priority',
        label: 'Priority',
        section: 'Identity',
        type: 'number',
        min: 0,
        placeholder: '1',
      },
      {
        name: 'tier',
        label: 'Tier',
        section: 'Identity',
        type: 'number',
        min: 0,
        description: 'Optional decision tier used by tier-scoped learning and selection.',
      },
      {
        name: 'description',
        label: 'Description',
        section: 'Identity',
        type: 'textarea',
        placeholder: 'What does this decision route?',
      },
      {
        name: 'rules',
        label: 'Rule Tree',
        section: 'Routing policy',
        type: 'custom',
        description:
          'Configure an unconditional route or a recursive AND, OR, and NOT tree with predicates and classifier failure policy.',
        customRender: (value, onChange) => (
          <ConfigPageDecisionRulesEditor
            value={(value as DecisionFormState['rules']) || {}}
            onChange={(nextValue) => onChange(nextValue)}
          />
        ),
      },
      {
        name: 'modelRefs',
        label: 'Model References',
        section: 'Routing policy',
        type: 'custom',
        description: 'Set target models and whether to enable reasoning.',
        customRender: (value, onChange) => (
          <ConfigPageDecisionModelRefsEditor
            value={Array.isArray(value) ? (value as DecisionFormState['modelRefs']) : []}
            onChange={(nextValue) => onChange(nextValue)}
            models={models}
            reasoningFamilies={reasoningFamilies}
          />
        ),
      },
      {
        name: 'action',
        label: 'Direct Action',
        section: 'Routing policy',
        type: 'custom',
        description: 'An explicit route action used instead of candidate ranking.',
        customRender: (value, onChange) => (
          <ConfigPageSchemaFieldsEditor
            schema={DECISION_ACTION_SCHEMA}
            value={(value as Record<string, unknown>) || {}}
            onChange={onChange}
          />
        ),
      },
      {
        name: 'algorithm',
        label: 'Selection Algorithm',
        section: 'Selection & runtime',
        type: 'custom',
        description: 'How this decision selects or combines multiple candidate models.',
        customRender: (value, onChange) => (
          <ConfigPageDecisionAlgorithmEditor value={value} onChange={onChange} />
        ),
      },
      {
        name: 'adaptations',
        label: 'Learning & Protection',
        section: 'Selection & runtime',
        type: 'custom',
        description: 'Decision-level overrides for online adaptation and model-switch protection.',
        customRender: (value, onChange) => (
          <ConfigPageSchemaFieldsEditor
            schema={DECISION_ADAPTATIONS_SCHEMA}
            value={(value as Record<string, unknown>) || {}}
            onChange={onChange}
          />
        ),
      },
      {
        name: 'plugins',
        label: 'Plugins',
        section: 'Selection & runtime',
        type: 'custom',
        description: 'Optional plugins applied to this decision.',
        customRender: (value, onChange) => (
          <ConfigPageDecisionPluginsEditor
            value={Array.isArray(value) ? (value as DecisionFormState['plugins']) : []}
            onChange={(nextValue) => onChange(nextValue)}
          />
        ),
      },
      {
        name: 'output_contract',
        label: 'Output Contract',
        section: 'Output behavior',
        type: 'textarea',
        description: 'Optional model-facing output instructions.',
      },
      {
        name: 'output_contract_spec',
        label: 'Output Contract Specification',
        section: 'Output behavior',
        type: 'custom',
        description: 'Typed extraction, normalization, rendering, and post-processing behavior.',
        customRender: (value, onChange) => (
          <ConfigPageSchemaFieldsEditor
            schema={DECISION_OUTPUT_CONTRACT_SCHEMA}
            value={(value as Record<string, unknown>) || {}}
            onChange={onChange}
          />
        ),
      },
      {
        name: 'declarative',
        label: 'Declarative Extensions',
        section: 'Output behavior',
        type: 'custom',
        description: 'Candidate iteration, emitted directives, and bounded annotations.',
        customRender: (value, onChange) => (
          <ConfigPageSchemaFieldsEditor
            schema={DECISION_DECLARATIVE_SCHEMA}
            value={(value as Record<string, unknown>) || {}}
            onChange={onChange}
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

      const rules = decisionRulesForSave(formData.rules)
      const modelRefs = decisionModelRefsForSave(formData.modelRefs)
      const plugins = decisionPluginsForSave(formData.plugins)
      const action = configuredObject(formData.action)
      const adaptations = configuredObject(formData.adaptations)
      const outputContractSpec = configuredObject(formData.output_contract_spec)
      const declarative = formData.declarative || {}
      if (action) {
        const errors = requiredSchemaFieldErrors(DECISION_ACTION_SCHEMA, action)
        if (errors.length > 0) throw new Error(errors[0])
      }
      if (formData.algorithm) {
        const errors = requiredSchemaFieldErrors(
          getAlgorithmFieldSchema(algorithmType(formData.algorithm)),
          algorithmFields(formData.algorithm),
        )
        if (errors.length > 0) throw new Error(errors[0])
      }
      const declarativeErrors = requiredSchemaFieldErrors(DECISION_DECLARATIVE_SCHEMA, declarative)
      if (declarativeErrors.length > 0) throw new Error(declarativeErrors[0])

      const newDecision = mergeDecisionForSave(mode === 'edit' ? decision : undefined, {
        name,
        description: formData.description,
        priority: priority || 0,
        rules,
        modelRefs,
        plugins,
        tier: Number.isFinite(formData.tier) ? formData.tier : undefined,
        output_contract: formData.output_contract?.trim() || undefined,
        output_contract_spec: outputContractSpec,
        action: action as DecisionConfig['action'],
        algorithm: formData.algorithm,
        adaptations,
        candidateIterations: Array.isArray(declarative.candidateIterations)
          ? (declarative.candidateIterations as Array<Record<string, unknown>>)
          : undefined,
        emits: Array.isArray(declarative.emits)
          ? (declarative.emits as Array<Record<string, unknown>>)
          : undefined,
        annotations:
          declarative.annotations && typeof declarative.annotations === 'object'
            ? (declarative.annotations as Record<string, unknown>)
            : undefined,
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
