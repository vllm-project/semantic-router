import { useMemo, useState, type Dispatch, type SetStateAction } from 'react'
import styles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageModelInventoryPanel from './ConfigPageModelInventoryPanel'
import ModelDeleteDialog from './ModelDeleteDialog'
import ConfirmDialog from '../components/ConfirmDialog'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import { normalizeStringList } from '../components/structuredFieldEditorSupport'
import type { ViewSection } from '../components/ViewModal'
import {
  ConfigData,
  NormalizedModel,
  ReasoningFamily,
} from './configPageSupport'
import {
  ensureProviderDefaultsConfig,
  ensureProvidersConfig,
  cloneConfigData,
  removeRoutingModelCard,
  upsertRoutingModelCard,
} from './configPageCanonicalization'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import {
  filterModelInventory,
  getModelDeleteBlocker,
  getModelReferenceCounts,
  getReasoningFamilyFilterOptions,
  validateModelStructuredFields,
  validateNewModelName,
  type ModelEndpointFilter,
  type ModelRoleFilter,
} from './configPageModelInventory'
import {
  buildProviderModelPayload,
  normalizeModelBackendRefs,
  normalizeModelLoras,
  normalizeModelPricing,
  normalizeModelStringMap,
} from './configPageModelFormSupport'
import { getModelStructuredFormFields } from './configPageModelFormFields'
import {
  ModelBackendRefsEditor,
  ModelCapabilitiesEditor,
  ModelExternalIdsEditor,
  ModelLorasEditor,
  ModelPricingEditor,
  ModelTagsEditor,
} from './configPageModelStructuredEditors'

interface ConfigPageModelsSectionProps {
  config: ConfigData | null
  isPythonCLI: boolean
  isReadonly: boolean
  models: NormalizedModel[]
  defaultModel: string
  reasoningFamilies: Record<string, ReasoningFamily>
  modelsSearch: string
  onModelsSearchChange: (value: string) => void
  expandedModels: Set<string>
  onExpandedModelsChange: Dispatch<SetStateAction<Set<string>>>
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
  listInputToArray: (input: string) => string[]
}

interface ReasoningFamilyFormState {
  name: string
  type: string
  parameter: string
}

export default function ConfigPageModelsSection({
  config,
  isPythonCLI,
  isReadonly,
  models,
  defaultModel,
  reasoningFamilies,
  modelsSearch,
  onModelsSearchChange,
  expandedModels,
  onExpandedModelsChange,
  saveConfig,
  openEditModal,
  openViewModal,
}: ConfigPageModelsSectionProps) {
  const [reasoningFamilyFilter, setReasoningFamilyFilter] = useState('all')
  const [endpointFilter, setEndpointFilter] = useState<ModelEndpointFilter>('all')
  const [roleFilter, setRoleFilter] = useState<ModelRoleFilter>('all')
  const [reasoningFamilySearch, setReasoningFamilySearch] = useState('')
  const [selectedModelKeys, setSelectedModelKeys] = useState<Set<string>>(new Set())
  const [bulkDeletePending, setBulkDeletePending] = useState(false)
  const [operationError, setOperationError] = useState<string | null>(null)
  const [modelsPendingDelete, setModelsPendingDelete] = useState<string[]>([])
  const [reasoningFamilyPendingDelete, setReasoningFamilyPendingDelete] = useState<string | null>(null)
  const [reasoningFamilyDeletePending, setReasoningFamilyDeletePending] = useState(false)
  const [reasoningFamilyDeleteError, setReasoningFamilyDeleteError] = useState<string | null>(null)

  const reasoningFamilyOptions = useMemo(() => getReasoningFamilyFilterOptions(models), [models])
  const modelReferenceCounts = useMemo(() => getModelReferenceCounts(config), [config])
  const filteredModels = useMemo(() => filterModelInventory(models, {
    search: modelsSearch,
    reasoningFamily: reasoningFamilyFilter,
    endpointState: endpointFilter,
    role: roleFilter,
    defaultModel,
  }), [defaultModel, endpointFilter, models, modelsSearch, reasoningFamilyFilter, roleFilter])
  const filtersActive = Boolean(
    modelsSearch.trim()
    || reasoningFamilyFilter !== 'all'
    || endpointFilter !== 'all'
    || roleFilter !== 'all',
  )

  const getDeleteBlocker = (modelName: string) => getModelDeleteBlocker(
    modelName,
    defaultModel,
    modelReferenceCounts,
  )

  const clearModelFilters = () => {
    onModelsSearchChange('')
    setReasoningFamilyFilter('all')
    setEndpointFilter('all')
    setRoleFilter('all')
  }

  type ModelRow = NormalizedModel
  const renderModelEndpoints = (model: ModelRow) => {
    if (!model.endpoints || model.endpoints.length === 0) {
      return (
        <div style={{ padding: '1rem', color: 'var(--color-text-secondary)', textAlign: 'center' }}>
          No endpoints configured for this model
        </div>
      )
    }

    return (
      <div style={{ padding: '1rem', background: 'rgba(0, 0, 0, 0.3)' }}>
        <h4 style={{
          margin: '0 0 1rem 0',
          fontSize: '0.875rem',
          fontWeight: 600,
          color: 'var(--color-text-secondary)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em'
        }}>
          Endpoints for {model.name}
        </h4>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
              <th style={{ padding: '0.5rem', textAlign: 'left', fontSize: '0.875rem', fontWeight: 600, color: 'var(--color-text-secondary)' }}>Name</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', fontSize: '0.875rem', fontWeight: 600, color: 'var(--color-text-secondary)' }}>Address</th>
              <th style={{ padding: '0.5rem', textAlign: 'center', fontSize: '0.875rem', fontWeight: 600, color: 'var(--color-text-secondary)', width: '100px' }}>Protocol</th>
              <th style={{ padding: '0.5rem', textAlign: 'center', fontSize: '0.875rem', fontWeight: 600, color: 'var(--color-text-secondary)', width: '100px' }}>Weight</th>
            </tr>
          </thead>
          <tbody>
            {model.endpoints.map((ep, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                <td style={{ padding: '0.75rem 0.5rem', fontSize: '0.875rem', fontWeight: 500 }}>{ep.name}</td>
                <td style={{ padding: '0.75rem 0.5rem', fontSize: '0.875rem', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
                  {isReadonly ? '************' : (ep.endpoint || 'N/A')}
                </td>
                <td style={{ padding: '0.75rem 0.5rem', textAlign: 'center' }}>
                  <span style={{
                    padding: '0.25rem 0.5rem',
                    background: ep.protocol === 'https' ? 'rgba(34, 197, 94, 0.15)' : 'rgba(234, 179, 8, 0.15)',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    textTransform: 'uppercase'
                  }}>
                    {ep.protocol}
                  </span>
                </td>
                <td style={{ padding: '0.75rem 0.5rem', textAlign: 'center', fontSize: '0.875rem', fontFamily: 'var(--font-mono)' }}>
                  {ep.weight}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  const handleViewModel = (model: ModelRow) => {
    const sections: ViewSection[] = [
      {
        title: 'Basic Information',
        fields: [
          { label: 'Model Name', value: model.name },
          { label: 'Reasoning Family', value: model.reasoning_family || 'N/A' },
          { label: 'Is Default', value: model.name === defaultModel ? 'Yes' : 'No' },
          { label: 'Provider Model ID', value: model.provider_model_id || 'N/A' },
          { label: 'API Format', value: model.api_format || 'N/A' },
          { label: 'Modality', value: model.modality || 'N/A' },
          { label: 'Param Size', value: model.param_size || 'N/A' },
          { label: 'Context Window', value: model.context_window_size ? `${model.context_window_size}` : 'N/A' },
        ]
      }
    ]

    if (model.description || model.capabilities?.length || model.tags?.length || model.loras?.length || typeof model.quality_score === 'number') {
      sections.push({
        title: 'Routing Metadata',
        fields: [
          { label: 'Description', value: model.description || 'N/A', fullWidth: true },
          {
            label: 'Capabilities',
            value: <ModelCapabilitiesEditor value={model.capabilities || []} readOnly />,
            fullWidth: true,
          },
          {
            label: 'Tags',
            value: <ModelTagsEditor value={model.tags || []} readOnly />,
            fullWidth: true,
          },
          {
            label: 'LoRAs',
            value: <ModelLorasEditor value={model.loras || []} readOnly />,
            fullWidth: true,
          },
          { label: 'Quality Score', value: typeof model.quality_score === 'number' ? `${model.quality_score}` : 'N/A' },
        ]
      })
    }

    if (model.external_model_ids && Object.keys(model.external_model_ids).length > 0) {
      sections.push({
        title: 'External Model IDs',
        fields: [
          {
            label: 'Provider IDs',
            value: <ModelExternalIdsEditor value={model.external_model_ids} readOnly />,
            fullWidth: true,
          },
        ],
      })
    }

    if (model.backend_refs && model.backend_refs.length > 0) {
      sections.push({
        title: `Provider Backends (${model.backend_refs.length})`,
        fields: [
          {
            label: 'Configured Backend Refs',
            value: (
              <ModelBackendRefsEditor
                value={model.backend_refs}
                readOnly
                maskSensitive={isReadonly}
              />
            ),
            fullWidth: true
          }
        ]
      })
    }

    if (model.pricing) {
      sections.push({
        title: 'Pricing',
        fields: [
          {
            label: 'Token Pricing',
            value: <ModelPricingEditor value={model.pricing} readOnly />,
            fullWidth: true,
          },
        ]
      })
    }

    openViewModal(`Model: ${model.name}`, sections, () => handleEditModel(model))
  }

  const handleAddModel = () => {
    const reasoningFamilyNames = Object.keys(reasoningFamilies)

    openEditModal(
      'Add New Model',
      {
        model_name: '',
        reasoning_family: reasoningFamilyNames[0] || '',
        provider_model_id: '',
        api_format: '',
        external_model_ids: {},
        param_size: '',
        context_window_size: '',
        description: '',
        capabilities: [],
        loras: [],
        tags: [],
        quality_score: '',
        modality: '',
        backend_refs: [{
          name: 'endpoint-1',
          endpoint: 'localhost:8000',
          protocol: 'http' as const,
          weight: 1,
        }],
        pricing: {
          currency: 'USD',
          prompt_per_1m: 0,
          cached_input_per_1m: 0,
          completion_per_1m: 0,
        },
      },
      [
        {
          name: 'model_name',
          label: 'Model Name',
          type: 'text',
          required: true,
          placeholder: 'e.g., openai/gpt-4',
          description: 'Unique identifier for the model'
        },
        {
          name: 'reasoning_family',
          label: 'Reasoning Family',
          type: 'select',
          options: reasoningFamilyNames,
          description: 'Select from configured reasoning families'
        },
        {
          name: 'provider_model_id',
          label: 'Provider Model ID',
          type: 'text',
          placeholder: 'e.g., openai/gpt-4.1',
          description: 'Concrete upstream model identifier stored under providers.models[].provider_model_id'
        },
        {
          name: 'api_format',
          label: 'API Format',
          type: 'text',
          placeholder: 'e.g., openai',
          description: 'Provider-specific wire format stored under providers.models[].api_format'
        },
        {
          name: 'param_size',
          label: 'Parameter Size',
          type: 'text',
          placeholder: 'e.g., 8B'
        },
        {
          name: 'context_window_size',
          label: 'Context Window Size',
          type: 'number',
          placeholder: 'e.g., 131072'
        },
        {
          name: 'modality',
          label: 'Modality',
          type: 'text',
          placeholder: 'e.g., text, omni, diffusion'
        },
        {
          name: 'description',
          label: 'Description',
          type: 'textarea',
          placeholder: 'Short routing-facing model description'
        },
        ...getModelStructuredFormFields(),
      ],
      async (data) => {
        if (!config) {
          return
        }
        validateModelStructuredFields(data)
        const modelName = validateNewModelName(data.model_name, models)
        const capabilities = normalizeStringList(data.capabilities)
        const tags = normalizeStringList(data.tags)
        const loras = normalizeModelLoras(data.loras)

        const newConfig = cloneConfigData(config)

        if (isPythonCLI) {
          const providers = ensureProvidersConfig(newConfig)
          upsertRoutingModelCard(newConfig, modelName, {
            param_size: data.param_size || undefined,
            context_window_size: data.context_window_size ? Number(data.context_window_size) : undefined,
            description: data.description || undefined,
            capabilities: capabilities.length > 0 ? capabilities : undefined,
            loras: loras.length > 0 ? loras : undefined,
            tags: tags.length > 0 ? tags : undefined,
            quality_score: data.quality_score === '' || data.quality_score === undefined ? undefined : Number(data.quality_score),
            modality: data.modality || undefined,
          })
          providers.models.push(buildProviderModelPayload(modelName, data))
        } else {
          if (!newConfig.model_config) {
            newConfig.model_config = {}
          }
          newConfig.model_config[modelName] = {
            reasoning_family: data.reasoning_family,
            pricing: normalizeModelPricing(data.pricing),
            api_format: typeof data.api_format === 'string' ? data.api_format : undefined,
            external_model_ids: normalizeModelStringMap(data.external_model_ids),
            preferred_endpoints: normalizeModelBackendRefs(data.backend_refs).map((backendRef) => backendRef.name || '').filter(Boolean),
            model_id: typeof data.provider_model_id === 'string' && data.provider_model_id.trim()
              ? data.provider_model_id.trim()
              : modelName,
          }
        }
        await saveConfig(newConfig)
      },
      'add'
    )
  }

  const handleEditModel = (model: ModelRow) => {
    const reasoningFamilyNames = Object.keys(reasoningFamilies)

    openEditModal(
      `Edit Model: ${model.name}`,
      {
        reasoning_family: model.reasoning_family || '',
        provider_model_id: model.provider_model_id || '',
        api_format: model.api_format || '',
        external_model_ids: model.external_model_ids || {},
        param_size: model.param_size || '',
        context_window_size: model.context_window_size || '',
        description: model.description || '',
        capabilities: model.capabilities || [],
        loras: model.loras || [],
        tags: model.tags || [],
        quality_score: model.quality_score ?? '',
        modality: model.modality || '',
        backend_refs: model.backend_refs || [],
        pricing: model.pricing || {}
      },
      [
        {
          name: 'reasoning_family',
          label: 'Reasoning Family',
          type: 'select',
          options: reasoningFamilyNames,
          description: 'Select from configured reasoning families'
        },
        {
          name: 'provider_model_id',
          label: 'Provider Model ID',
          type: 'text',
          placeholder: 'e.g., openai/gpt-4.1',
          description: 'Concrete upstream model identifier stored under providers.models[].provider_model_id'
        },
        {
          name: 'api_format',
          label: 'API Format',
          type: 'text',
          placeholder: 'e.g., openai',
          description: 'Provider-specific wire format stored under providers.models[].api_format'
        },
        {
          name: 'param_size',
          label: 'Parameter Size',
          type: 'text',
          placeholder: 'e.g., 8B'
        },
        {
          name: 'context_window_size',
          label: 'Context Window Size',
          type: 'number',
          placeholder: 'e.g., 131072'
        },
        {
          name: 'modality',
          label: 'Modality',
          type: 'text',
          placeholder: 'e.g., text, omni, diffusion'
        },
        {
          name: 'description',
          label: 'Description',
          type: 'textarea',
          placeholder: 'Short routing-facing model description'
        },
        ...getModelStructuredFormFields(),
      ],
      async (data) => {
        if (!config) {
          return
        }
        validateModelStructuredFields(data)
        const newConfig = cloneConfigData(config)
        const capabilities = normalizeStringList(data.capabilities)
        const tags = normalizeStringList(data.tags)
        const loras = normalizeModelLoras(data.loras)

        if (isPythonCLI && newConfig.providers?.models) {
          const providers = ensureProvidersConfig(newConfig)
          upsertRoutingModelCard(newConfig, model.name, {
            param_size: data.param_size || undefined,
            context_window_size: data.context_window_size ? Number(data.context_window_size) : undefined,
            description: data.description || undefined,
            capabilities: capabilities.length > 0 ? capabilities : undefined,
            loras: loras.length > 0 ? loras : undefined,
            tags: tags.length > 0 ? tags : undefined,
            quality_score: data.quality_score === '' || data.quality_score === undefined ? undefined : Number(data.quality_score),
            modality: data.modality || undefined,
          })
          type ProviderModel = NonNullable<ConfigData['providers']>['models'][number]
          providers.models = providers.models.map((providerModel: ProviderModel) =>
            providerModel.name === model.name ? {
              ...providerModel,
              ...buildProviderModelPayload(model.name, data, providerModel),
            } : providerModel
          )
        } else if (newConfig.model_config) {
          newConfig.model_config[model.name] = {
            ...newConfig.model_config[model.name],
            reasoning_family: data.reasoning_family,
            pricing: normalizeModelPricing(data.pricing),
            api_format: typeof data.api_format === 'string' ? data.api_format : undefined,
            external_model_ids: normalizeModelStringMap(data.external_model_ids),
            preferred_endpoints: normalizeModelBackendRefs(data.backend_refs).map((backendRef) => backendRef.name || '').filter(Boolean),
            model_id: typeof data.provider_model_id === 'string' ? data.provider_model_id : model.name,
          }
        }
        await saveConfig(newConfig)
      },
      'edit'
    )
  }

  const handleDeleteModelsAction = async (modelNames: string[]) => {
    if (!config || modelNames.length === 0) {
      return
    }
    const blockedModel = modelNames.find((modelName) => getDeleteBlocker(modelName))
    if (blockedModel) {
      setOperationError(getDeleteBlocker(blockedModel))
      setModelsPendingDelete([])
      return
    }

    setBulkDeletePending(true)
    setOperationError(null)
    try {
      const namesToDelete = new Set(modelNames)
      const newConfig = cloneConfigData(config)
      if (isPythonCLI && newConfig.providers?.models) {
        const providers = ensureProvidersConfig(newConfig)
        type ProviderModel = NonNullable<ConfigData['providers']>['models'][number]
        providers.models = providers.models.filter((providerModel: ProviderModel) => !namesToDelete.has(providerModel.name))
        for (const modelName of namesToDelete) {
          removeRoutingModelCard(newConfig, modelName)
        }
      } else if (newConfig.model_config) {
        for (const modelName of namesToDelete) {
          delete newConfig.model_config[modelName]
        }
      }
      await saveConfig(newConfig)
      setSelectedModelKeys((current) => {
        const next = new Set(current)
        for (const modelName of namesToDelete) next.delete(modelName)
        return next
      })
      setModelsPendingDelete([])
    } catch (error) {
      setOperationError(error instanceof Error ? error.message : 'Failed to delete the selected models.')
    } finally {
      setBulkDeletePending(false)
    }
  }

  const handleDeleteModel = (model: ModelRow) => {
    const blocker = getDeleteBlocker(model.name)
    if (blocker) {
      setOperationError(`${model.name}: ${blocker}`)
      return
    }
    setOperationError(null)
    setModelsPendingDelete([model.name])
  }

  const handleToggleExpand = (model: ModelRow) => {
    onExpandedModelsChange(prev => {
      const next = new Set(prev)
      if (next.has(model.name)) {
        next.delete(model.name)
      } else {
        next.add(model.name)
      }
      return next
    })
  }

  const handleViewReasoningFamily = (familyName: string) => {
    const familyConfig = reasoningFamilies[familyName]
    if (!familyConfig) return

    openViewModal(
      `Reasoning Family: ${familyName}`,
      [
        {
          title: 'Configuration',
          fields: [
            { label: 'Family Name', value: familyName },
            { label: 'Type', value: familyConfig.type },
            { label: 'Parameter', value: familyConfig.parameter }
          ]
        }
      ],
      () => handleEditReasoningFamily(familyName),
    )
  }

  const handleEditReasoningFamily = (familyName: string) => {
    const familyConfig = reasoningFamilies[familyName]
    if (!familyConfig || !config) return

    openEditModal(
      `Edit Reasoning Family: ${familyName}`,
      { ...familyConfig },
      [
        {
          name: 'type',
          label: 'Type',
          type: 'select',
          options: ['reasoning_effort', 'chat_template_kwargs'],
          required: true,
          description: 'Type of reasoning family'
        },
        {
          name: 'parameter',
          label: 'Parameter',
          type: 'text',
          required: true,
          placeholder: 'e.g., reasoning_effort',
          description: 'Parameter name for reasoning control'
        }
      ],
      async (data) => {
        const newConfig = cloneConfigData(config)
        if (isPythonCLI) {
          const defaults = ensureProviderDefaultsConfig(newConfig)
          if (!defaults.reasoning_families) {
            defaults.reasoning_families = {}
          }
          defaults.reasoning_families[familyName] = data
        } else if (newConfig.reasoning_families) {
          newConfig.reasoning_families[familyName] = data
        }
        await saveConfig(newConfig)
      }
    )
  }

  const handleAddReasoningFamily = () => {
    if (!config) return

    openEditModal<ReasoningFamilyFormState>(
      'Add Reasoning Family',
      { name: '', type: 'reasoning_effort', parameter: '' },
      [
        {
          name: 'name',
          label: 'Family Name',
          type: 'text',
          required: true,
          placeholder: 'e.g., o1-reasoning',
          description: 'Unique name for this reasoning family'
        },
        {
          name: 'type',
          label: 'Type',
          type: 'select',
          options: ['reasoning_effort', 'chat_template_kwargs'],
          required: true,
          description: 'Type of reasoning family'
        },
        {
          name: 'parameter',
          label: 'Parameter',
          type: 'text',
          required: true,
          placeholder: 'e.g., reasoning_effort',
          description: 'Parameter name for reasoning control'
        }
      ],
      async (data) => {
        const familyName = data.name
        const familyConfig = {
          type: data.type,
          parameter: data.parameter,
        }

        const newConfig = cloneConfigData(config)
        if (isPythonCLI) {
          const defaults = ensureProviderDefaultsConfig(newConfig)
          if (!defaults.reasoning_families) {
            defaults.reasoning_families = {}
          }
          defaults.reasoning_families[familyName] = familyConfig
        } else {
          if (!newConfig.reasoning_families) {
            newConfig.reasoning_families = {}
          }
          newConfig.reasoning_families[familyName] = familyConfig
        }
        await saveConfig(newConfig)
      },
      'add'
    )
  }

  const handleDeleteReasoningFamily = (familyName: string) => {
    setReasoningFamilyDeleteError(null)
    setReasoningFamilyPendingDelete(familyName)
  }

  const confirmDeleteReasoningFamily = async () => {
    if (!reasoningFamilyPendingDelete) return
    if (!config) {
      setReasoningFamilyDeleteError('No active configuration is available.')
      return
    }

    setReasoningFamilyDeletePending(true)
    setReasoningFamilyDeleteError(null)
    try {
      const newConfig = cloneConfigData(config)
      if (isPythonCLI && newConfig.providers?.defaults?.reasoning_families) {
        const defaults = ensureProviderDefaultsConfig(newConfig)
        defaults.reasoning_families = { ...defaults.reasoning_families }
        delete defaults.reasoning_families[reasoningFamilyPendingDelete]
      } else if (newConfig.reasoning_families) {
        delete newConfig.reasoning_families[reasoningFamilyPendingDelete]
      }
      await saveConfig(newConfig)
      setReasoningFamilyPendingDelete(null)
    } catch (error) {
      setReasoningFamilyDeleteError(
        error instanceof Error ? error.message : 'Failed to delete reasoning family.',
      )
    } finally {
      setReasoningFamilyDeletePending(false)
    }
  }

  type ReasoningFamilyRow = { name: string; type: string; parameter: string }
  const reasoningFamilyData: ReasoningFamilyRow[] = Object.entries(reasoningFamilies).map(([name, config]) => ({
    name,
    type: config.type,
    parameter: config.parameter
  }))
  const normalizedReasoningFamilySearch = reasoningFamilySearch.trim().toLocaleLowerCase()
  const filteredReasoningFamilyData = normalizedReasoningFamilySearch
    ? reasoningFamilyData.filter((family) => (
        family.name.toLocaleLowerCase().includes(normalizedReasoningFamilySearch)
        || family.type.toLocaleLowerCase().includes(normalizedReasoningFamilySearch)
        || family.parameter.toLocaleLowerCase().includes(normalizedReasoningFamilySearch)
      ))
    : reasoningFamilyData

  const reasoningFamilyColumns: Column<ReasoningFamilyRow>[] = [
    {
      key: 'name',
      header: 'Family Name',
      sortable: true,
      render: (row) => (
        <span style={{ fontWeight: 600 }}>{row.name}</span>
      )
    },
    {
      key: 'type',
      header: 'Type',
      width: '200px',
      sortable: true,
      render: (row) => (
        <span className={styles.badge} style={{ background: 'rgba(166, 171, 179, 0.15)', color: 'var(--color-accent-cyan)' }}>
          {row.type}
        </span>
      )
    },
    {
      key: 'parameter',
      header: 'Parameter',
      sortable: true,
      render: (row) => (
        <code style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>{row.parameter}</code>
      )
    }
  ]

  return (
    <>
      <ConfigPageManagerLayout title="Models" description="Manage provider models, reasoning families, and the endpoint inventory available to routing decisions.">
        <div className={styles.sectionPanel}>
          <div className={styles.sectionTableBlock}>
            <ConfigPageModelInventoryPanel
              models={models}
              filteredModels={filteredModels}
              defaultModel={defaultModel}
              modelReferenceCounts={modelReferenceCounts}
              modelsSearch={modelsSearch}
              onModelsSearchChange={onModelsSearchChange}
              reasoningFamilyFilter={reasoningFamilyFilter}
              onReasoningFamilyFilterChange={setReasoningFamilyFilter}
              reasoningFamilyOptions={reasoningFamilyOptions}
              endpointFilter={endpointFilter}
              onEndpointFilterChange={setEndpointFilter}
              roleFilter={roleFilter}
              onRoleFilterChange={setRoleFilter}
              filtersActive={filtersActive}
              onClearFilters={clearModelFilters}
              isReadonly={isReadonly}
              selectedModelKeys={selectedModelKeys}
              onSelectedModelKeysChange={setSelectedModelKeys}
              onClearSelection={() => setSelectedModelKeys(new Set())}
              onDeleteSelected={() => {
                setOperationError(null)
                setModelsPendingDelete([...selectedModelKeys])
              }}
              operationError={operationError}
              onDismissOperationError={() => setOperationError(null)}
              onAddModel={handleAddModel}
              onViewModel={handleViewModel}
              onEditModel={handleEditModel}
              onDeleteModel={handleDeleteModel}
              expandedModels={expandedModels}
              onToggleExpand={handleToggleExpand}
              renderExpandedRow={renderModelEndpoints}
              getDeleteBlocker={getDeleteBlocker}
            />
          </div>

          <div className={styles.sectionTableBlock}>
            <TableHeader
              title="Reasoning Families"
              count={filteredReasoningFamilyData.length}
              searchPlaceholder="Search family, type, or parameter..."
              searchValue={reasoningFamilySearch}
              onSearchChange={setReasoningFamilySearch}
              onAdd={handleAddReasoningFamily}
              addButtonText="Add Family"
              disabled={isReadonly}
              variant="embedded"
            />
            <DataTable
              columns={reasoningFamilyColumns}
              data={filteredReasoningFamilyData}
              keyExtractor={(row) => row.name}
              onView={(row) => handleViewReasoningFamily(row.name)}
              onEdit={(row) => handleEditReasoningFamily(row.name)}
              onDelete={(row) => handleDeleteReasoningFamily(row.name)}
              emptyMessage="No reasoning families configured"
              className={styles.managerTable}
              readonly={isReadonly}
              pagination={{
                pageSize: 25,
                pageSizeOptions: [25, 50, 100],
                itemLabel: 'families',
                resetKey: reasoningFamilySearch,
              }}
            />
          </div>
        </div>
      </ConfigPageManagerLayout>

      <ModelDeleteDialog
        modelNames={modelsPendingDelete}
        pending={bulkDeletePending}
        onCancel={() => setModelsPendingDelete([])}
        onConfirm={() => void handleDeleteModelsAction(modelsPendingDelete)}
      />

      <ConfirmDialog
        isOpen={reasoningFamilyPendingDelete !== null}
        title={`Delete reasoning family “${reasoningFamilyPendingDelete || ''}”?`}
        description="Remove this reasoning control definition from the active model configuration. Models that still reference it may need to be updated separately."
        eyebrow="Destructive configuration change"
        confirmLabel="Delete family"
        pending={reasoningFamilyDeletePending}
        details={reasoningFamilyDeleteError ? (
          <span role="alert">{reasoningFamilyDeleteError}</span>
        ) : undefined}
        onCancel={() => {
          if (reasoningFamilyDeletePending) return
          setReasoningFamilyPendingDelete(null)
          setReasoningFamilyDeleteError(null)
        }}
        onConfirm={confirmDeleteReasoningFamily}
      />
    </>
  )
}
