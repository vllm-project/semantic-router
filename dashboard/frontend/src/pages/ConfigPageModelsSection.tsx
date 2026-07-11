import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from 'react'
import styles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageModelInventoryPanel from './ConfigPageModelInventoryPanel'
import ModelDeleteDialog from './ModelDeleteDialog'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
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
  normalizeModelPricing,
  normalizeModelStringMap,
} from './configPageModelFormSupport'

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
  listInputToArray,
}: ConfigPageModelsSectionProps) {
  const [reasoningFamilyFilter, setReasoningFamilyFilter] = useState('all')
  const [endpointFilter, setEndpointFilter] = useState<ModelEndpointFilter>('all')
  const [roleFilter, setRoleFilter] = useState<ModelRoleFilter>('all')
  const [selectedModelKeys, setSelectedModelKeys] = useState<Set<string>>(new Set())
  const [bulkDeletePending, setBulkDeletePending] = useState(false)
  const [operationError, setOperationError] = useState<string | null>(null)
  const [modelsPendingDelete, setModelsPendingDelete] = useState<string[]>([])

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

  const normalizePricing = (value: unknown): ModelPricing | undefined => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return undefined
    }
    const pricing = value as Record<string, unknown>
    const normalized: ModelPricing = {}
    if (typeof pricing.currency === 'string' && pricing.currency.trim()) normalized.currency = pricing.currency.trim()
    if (typeof pricing.prompt_per_1m === 'number' && Number.isFinite(pricing.prompt_per_1m)) normalized.prompt_per_1m = pricing.prompt_per_1m
    if (typeof pricing.cached_input_per_1m === 'number' && Number.isFinite(pricing.cached_input_per_1m)) normalized.cached_input_per_1m = pricing.cached_input_per_1m
    if (typeof pricing.cache_write_per_1m === 'number' && Number.isFinite(pricing.cache_write_per_1m)) normalized.cache_write_per_1m = pricing.cache_write_per_1m
    if (typeof pricing.completion_per_1m === 'number' && Number.isFinite(pricing.completion_per_1m)) normalized.completion_per_1m = pricing.completion_per_1m
    return Object.keys(normalized).length > 0 ? normalized : undefined
  }

  const buildProviderModelPayload = (
    name: string,
    data: Record<string, unknown>,
    existingModel?: NonNullable<NonNullable<ConfigData['providers']>['models']>[number],
  ) => ({
    name,
    reasoning_family:
      typeof data.reasoning_family === 'string' && data.reasoning_family.trim()
        ? data.reasoning_family.trim()
        : undefined,
    provider_model_id:
      typeof data.provider_model_id === 'string' && data.provider_model_id.trim()
        ? data.provider_model_id.trim()
        : existingModel?.provider_model_id || name,
    api_format:
      typeof data.api_format === 'string' && data.api_format.trim()
        ? data.api_format.trim()
        : undefined,
    external_model_ids: normalizeStringMap(data.external_model_ids),
    backend_refs: normalizeBackendRefs(data.backend_refs),
    pricing: normalizePricing(data.pricing),
  })

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
          { label: 'Capabilities', value: model.capabilities?.join(', ') || 'N/A', fullWidth: true },
          { label: 'Tags', value: model.tags?.join(', ') || 'N/A', fullWidth: true },
          { label: 'LoRAs', value: model.loras?.map((lora) => lora.name).join(', ') || 'N/A', fullWidth: true },
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
            value: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                {Object.entries(model.external_model_ids).map(([key, value]) => (
                  <div key={key}>{key}: {value}</div>
                ))}
              </div>
            ),
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
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {model.backend_refs.map((backendRef, i) => {
                  const displayAddress = backendRef.endpoint || backendRef.base_url || 'N/A'
                  return (
                    <div key={i} style={{
                      border: '1px solid var(--color-border)',
                      borderRadius: '6px',
                      padding: '0.75rem',
                      background: 'rgba(0, 0, 0, 0.2)'
                    }}>
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '0.5rem'
                      }}>
                        <span style={{
                          fontWeight: 600,
                          fontSize: '0.95rem'
                        }}>
                          {backendRef.name || `backend-${i + 1}`}
                        </span>
                      </div>
                      <div style={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: '1rem',
                        fontSize: '0.875rem',
                        color: 'var(--color-text-secondary)'
                      }}>
                        <span style={{ fontFamily: 'var(--font-mono)' }}>
                          {isReadonly ? '************' : displayAddress}
                        </span>
                        {backendRef.protocol ? <span>Protocol: {backendRef.protocol}</span> : null}
                        {typeof backendRef.weight === 'number' ? <span>Weight: {backendRef.weight}</span> : null}
                        {backendRef.provider ? <span>Provider: {backendRef.provider}</span> : null}
                        {backendRef.type ? <span>Type: {backendRef.type}</span> : null}
                        {backendRef.api_key_env ? <span>API Key Env: {backendRef.api_key_env}</span> : null}
                        {backendRef.api_key ? <span>API Key: ••••••••</span> : null}
                        {backendRef.chat_path ? <span>Chat Path: {backendRef.chat_path}</span> : null}
                        {backendRef.api_version ? <span>API Version: {backendRef.api_version}</span> : null}
                      </div>
                      {backendRef.extra_headers && Object.keys(backendRef.extra_headers).length > 0 ? (
                        <div style={{ marginTop: '0.6rem', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
                          extra_headers: {JSON.stringify(backendRef.extra_headers)}
                        </div>
                      ) : null}
                    </div>
                  )
                })}
              </div>
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
          { label: 'Currency', value: model.pricing.currency || 'USD' },
          { label: 'Prompt (per 1M tokens)', value: model.pricing.prompt_per_1m?.toFixed(2) || '0.00' },
          { label: 'Cached input (per 1M tokens)', value: model.pricing.cached_input_per_1m?.toFixed(2) || '0.00' },
          { label: 'Cache write (per 1M tokens)', value: model.pricing.cache_write_per_1m?.toFixed(2) || 'Prompt rate' },
          { label: 'Completion (per 1M tokens)', value: model.pricing.completion_per_1m?.toFixed(2) || '0.00' }
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
        capabilities: '',
        loras: [],
        tags: '',
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
        {
          name: 'capabilities',
          label: 'Capabilities',
          type: 'textarea',
          placeholder: 'Comma or newline separated capabilities'
        },
        {
          name: 'tags',
          label: 'Tags',
          type: 'textarea',
          placeholder: 'Comma or newline separated tags'
        },
        {
          name: 'quality_score',
          label: 'Quality Score',
          type: 'number',
          min: 0,
          max: 1,
          step: 0.01,
          placeholder: '0.85'
        },
        {
          name: 'loras',
          label: 'LoRAs (JSON)',
          type: 'json',
          placeholder: '[{"name":"adapter","description":"optional"}]'
        },
        {
          name: 'backend_refs',
          label: 'Backend Refs (JSON)',
          type: 'json',
          placeholder: '[{"name":"endpoint-1","endpoint":"localhost:8000","protocol":"http","weight":1}]',
          description: 'Latest provider binding format stored under providers.models[].backend_refs'
        },
        {
          name: 'external_model_ids',
          label: 'External Model IDs (JSON)',
          type: 'json',
          placeholder: '{"openai":"gpt-4.1"}',
          description: 'Optional external provider aliases stored under providers.models[].external_model_ids'
        },
        {
          name: 'pricing',
          label: 'Pricing (JSON)',
          type: 'json',
          placeholder: '{"currency":"USD","prompt_per_1m":0.5,"cached_input_per_1m":0.05,"cache_write_per_1m":0.625,"completion_per_1m":1.5}',
          description: 'Structured pricing block stored under providers.models[].pricing'
        }
      ],
      async (data) => {
        if (!config) {
          return
        }
        validateModelStructuredFields(data)
        const modelName = validateNewModelName(data.model_name, models)
        const capabilities = listInputToArray(data.capabilities || '')
        const tags = listInputToArray(data.tags || '')
        const loras = Array.isArray(data.loras) ? data.loras : []

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
        capabilities: (model.capabilities || []).join('\n'),
        loras: model.loras || [],
        tags: (model.tags || []).join('\n'),
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
        {
          name: 'capabilities',
          label: 'Capabilities',
          type: 'textarea',
          placeholder: 'Comma or newline separated capabilities'
        },
        {
          name: 'tags',
          label: 'Tags',
          type: 'textarea',
          placeholder: 'Comma or newline separated tags'
        },
        {
          name: 'quality_score',
          label: 'Quality Score',
          type: 'number',
          min: 0,
          max: 1,
          step: 0.01,
          placeholder: '0.85'
        },
        {
          name: 'loras',
          label: 'LoRAs (JSON)',
          type: 'json',
          placeholder: '[{"name":"adapter","description":"optional"}]'
        },
        {
          name: 'backend_refs',
          label: 'Backend Refs (JSON)',
          type: 'json',
          placeholder: '[{"name":"endpoint-1","endpoint":"localhost:8000","protocol":"http","weight":1}]',
          description: 'Latest provider binding format stored under providers.models[].backend_refs'
        },
        {
          name: 'external_model_ids',
          label: 'External Model IDs (JSON)',
          type: 'json',
          placeholder: '{"openai":"gpt-4.1"}',
          description: 'Optional external provider aliases stored under providers.models[].external_model_ids'
        },
        {
          name: 'pricing',
          label: 'Pricing (JSON)',
          type: 'json',
          placeholder: '{"currency":"USD","prompt_per_1m":0.5,"cached_input_per_1m":0.05,"cache_write_per_1m":0.625,"completion_per_1m":1.5}',
          description: 'Structured pricing block stored under providers.models[].pricing'
        }
      ],
      async (data) => {
        if (!config) {
          return
        }
        validateModelStructuredFields(data)
        const newConfig = cloneConfigData(config)
        const capabilities = listInputToArray(data.capabilities || '')
        const tags = listInputToArray(data.tags || '')
        const loras = Array.isArray(data.loras) ? data.loras : []

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

  const handleDeleteReasoningFamily = async (familyName: string) => {
    if (!config) return
    if (!confirm(`Are you sure you want to delete reasoning family "${familyName}"?`)) {
      return
    }

    const newConfig = cloneConfigData(config)
    if (isPythonCLI && newConfig.providers?.defaults?.reasoning_families) {
      const defaults = ensureProviderDefaultsConfig(newConfig)
      defaults.reasoning_families = { ...defaults.reasoning_families }
      delete defaults.reasoning_families[familyName]
    } else if (newConfig.reasoning_families) {
      delete newConfig.reasoning_families[familyName]
    }
    await saveConfig(newConfig)
  }

  type ReasoningFamilyRow = { name: string; type: string; parameter: string }
  const reasoningFamilyData: ReasoningFamilyRow[] = Object.entries(reasoningFamilies).map(([name, config]) => ({
    name,
    type: config.type,
    parameter: config.parameter
  }))

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
            <TableHeader title="Reasoning Families" count={reasoningFamilyData.length} onAdd={handleAddReasoningFamily} addButtonText="Add Family" disabled={isReadonly} variant="embedded" />
            <DataTable
              columns={reasoningFamilyColumns}
              data={reasoningFamilyData}
              keyExtractor={(row) => row.name}
              onView={(row) => handleViewReasoningFamily(row.name)}
              onEdit={(row) => handleEditReasoningFamily(row.name)}
              onDelete={(row) => handleDeleteReasoningFamily(row.name)}
              emptyMessage="No reasoning families configured"
              className={styles.managerTable}
              readonly={isReadonly}
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
    </>
  )
}
