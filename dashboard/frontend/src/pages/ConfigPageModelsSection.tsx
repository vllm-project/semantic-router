import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from 'react'
import styles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageAddModelsDialog, { type ModelBatchImportInput } from './ConfigPageAddModelsDialog'
import ConfigPageModelInventoryPanel from './ConfigPageModelInventoryPanel'
import ModelDeleteDialog from './ModelDeleteDialog'
import ConfirmDialog from '../components/ConfirmDialog'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import { normalizeStringList } from '../components/structuredFieldEditorSupport'
import type { ViewSection } from '../components/ViewModal'
import { ConfigData, NormalizedModel, ReasoningFamily } from './configPageSupport'
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
import { useModelLiveVerification } from './useModelLiveVerification'
import modelStyles from './ConfigPageModelsSection.module.css'

interface ConfigPageModelsSectionProps {
  config: ConfigData | null
  isPythonCLI: boolean
  isReadonly: boolean
  canVerifyModels: boolean
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
  canVerifyModels,
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
  const [addModelsOpen, setAddModelsOpen] = useState(false)
  const [endpointFilter, setEndpointFilter] = useState<ModelEndpointFilter>('all')
  const [roleFilter, setRoleFilter] = useState<ModelRoleFilter>('all')
  const [reasoningFamilySearch, setReasoningFamilySearch] = useState('')
  const [selectedModelKeys, setSelectedModelKeys] = useState<Set<string>>(new Set())
  const [bulkDeletePending, setBulkDeletePending] = useState(false)
  const [operationError, setOperationError] = useState<string | null>(null)
  const [modelsPendingDelete, setModelsPendingDelete] = useState<string[]>([])
  const [reasoningFamilyPendingDelete, setReasoningFamilyPendingDelete] = useState<string | null>(
    null,
  )
  const [reasoningFamilyDeletePending, setReasoningFamilyDeletePending] = useState(false)
  const [reasoningFamilyDeleteError, setReasoningFamilyDeleteError] = useState<string | null>(null)
  const autoVerificationKey = useMemo(
    () =>
      JSON.stringify(
        models
          .filter((model) => model.endpoints?.some((endpoint) => endpoint.endpoint.trim()))
          .map((model) => ({
            name: model.name,
            endpoints: model.endpoints?.map((endpoint) => endpoint.endpoint),
          })),
      ),
    [models],
  )
  const { states: liveVerificationStates, verify: verifyModel } =
    useModelLiveVerification(autoVerificationKey)
  useEffect(() => {
    const configuredModels = JSON.parse(autoVerificationKey) as Array<{ name: string }>
    if (!canVerifyModels || configuredModels.length === 0) return

    let cancelled = false
    const cacheKey = `vllm-sr.models.auto-verified.${Array.from(autoVerificationKey).reduce(
      (hash, character) => ((hash << 5) - hash + character.charCodeAt(0)) | 0,
      0,
    )}`
    const timeout = window.setTimeout(() => {
      try {
        if (window.sessionStorage.getItem(cacheKey)) return
        window.sessionStorage.setItem(cacheKey, '1')
      } catch {
        // Browsers can disable session storage; verification still works.
      }
      void (async () => {
        for (const model of configuredModels.slice(0, 10)) {
          if (cancelled) return
          await verifyModel(model.name)
        }
      })()
    }, 0)
    return () => {
      cancelled = true
      window.clearTimeout(timeout)
    }
  }, [autoVerificationKey, canVerifyModels, verifyModel])

  const reasoningFamilyOptions = useMemo(() => getReasoningFamilyFilterOptions(models), [models])
  const modelReferenceCounts = useMemo(() => getModelReferenceCounts(config), [config])
  const filteredModels = useMemo(
    () =>
      filterModelInventory(models, {
        search: modelsSearch,
        reasoningFamily: reasoningFamilyFilter,
        endpointState: endpointFilter,
        role: roleFilter,
        defaultModel,
      }),
    [defaultModel, endpointFilter, models, modelsSearch, reasoningFamilyFilter, roleFilter],
  )
  const filtersActive = Boolean(
    modelsSearch.trim() ||
      reasoningFamilyFilter !== 'all' ||
      endpointFilter !== 'all' ||
      roleFilter !== 'all',
  )

  const getDeleteBlocker = (modelName: string) =>
    getModelDeleteBlocker(modelName, defaultModel, modelReferenceCounts)

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
        <h4
          style={{
            margin: '0 0 1rem 0',
            fontSize: '0.875rem',
            fontWeight: 600,
            color: 'var(--color-text-secondary)',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}
        >
          Endpoints for {model.name}
        </h4>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
              <th
                style={{
                  padding: '0.5rem',
                  textAlign: 'left',
                  fontSize: '0.875rem',
                  fontWeight: 600,
                  color: 'var(--color-text-secondary)',
                }}
              >
                Name
              </th>
              <th
                style={{
                  padding: '0.5rem',
                  textAlign: 'left',
                  fontSize: '0.875rem',
                  fontWeight: 600,
                  color: 'var(--color-text-secondary)',
                }}
              >
                Address
              </th>
              <th
                style={{
                  padding: '0.5rem',
                  textAlign: 'center',
                  fontSize: '0.875rem',
                  fontWeight: 600,
                  color: 'var(--color-text-secondary)',
                  width: '100px',
                }}
              >
                Protocol
              </th>
              <th
                style={{
                  padding: '0.5rem',
                  textAlign: 'center',
                  fontSize: '0.875rem',
                  fontWeight: 600,
                  color: 'var(--color-text-secondary)',
                  width: '100px',
                }}
              >
                Weight
              </th>
            </tr>
          </thead>
          <tbody>
            {model.endpoints.map((ep, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                <td style={{ padding: '0.75rem 0.5rem', fontSize: '0.875rem', fontWeight: 500 }}>
                  {ep.name}
                </td>
                <td
                  style={{
                    padding: '0.75rem 0.5rem',
                    fontSize: '0.875rem',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--color-text-secondary)',
                  }}
                >
                  {isReadonly ? '************' : ep.endpoint || 'N/A'}
                </td>
                <td style={{ padding: '0.75rem 0.5rem', textAlign: 'center' }}>
                  <span
                    style={{
                      padding: '0.25rem 0.5rem',
                      background:
                        ep.protocol === 'https'
                          ? 'rgba(34, 197, 94, 0.15)'
                          : 'rgba(234, 179, 8, 0.15)',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      textTransform: 'uppercase',
                    }}
                  >
                    {ep.protocol}
                  </span>
                </td>
                <td
                  style={{
                    padding: '0.75rem 0.5rem',
                    textAlign: 'center',
                    fontSize: '0.875rem',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
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
          {
            label: 'Context Window',
            value: model.context_window_size ? `${model.context_window_size}` : 'N/A',
          },
        ],
      },
    ]

    if (
      model.description ||
      model.capabilities?.length ||
      model.tags?.length ||
      model.loras?.length ||
      typeof model.quality_score === 'number'
    ) {
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
          {
            label: 'Quality Score',
            value: typeof model.quality_score === 'number' ? `${model.quality_score}` : 'N/A',
          },
        ],
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
            fullWidth: true,
          },
        ],
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
        ],
      })
    }

    openViewModal(
      `Model: ${model.name}`,
      sections,
      () => handleEditModel(model),
      isReadonly
        ? []
        : [
            {
              label: 'Delete model',
              tone: 'destructive',
              onClick: () => handleDeleteModel(model),
            },
          ],
    )
  }

  const handleBatchImport = async (input: ModelBatchImportInput) => {
    if (!config) return
    const nextConfig = cloneConfigData(config)
    const providers = ensureProvidersConfig(nextConfig)
    const knownNames = new Set(models.map((model) => model.name))

    for (const discovered of input.models) {
      const modelName = `${input.namePrefix}${discovered.id}`.trim()
      if (!modelName) throw new Error('Model name is required.')
      if (knownNames.has(modelName)) throw new Error(`Model “${modelName}” already exists.`)
      knownNames.add(modelName)
      const endpointName = `${modelName.replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-|-$/g, '') || 'model'}-primary`
      upsertRoutingModelCard(nextConfig, modelName, {
        param_size: input.paramSize || undefined,
        context_window_size: input.contextWindowSize,
        description: input.description || undefined,
        capabilities: input.capabilities.length > 0 ? input.capabilities : undefined,
        tags: input.tags.length > 0 ? input.tags : undefined,
        loras: input.loras.length > 0 ? input.loras.map((name) => ({ name })) : undefined,
        quality_score: input.qualityScore,
        modality: input.modality || undefined,
      })
      providers.models.push({
        name: modelName,
        reasoning_family: input.reasoningFamily || undefined,
        provider_model_id: discovered.id,
        api_format: input.apiFormat,
        external_model_ids: { [input.providerId]: discovered.id },
        pricing: input.pricing,
        backend_refs: [
          {
            name: endpointName,
            weight: input.endpointWeight,
            type: input.providerId,
            provider: input.runtimeProvider,
            base_url: input.baseUrl,
            auth_header: input.authHeader || undefined,
            auth_prefix: input.authPrefix || undefined,
            extra_headers:
              Object.keys(input.extraHeaders).length > 0 ? input.extraHeaders : undefined,
            api_version: input.apiVersion || undefined,
            chat_path: input.chatPath || undefined,
            api_key: input.apiKey || undefined,
            api_key_env: input.apiKeyEnv || undefined,
          },
        ],
      })
    }
    await saveConfig(nextConfig)
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
        pricing: model.pricing || {},
      },
      [
        {
          name: 'reasoning_family',
          label: 'Reasoning Family',
          type: 'select',
          options: reasoningFamilyNames,
          description: 'Select from configured reasoning families',
        },
        {
          name: 'provider_model_id',
          label: 'Provider Model ID',
          type: 'text',
          placeholder: 'e.g., openai/gpt-4.1',
          description:
            'Concrete upstream model identifier stored under providers.models[].provider_model_id',
        },
        {
          name: 'api_format',
          label: 'API Format',
          type: 'text',
          placeholder: 'e.g., openai',
          description: 'Provider-specific wire format stored under providers.models[].api_format',
        },
        {
          name: 'param_size',
          label: 'Parameter Size',
          type: 'text',
          placeholder: 'e.g., 8B',
        },
        {
          name: 'context_window_size',
          label: 'Context Window Size',
          type: 'number',
          placeholder: 'e.g., 131072',
        },
        {
          name: 'modality',
          label: 'Modality',
          type: 'text',
          placeholder: 'e.g., text, omni, diffusion',
        },
        {
          name: 'description',
          label: 'Description',
          type: 'textarea',
          placeholder: 'Short routing-facing model description',
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
            context_window_size: data.context_window_size
              ? Number(data.context_window_size)
              : undefined,
            description: data.description || undefined,
            capabilities: capabilities.length > 0 ? capabilities : undefined,
            loras: loras.length > 0 ? loras : undefined,
            tags: tags.length > 0 ? tags : undefined,
            quality_score:
              data.quality_score === '' || data.quality_score === undefined
                ? undefined
                : Number(data.quality_score),
            modality: data.modality || undefined,
          })
          type ProviderModel = NonNullable<ConfigData['providers']>['models'][number]
          providers.models = providers.models.map((providerModel: ProviderModel) =>
            providerModel.name === model.name
              ? {
                  ...providerModel,
                  ...buildProviderModelPayload(model.name, data, providerModel),
                }
              : providerModel,
          )
        } else if (newConfig.model_config) {
          newConfig.model_config[model.name] = {
            ...newConfig.model_config[model.name],
            reasoning_family: data.reasoning_family,
            pricing: normalizeModelPricing(data.pricing),
            api_format: typeof data.api_format === 'string' ? data.api_format : undefined,
            external_model_ids: normalizeModelStringMap(data.external_model_ids),
            preferred_endpoints: normalizeModelBackendRefs(data.backend_refs)
              .map((backendRef) => backendRef.name || '')
              .filter(Boolean),
            model_id:
              typeof data.provider_model_id === 'string' ? data.provider_model_id : model.name,
          }
        }
        await saveConfig(newConfig)
      },
      'edit',
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
        providers.models = providers.models.filter(
          (providerModel: ProviderModel) => !namesToDelete.has(providerModel.name),
        )
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
      setOperationError(
        error instanceof Error ? error.message : 'Failed to delete the selected models.',
      )
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
    onExpandedModelsChange((prev) => {
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
            { label: 'Parameter', value: familyConfig.parameter },
          ],
        },
      ],
      () => handleEditReasoningFamily(familyName),
      isReadonly
        ? []
        : [
            {
              label: 'Delete family',
              tone: 'destructive',
              onClick: () => handleDeleteReasoningFamily(familyName),
            },
          ],
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
          description: 'Type of reasoning family',
        },
        {
          name: 'parameter',
          label: 'Parameter',
          type: 'text',
          required: true,
          placeholder: 'e.g., reasoning_effort',
          description: 'Parameter name for reasoning control',
        },
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
      },
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
          description: 'Unique name for this reasoning family',
        },
        {
          name: 'type',
          label: 'Type',
          type: 'select',
          options: ['reasoning_effort', 'chat_template_kwargs'],
          required: true,
          description: 'Type of reasoning family',
        },
        {
          name: 'parameter',
          label: 'Parameter',
          type: 'text',
          required: true,
          placeholder: 'e.g., reasoning_effort',
          description: 'Parameter name for reasoning control',
        },
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
      'add',
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

  type ReasoningFamilyRow = { name: string; type: string; parameter: string; modelCount: number }
  const reasoningFamilyData: ReasoningFamilyRow[] = Object.entries(reasoningFamilies).map(
    ([name, config]) => ({
      name,
      type: config.type,
      parameter: config.parameter,
      modelCount: models.filter((model) => model.reasoning_family === name).length,
    }),
  )
  const normalizedReasoningFamilySearch = reasoningFamilySearch.trim().toLocaleLowerCase()
  const filteredReasoningFamilyData = normalizedReasoningFamilySearch
    ? reasoningFamilyData.filter(
        (family) =>
          family.name.toLocaleLowerCase().includes(normalizedReasoningFamilySearch) ||
          family.type.toLocaleLowerCase().includes(normalizedReasoningFamilySearch) ||
          family.parameter.toLocaleLowerCase().includes(normalizedReasoningFamilySearch),
      )
    : reasoningFamilyData

  const reasoningFamilyColumns: Column<ReasoningFamilyRow>[] = [
    {
      key: 'name',
      header: 'Reasoning Family',
      width: '360px',
      sortable: true,
      render: (row) => (
        <div className={modelStyles.reasoningFamilyIdentity}>
          <span className={modelStyles.reasoningFamilyGlyph} aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <circle cx="7" cy="7" r="2.25" />
              <circle cx="17" cy="7" r="2.25" />
              <circle cx="12" cy="17" r="2.25" />
              <path d="M8.8 8.4 11 14.7M15.2 8.4 13 14.7M9.3 7h5.4" />
            </svg>
          </span>
          <span className={modelStyles.reasoningFamilyCopy}>
            <strong title={row.name}>{row.name}</strong>
            <small>
              {row.modelCount
                ? `${row.modelCount} ${row.modelCount === 1 ? 'model' : 'models'}`
                : 'Not assigned'}
            </small>
          </span>
        </div>
      ),
    },
    {
      key: 'type',
      header: 'Control Type',
      width: '250px',
      sortable: true,
      render: (row) => {
        const label =
          row.type === 'reasoning_effort'
            ? 'Reasoning effort'
            : row.type === 'chat_template_kwargs'
              ? 'Template argument'
              : 'Custom control'
        return (
          <span className={modelStyles.reasoningFamilyType}>
            <i aria-hidden="true" />
            <span>
              <strong>{label}</strong>
              <small>{row.type}</small>
            </span>
          </span>
        )
      },
    },
    {
      key: 'parameter',
      header: 'Request Parameter',
      sortable: true,
      render: (row) => (
        <code className={modelStyles.reasoningFamilyParameter}>{row.parameter}</code>
      ),
    },
  ]

  return (
    <>
      <ConfigPageManagerLayout
        title="Models"
        description="Connect the models behind your Mixture-of-Models."
      >
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
              onAddModel={() => setAddModelsOpen(true)}
              onViewModel={handleViewModel}
              expandedModels={expandedModels}
              onToggleExpand={handleToggleExpand}
              renderExpandedRow={renderModelEndpoints}
              getDeleteBlocker={getDeleteBlocker}
              liveVerificationStates={liveVerificationStates}
              onVerifyModel={(modelName) => void verifyModel(modelName)}
              canVerifyModels={canVerifyModels}
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
              openOnRowClick
              emptyMessage="No reasoning families configured"
              className={`${styles.managerTable} ${modelStyles.reasoningFamilyTable}`}
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

      {addModelsOpen ? (
        <ConfigPageAddModelsDialog
          isOpen
          existingModelNames={models.map((model) => model.name)}
          reasoningFamilies={Object.keys(reasoningFamilies)}
          onClose={() => setAddModelsOpen(false)}
          onImport={handleBatchImport}
        />
      ) : null}

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
        details={
          reasoningFamilyDeleteError ? (
            <span role="alert">{reasoningFamilyDeleteError}</span>
          ) : undefined
        }
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
