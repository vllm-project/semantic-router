import { useMemo, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react'
import styles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageModelEndpoints from './ConfigPageModelEndpoints'
import ConfigPageModelInventoryPanel from './ConfigPageModelInventoryPanel'
import ConfigPageConnectModelsDialog, {
  type ConnectedModelInput,
} from './ConfigPageConnectModelsDialog'
import ModelDeleteDialog from './ModelDeleteDialog'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'
import { normalizeStringList } from '../components/structuredFieldEditorSupport'
import type { ViewSection } from '../components/ViewModal'
import { ConfigData, NormalizedModel, ReasoningFamily } from './configPageSupport'
import type { RoutingModelCard } from './configPageSupport'
import useBuiltInModelCatalog from '../hooks/useBuiltInModelCatalog'
import {
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
  normalizeModelEvaluations,
  normalizeModelLoras,
  normalizeModelPricing,
  normalizeModelStringMap,
} from './configPageModelFormSupport'
import { getModelStructuredFormFields } from './configPageModelFormFields'
import {
  ModelBackendRefsEditor,
  ModelCapabilitiesEditor,
  ModelExternalIdsEditor,
  ModelEvaluationsEditor,
  ModelLorasEditor,
  ModelPricingEditor,
  ModelReliabilityEditor,
  ModelTagsEditor,
} from './configPageModelStructuredEditors'
import { useModelLiveVerification } from './useModelLiveVerification'

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

const hasCardMetadata = (card: Omit<RoutingModelCard, 'name'>): boolean =>
  Object.values(card).some((value) =>
    Array.isArray(value) ? value.length > 0 : value !== undefined && value !== '',
  )

const modelCardPatch = (data: Record<string, unknown>): Omit<RoutingModelCard, 'name'> => {
  const capabilities = normalizeStringList(data.capabilities)
  const tags = normalizeStringList(data.tags)
  const loras = normalizeModelLoras(data.loras)
  const evaluations = normalizeModelEvaluations(data.evaluations)
  return {
    param_size:
      typeof data.param_size === 'string' && data.param_size.trim()
        ? data.param_size.trim()
        : undefined,
    context_window_size: data.context_window_size ? Number(data.context_window_size) : undefined,
    description:
      typeof data.description === 'string' && data.description.trim()
        ? data.description.trim()
        : undefined,
    capabilities: capabilities.length > 0 ? capabilities : undefined,
    loras: loras.length > 0 ? loras : undefined,
    tags: tags.length > 0 ? tags : undefined,
    evaluations: evaluations.length > 0 ? evaluations : undefined,
    modality:
      typeof data.modality === 'string' && data.modality.trim() ? data.modality.trim() : undefined,
  }
}

const writeModelCard = (
  config: ConfigData,
  cardID: string,
  card: Omit<RoutingModelCard, 'name'>,
) => {
  if (hasCardMetadata(card)) upsertRoutingModelCard(config, cardID, card)
  else removeRoutingModelCard(config, cardID)
}

const modelDialogFields = (reasoningFamilyNames: string[], mode: 'add' | 'edit'): FieldConfig[] => [
  ...(mode === 'add'
    ? [
        {
          name: 'model_name',
          label: 'Model Name',
          type: 'text' as const,
          required: true,
          placeholder: 'e.g., openai/gpt-4',
          description: 'Unique identifier for the model',
        },
      ]
    : []),
  {
    name: 'catalog',
    label: 'Built-in Catalog Model',
    type: 'text',
    placeholder: 'e.g., organization/model-id',
    description:
      mode === 'add'
        ? 'Optional. Leave empty for a custom or self-hosted model.'
        : 'Canonical card identity. Leave empty for a custom or self-hosted model.',
  },
  {
    name: 'reasoning_family',
    label: 'Reasoning Family',
    type: 'select',
    options: reasoningFamilyNames,
    description: 'Optional for custom models. Built-in models inherit this automatically.',
  },
  {
    name: 'reasoning_type',
    label: 'Inline Reasoning Type',
    type: 'select',
    options: ['reasoning_effort', 'chat_template_kwargs', 'top_level_reasoning_effort'],
    description: 'Use only when no built-in family matches a custom model.',
  },
  {
    name: 'reasoning_parameter',
    label: 'Inline Reasoning Parameter',
    type: 'text',
    placeholder: 'e.g., enable_thinking',
  },
  {
    name: 'reasoning_levels',
    label: 'Inline Reasoning Levels',
    type: 'text',
    placeholder: 'low, medium, high',
  },
  {
    name: 'reasoning_default',
    label: 'Inline Reasoning Default',
    type: 'text',
    placeholder: 'medium',
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
  { name: 'param_size', label: 'Parameter Size', type: 'text', placeholder: 'e.g., 8B' },
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
]

const newModelFormData = (): Record<string, unknown> => ({
  model_name: '',
  catalog: '',
  reasoning_family: '',
  reasoning_type: '',
  reasoning_parameter: '',
  reasoning_levels: '',
  reasoning_default: '',
  provider_model_id: '',
  api_format: '',
  external_model_ids: {},
  param_size: '',
  context_window_size: '',
  description: '',
  capabilities: [],
  loras: [],
  tags: [],
  evaluations: [],
  modality: '',
  backend_refs: [
    {
      name: 'endpoint-1',
      endpoint: 'localhost:8000',
      protocol: 'http',
      weight: 1,
      provider: 'vllm',
    },
  ],
  pricing: {
    currency: 'USD',
    prompt_per_1m: 0,
    cached_input_per_1m: 0,
    completion_per_1m: 0,
  },
})

const editModelFormData = (model: NormalizedModel): Record<string, unknown> => ({
  catalog: model.catalog || '',
  reasoning_family: model.reasoning?.family || '',
  reasoning_type: model.reasoning?.type || '',
  reasoning_parameter: model.reasoning?.parameter || '',
  reasoning_levels: model.reasoning?.levels?.join(', ') || '',
  reasoning_default: model.reasoning?.default || '',
  provider_model_id: model.provider_model_id || '',
  api_format: model.api_format || '',
  external_model_ids: model.external_model_ids || {},
  param_size: model.card_override?.param_size || '',
  context_window_size: model.card_override?.context_window_size || '',
  description: model.card_override?.description || '',
  capabilities: model.card_override?.capabilities || [],
  loras: model.card_override?.loras || [],
  tags: model.card_override?.tags || [],
  evaluations: model.card_override?.evaluations || [],
  modality: model.card_override?.modality || '',
  backend_refs: model.backend_refs || [],
  pricing: model.pricing || {},
  reliability: model.reliability || {},
})

const baseModelViewSection = (model: NormalizedModel, defaultModel: string): ViewSection => ({
  title: 'Basic Information',
  fields: [
    { label: 'Model Name', value: model.name },
    { label: 'Catalog Model', value: model.catalog || 'Custom' },
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
})

const modelViewSections = (
  model: NormalizedModel,
  defaultModel: string,
  isReadonly: boolean,
): ViewSection[] => {
  const sections = [baseModelViewSection(model, defaultModel)]
  const metadata = modelRoutingMetadataSection(model)
  if (metadata) sections.push(metadata)
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
  if (model.backend_refs?.length) {
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
  if (model.pricing)
    sections.push(
      editorViewSection(
        'Pricing',
        'Token Pricing',
        <ModelPricingEditor value={model.pricing} readOnly />,
      ),
    )
  if (model.reliability)
    sections.push(
      editorViewSection(
        'Delivery',
        'Policy',
        <ModelReliabilityEditor value={model.reliability} readOnly />,
      ),
    )
  return sections
}

const modelRoutingMetadataSection = (model: NormalizedModel): ViewSection | null => {
  const present =
    model.description ||
    model.capabilities?.length ||
    model.tags?.length ||
    model.loras?.length ||
    model.evaluations?.length
  if (!present) return null
  return {
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
      ...(model.evaluations?.length
        ? [
            {
              label: 'Operator Evaluations',
              value: <ModelEvaluationsEditor value={model.evaluations} readOnly />,
              fullWidth: true,
            },
          ]
        : []),
    ],
  }
}

const editorViewSection = (title: string, label: string, value: ReactNode): ViewSection => ({
  title,
  fields: [{ label, value, fullWidth: true }],
})

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
  const [endpointFilter, setEndpointFilter] = useState<ModelEndpointFilter>('all')
  const [roleFilter, setRoleFilter] = useState<ModelRoleFilter>('all')
  const [reasoningFamilySearch, setReasoningFamilySearch] = useState('')
  const [selectedModelKeys, setSelectedModelKeys] = useState<Set<string>>(new Set())
  const [bulkDeletePending, setBulkDeletePending] = useState(false)
  const [operationError, setOperationError] = useState<string | null>(null)
  const [modelsPendingDelete, setModelsPendingDelete] = useState<string[]>([])
  const [connectModelsOpen, setConnectModelsOpen] = useState(false)
  const liveVerification = useModelLiveVerification(config)
  const { catalog: modelCatalog, error: modelCatalogError } = useBuiltInModelCatalog()

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

  const handleViewModel = (model: ModelRow) => {
    openViewModal(`Model: ${model.name}`, modelViewSections(model, defaultModel, isReadonly), () =>
      handleEditModel(model),
    )
  }

  const handleAddModel = () => {
    const reasoningFamilyNames = Object.keys(reasoningFamilies)

    openEditModal(
      'Add New Model',
      newModelFormData(),
      modelDialogFields(reasoningFamilyNames, 'add'),
      async (data) => {
        if (!config) {
          return
        }
        validateModelStructuredFields(data)
        const modelName = validateNewModelName(data.model_name, models)
        const newConfig = cloneConfigData(config)

        if (isPythonCLI) {
          const providers = ensureProvidersConfig(newConfig)
          const catalogID =
            typeof data.catalog === 'string' && data.catalog.trim()
              ? data.catalog.trim()
              : modelName
          writeModelCard(newConfig, catalogID, modelCardPatch(data))
          providers.models.push(buildProviderModelPayload(modelName, data))
        } else {
          if (!newConfig.model_config) {
            newConfig.model_config = {}
          }
          newConfig.model_config[modelName] = {
            reasoning_family:
              typeof data.reasoning_family === 'string' ? data.reasoning_family : undefined,
            pricing: normalizeModelPricing(data.pricing),
            api_format: typeof data.api_format === 'string' ? data.api_format : undefined,
            external_model_ids: normalizeModelStringMap(data.external_model_ids),
            preferred_endpoints: normalizeModelBackendRefs(data.backend_refs)
              .map((backendRef) => backendRef.name || '')
              .filter(Boolean),
            model_id:
              typeof data.provider_model_id === 'string' && data.provider_model_id.trim()
                ? data.provider_model_id.trim()
                : modelName,
          }
        }
        await saveConfig(newConfig)
      },
      'add',
    )
  }

  const handleConnectModels = async ({
    provider,
    baseUrl,
    apiKey,
    modelIds,
    modelNames,
    catalogModels,
    reasoningFamily,
    metadata,
    pricing,
    reliability,
  }: ConnectedModelInput) => {
    if (!config) return
    if (!isPythonCLI) {
      throw new Error('Quick connect requires the canonical providers.models configuration.')
    }
    const newConfig = cloneConfigData(config)
    const providers = ensureProvidersConfig(newConfig)

    for (const modelId of modelIds) {
      const modelName = validateNewModelName(modelNames[modelId] ?? modelId, models)
      const catalogID = catalogModels[modelId]
      providers.models.push({
        name: modelName,
        ...(catalogID ? { catalog: catalogID } : {}),
        ...(!catalogID && reasoningFamily ? { reasoning: { family: reasoningFamily } } : {}),
        provider_model_id: modelId,
        api_format: provider.apiFormat,
        pricing,
        reliability,
        backend_refs: [
          {
            name: `${provider.id}-primary`,
            base_url: baseUrl,
            provider: provider.id,
            ...(apiKey ? { api_key: apiKey } : {}),
          },
        ],
      })
      writeModelCard(newConfig, catalogID || modelName, metadata)
    }
    await saveConfig(newConfig)
  }

  const handleEditModel = (model: ModelRow) => {
    const reasoningFamilyNames = Object.keys(reasoningFamilies)

    openEditModal(
      `Edit Model: ${model.name}`,
      editModelFormData(model),
      modelDialogFields(reasoningFamilyNames, 'edit'),
      async (data) => {
        if (!config) {
          return
        }
        validateModelStructuredFields(data)
        const newConfig = cloneConfigData(config)
        if (isPythonCLI && newConfig.providers?.models) {
          const providers = ensureProvidersConfig(newConfig)
          const oldCardID = model.catalog || model.name
          const nextCatalog =
            typeof data.catalog === 'string' && data.catalog.trim()
              ? data.catalog.trim()
              : undefined
          const nextCardID = nextCatalog || model.name
          if (oldCardID !== nextCardID) removeRoutingModelCard(newConfig, oldCardID)
          writeModelCard(newConfig, nextCardID, modelCardPatch(data))
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
            reasoning_family:
              typeof data.reasoning_family === 'string' ? data.reasoning_family : undefined,
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
          const model = models.find((candidate) => candidate.name === modelName)
          removeRoutingModelCard(newConfig, model?.catalog || modelName)
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

    openViewModal(`Reasoning Family: ${familyName}`, [
      {
        title: 'Configuration',
        fields: [
          { label: 'Family Name', value: familyName },
          { label: 'Type', value: familyConfig.type },
          { label: 'Parameter', value: familyConfig.parameter },
          { label: 'Levels', value: familyConfig.levels?.join(', ') || 'N/A' },
          { label: 'Default', value: familyConfig.default || 'N/A' },
        ],
      },
    ])
  }
  type ReasoningFamilyRow = {
    name: string
    type: string
    parameter: string
    levels: string
    defaultLevel: string
  }
  const reasoningFamilyData: ReasoningFamilyRow[] = Object.entries(reasoningFamilies).map(
    ([name, config]) => ({
      name,
      type: config.type,
      parameter: config.parameter,
      levels: config.levels?.join(', ') || '',
      defaultLevel: config.default || '',
    }),
  )
  const normalizedReasoningFamilySearch = reasoningFamilySearch.trim().toLocaleLowerCase()
  const filteredReasoningFamilyData = normalizedReasoningFamilySearch
    ? reasoningFamilyData.filter(
        (family) =>
          family.name.toLocaleLowerCase().includes(normalizedReasoningFamilySearch) ||
          family.type.toLocaleLowerCase().includes(normalizedReasoningFamilySearch) ||
          family.parameter.toLocaleLowerCase().includes(normalizedReasoningFamilySearch) ||
          family.levels.toLocaleLowerCase().includes(normalizedReasoningFamilySearch),
      )
    : reasoningFamilyData

  const reasoningFamilyColumns: Column<ReasoningFamilyRow>[] = [
    {
      key: 'name',
      header: 'Family Name',
      sortable: true,
      render: (row) => <span className={styles.reasoningFamilyName}>{row.name}</span>,
    },
    {
      key: 'type',
      header: 'Type',
      width: '200px',
      sortable: true,
      render: (row) => <span className={styles.reasoningFamilyType}>{row.type}</span>,
    },
    {
      key: 'parameter',
      header: 'Parameter',
      sortable: true,
      render: (row) => <code className={styles.reasoningFamilyParameter}>{row.parameter}</code>,
    },
    {
      key: 'levels',
      header: 'Levels',
      render: (row) => row.levels || 'N/A',
    },
    {
      key: 'defaultLevel',
      header: 'Default',
      width: '120px',
      render: (row) => row.defaultLevel || 'N/A',
    },
  ]

  return (
    <>
      <ConfigPageManagerLayout
        title="Models"
        description="Manage provider models, reasoning families, and the endpoint inventory available to routing decisions."
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
              onAddModel={() => setConnectModelsOpen(true)}
              onViewModel={handleViewModel}
              onEditModel={handleEditModel}
              onDeleteModel={handleDeleteModel}
              expandedModels={expandedModels}
              onToggleExpand={handleToggleExpand}
              renderExpandedRow={(model) => (
                <ConfigPageModelEndpoints model={model} redactEndpoints={isReadonly} />
              )}
              getDeleteBlocker={getDeleteBlocker}
              liveVerificationStates={liveVerification.states}
              onVerifyModel={(modelName) => void liveVerification.verify(modelName)}
              canVerifyModels={canVerifyModels}
            />
          </div>

          <div className={styles.sectionTableBlock}>
            {modelCatalogError ? (
              <p role="status" className={styles.inlineInfo}>
                Using the bundled catalog because the live catalog API is unavailable.
              </p>
            ) : null}
            <TableHeader
              title="Built-in Reasoning Families"
              count={filteredReasoningFamilyData.length}
              searchPlaceholder="Search family, type, or parameter..."
              searchValue={reasoningFamilySearch}
              onSearchChange={setReasoningFamilySearch}
              variant="embedded"
            />
            <DataTable
              columns={reasoningFamilyColumns}
              data={filteredReasoningFamilyData}
              keyExtractor={(row) => row.name}
              onView={(row) => handleViewReasoningFamily(row.name)}
              emptyMessage="No built-in reasoning families are available"
              className={styles.managerTable}
              readonly
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

      <ConfigPageConnectModelsDialog
        isOpen={connectModelsOpen}
        existingModelNames={[
          ...models.map((model) => model.name),
          ...(config?.entrypoints ?? []).flatMap((entrypoint) => entrypoint.model_names),
        ]}
        reasoningFamilies={Object.keys(reasoningFamilies)}
        catalog={modelCatalog}
        onClose={() => setConnectModelsOpen(false)}
        onImport={handleConnectModels}
        onManualSetup={() => handleAddModel()}
      />
    </>
  )
}
