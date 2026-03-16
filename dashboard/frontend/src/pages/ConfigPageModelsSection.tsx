import type { Dispatch, SetStateAction } from 'react'
import styles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import EndpointsEditor, { type Endpoint } from '../components/EndpointsEditor'
import type { ViewSection } from '../components/ViewModal'
import type {
  ConfigData,
  ModelConfigEntry,
  NormalizedModel,
  ReasoningFamily,
  TABLE_COLUMN_WIDTH,
  VLLMEndpoint,
} from './configPageSupport'
import {
  ensureProviderDefaultsConfig,
  ensureProvidersConfig,
  cloneConfigData,
  removeRoutingModelCard,
  upsertRoutingModelCard,
} from './configPageCanonicalization'
import {
  mergeProviderBackendRefs,
  normalizeEndpoint,
  normalizeEndpoints,
} from './configPageSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'

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
  const filteredModels = models.filter(model =>
    model.name.toLowerCase().includes(modelsSearch.toLowerCase()) ||
    model.reasoning_family?.toLowerCase().includes(modelsSearch.toLowerCase())
  )

  type ModelRow = NormalizedModel
  const modelColumns: Column<ModelRow>[] = [
    {
      key: 'name',
      header: 'Model Name',
      sortable: true,
      render: (row) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontWeight: 600 }}>{row.name}</span>
          {row.name === defaultModel && (
            <span className={styles.badge} style={{ background: 'rgba(118, 185, 0, 0.15)', color: 'var(--color-primary)' }}>
              Default
            </span>
          )}
        </div>
      )
    },
    {
      key: 'reasoning_family',
      header: 'Reasoning Family',
      width: TABLE_COLUMN_WIDTH.medium,
      sortable: true,
      render: (row) => row.reasoning_family ? (
        <span className={styles.badge} style={{ background: 'rgba(0, 212, 255, 0.15)', color: 'var(--color-accent-cyan)' }}>
          {row.reasoning_family}
        </span>
      ) : <span style={{ color: 'var(--color-text-secondary)' }}>N/A</span>
    },
    {
      key: 'endpoints',
      header: 'Endpoints',
      width: TABLE_COLUMN_WIDTH.compact,
      align: 'center',
      render: (row) => {
        const count = row.endpoints?.length || 0
        return (
          <span style={{ color: count > 0 ? 'var(--color-text)' : 'var(--color-text-secondary)' }}>
            {count} {count === 1 ? 'endpoint' : 'endpoints'}
          </span>
        )
      }
    },
    {
      key: 'pricing',
      header: 'Pricing',
      width: TABLE_COLUMN_WIDTH.medium,
      render: (row) => {
        if (!row.pricing) return <span style={{ color: 'var(--color-text-secondary)' }}>N/A</span>
        const currency = row.pricing.currency || 'USD'
        const prompt = row.pricing.prompt_per_1m?.toFixed(2) || '0.00'
        return (
          <span style={{ fontSize: '0.875rem', fontFamily: 'var(--font-mono)' }}>
            {prompt} {currency}/1M
          </span>
        )
      }
    }
  ]

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

    if (model.endpoints && model.endpoints.length > 0) {
      sections.push({
        title: `Endpoints (${model.endpoints.length})`,
        fields: [
          {
            label: 'Configured Endpoints',
            value: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {model.endpoints.map((ep, i) => {
                  const isHttps = ep.protocol === 'https'
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
                          {ep.name}
                        </span>
                      </div>
                      <div style={{
                        display: 'flex',
                        gap: '1rem',
                        fontSize: '0.875rem',
                        color: 'var(--color-text-secondary)'
                      }}>
                        <span style={{ fontFamily: 'var(--font-mono)' }}>
                          {isReadonly ? '************' : ep.endpoint}
                        </span>
                        <span>
                          <span style={{
                            padding: '0.125rem 0.5rem',
                            borderRadius: '3px',
                            fontSize: '0.75rem',
                            fontWeight: 600,
                            textTransform: 'uppercase',
                            background: isHttps ? 'rgba(34, 197, 94, 0.15)' : 'rgba(234, 179, 8, 0.15)',
                            color: isHttps ? 'rgb(34, 197, 94)' : 'rgb(234, 179, 8)'
                          }}>
                            {ep.protocol.toUpperCase()}
                          </span>
                        </span>
                        <span>Weight: {ep.weight}</span>
                      </div>
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
          { label: 'Completion (per 1M tokens)', value: model.pricing.completion_per_1m?.toFixed(2) || '0.00' }
        ]
      })
    }

    if (model.access_key) {
      sections.push({
        title: 'Authentication',
        fields: [
          { label: 'API Key', value: '••••••••' }
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
        access_key: '',
        param_size: '',
        context_window_size: '',
        description: '',
        capabilities: '',
        loras: [],
        tags: '',
        quality_score: '',
        modality: '',
        endpoints: [{
          name: 'endpoint-1',
          endpoint: 'localhost:8000',
          protocol: 'http' as const,
          weight: 1
        }],
        currency: 'USD',
        prompt_per_1m: 0,
        completion_per_1m: 0
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
          placeholder: '[{\"name\":\"adapter\",\"description\":\"optional\"}]'
        },
        {
          name: 'endpoints',
          label: 'Endpoints',
          type: 'custom',
          description: 'Configure endpoints for this model',
          customRender: (value: Endpoint[], onChange: (value: Endpoint[]) => void) => (
            <EndpointsEditor endpoints={value || []} onChange={onChange} />
          )
        },
        {
          name: 'access_key',
          label: 'API Key',
          type: 'text',
          placeholder: 'API key for this model',
          description: 'Optional: API key for authentication'
        },
        {
          name: 'currency',
          label: 'Pricing Currency',
          type: 'text',
          placeholder: 'USD',
          description: 'ISO currency code (e.g., USD, EUR, CNY)'
        },
        {
          name: 'prompt_per_1m',
          label: 'Prompt Price per 1M Tokens',
          type: 'number',
          placeholder: '0.50',
          description: 'Cost per 1 million prompt tokens'
        },
        {
          name: 'completion_per_1m',
          label: 'Completion Price per 1M Tokens',
          type: 'number',
          placeholder: '1.50',
          description: 'Cost per 1 million completion tokens'
        }
      ],
      async (data) => {
        if (!config) {
          return
        }
        const endpoints = normalizeEndpoints(data.endpoints)
        const capabilities = listInputToArray(data.capabilities || '')
        const tags = listInputToArray(data.tags || '')
        const loras = Array.isArray(data.loras) ? data.loras : []

        const newConfig = cloneConfigData(config)

        if (isPythonCLI) {
          const providers = ensureProvidersConfig(newConfig)
          upsertRoutingModelCard(newConfig, data.model_name, {
            param_size: data.param_size || undefined,
            context_window_size: data.context_window_size ? Number(data.context_window_size) : undefined,
            description: data.description || undefined,
            capabilities: capabilities.length > 0 ? capabilities : undefined,
            loras: loras.length > 0 ? loras : undefined,
            tags: tags.length > 0 ? tags : undefined,
            quality_score: data.quality_score === '' || data.quality_score === undefined ? undefined : Number(data.quality_score),
            modality: data.modality || undefined,
          })
          providers.models.push({
            name: data.model_name,
            reasoning_family: data.reasoning_family || undefined,
            provider_model_id: data.model_name,
            backend_refs: mergeProviderBackendRefs(undefined, endpoints, data.access_key),
            pricing: {
              currency: data.currency,
              prompt_per_1m: parseFloat(data.prompt_per_1m) || 0,
              completion_per_1m: parseFloat(data.completion_per_1m) || 0
            }
          })
        } else {
          if (!newConfig.model_config) {
            newConfig.model_config = {}
          }
          newConfig.model_config[data.model_name] = {
            reasoning_family: data.reasoning_family,
            preferred_endpoints: endpoints.map((ep: { name: string }) => ep.name),
            pricing: {
              currency: data.currency,
              prompt_per_1m: parseFloat(data.prompt_per_1m) || 0,
              completion_per_1m: parseFloat(data.completion_per_1m) || 0
            }
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
        access_key: model.access_key || '',
        param_size: model.param_size || '',
        context_window_size: model.context_window_size || '',
        description: model.description || '',
        capabilities: (model.capabilities || []).join('\n'),
        loras: model.loras || [],
        tags: (model.tags || []).join('\n'),
        quality_score: model.quality_score ?? '',
        modality: model.modality || '',
        endpoints: normalizeEndpoints(model.endpoints),
        currency: model.pricing?.currency || 'USD',
        prompt_per_1m: model.pricing?.prompt_per_1m || 0,
        completion_per_1m: model.pricing?.completion_per_1m || 0
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
          placeholder: '[{\"name\":\"adapter\",\"description\":\"optional\"}]'
        },
        {
          name: 'endpoints',
          label: 'Endpoints',
          type: 'custom',
          description: 'Configure endpoints for this model',
          customRender: (value: Endpoint[], onChange: (value: Endpoint[]) => void) => (
            <EndpointsEditor endpoints={value || []} onChange={onChange} />
          )
        },
        {
          name: 'access_key',
          label: 'API Key',
          type: 'text',
          placeholder: 'API key for this model',
          description: 'Optional: API key for authentication'
        },
        {
          name: 'currency',
          label: 'Pricing Currency',
          type: 'text',
          placeholder: 'USD',
          description: 'ISO currency code (e.g., USD, EUR, CNY)'
        },
        {
          name: 'prompt_per_1m',
          label: 'Prompt Price per 1M Tokens',
          type: 'number',
          placeholder: '0.50',
          description: 'Cost per 1 million prompt tokens'
        },
        {
          name: 'completion_per_1m',
          label: 'Completion Price per 1M Tokens',
          type: 'number',
          placeholder: '1.50',
          description: 'Cost per 1 million completion tokens'
        }
      ],
      async (data) => {
        if (!config) {
          return
        }
        const newConfig = cloneConfigData(config)
        const endpoints = normalizeEndpoints(data.endpoints)
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
              reasoning_family: data.reasoning_family || undefined,
              provider_model_id: providerModel.provider_model_id || model.name,
              backend_refs: mergeProviderBackendRefs(providerModel.backend_refs, endpoints, data.access_key),
              pricing: {
                currency: data.currency,
                prompt_per_1m: parseFloat(data.prompt_per_1m) || 0,
                completion_per_1m: parseFloat(data.completion_per_1m) || 0
              }
            } : providerModel
          )
        } else if (newConfig.model_config) {
          newConfig.model_config[model.name] = {
            ...newConfig.model_config[model.name],
            reasoning_family: data.reasoning_family,
            preferred_endpoints: endpoints.map((ep: { name: string }) => ep.name),
            pricing: {
              currency: data.currency,
              prompt_per_1m: parseFloat(data.prompt_per_1m) || 0,
              completion_per_1m: parseFloat(data.completion_per_1m) || 0
            }
          }
        }
        await saveConfig(newConfig)
      },
      'edit'
    )
  }

  const handleDeleteModelAction = async (modelName: string) => {
    if (!config) {
      return
    }
    const newConfig = cloneConfigData(config)
    if (isPythonCLI && newConfig.providers?.models) {
      const providers = ensureProvidersConfig(newConfig)
      type ProviderModel = NonNullable<ConfigData['providers']>['models'][number]
      providers.models = providers.models.filter((providerModel: ProviderModel) => providerModel.name !== modelName)
      removeRoutingModelCard(newConfig, modelName)
      const defaults = ensureProviderDefaultsConfig(newConfig)
      if (defaults.default_model === modelName) {
        defaults.default_model = providers.models[0]?.name || ''
      }
    } else if (newConfig.model_config) {
      delete newConfig.model_config[modelName]
    }
    await saveConfig(newConfig)
  }

  const handleDeleteModel = (model: ModelRow) => {
    if (confirm(`Are you sure you want to delete model "${model.name}"?`)) {
      void handleDeleteModelAction(model.name)
    }
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

    openEditModal(
      'Add Reasoning Family',
      { type: 'reasoning_effort', parameter: '' },
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
        delete data.name

        const newConfig = cloneConfigData(config)
        if (isPythonCLI) {
          const defaults = ensureProviderDefaultsConfig(newConfig)
          if (!defaults.reasoning_families) {
            defaults.reasoning_families = {}
          }
          defaults.reasoning_families[familyName] = data
        } else {
          if (!newConfig.reasoning_families) {
            newConfig.reasoning_families = {}
          }
          newConfig.reasoning_families[familyName] = data
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
        <span className={styles.badge} style={{ background: 'rgba(0, 212, 255, 0.15)', color: 'var(--color-accent-cyan)' }}>
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
    <ConfigPageManagerLayout title="Models" description="Manage provider models, reasoning families, and the endpoint inventory available to routing decisions.">
      <div className={styles.sectionPanel}>
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
        <div className={styles.sectionTableBlock}>
          <TableHeader title="Models" count={models.length} searchPlaceholder="Search models..." searchValue={modelsSearch} onSearchChange={onModelsSearchChange} onAdd={handleAddModel} addButtonText="Add Model" disabled={isReadonly} variant="embedded" />
          <DataTable
            columns={modelColumns}
            data={filteredModels}
            keyExtractor={(row) => row.name}
            onView={handleViewModel}
            onEdit={handleEditModel}
            onDelete={handleDeleteModel}
            expandable={true}
            renderExpandedRow={renderModelEndpoints}
            isRowExpanded={(row) => expandedModels.has(row.name)}
            onToggleExpand={handleToggleExpand}
            emptyMessage={modelsSearch ? 'No models match your search' : 'No models configured'}
            className={styles.managerTable}
            readonly={isReadonly}
          />
        </div>
      </div>
    </ConfigPageManagerLayout>
  )
}
