import React, { useState, useEffect } from 'react'
import styles from './ConfigPage.module.css'
import { ConfigSection } from '../components/ConfigNav'
import EditModal, { FieldConfig } from '../components/EditModal'
import {
  ConfigFormat,
  detectConfigFormat,
} from '../types/config'

interface VLLMEndpoint {
  name: string
  address: string
  port: number
  weight: number
  health_check_path: string
}

interface ModelConfig {
  model_id: string
  use_modernbert?: boolean
  threshold: number
  use_cpu: boolean
  category_mapping_path?: string
  pii_mapping_path?: string
  jailbreak_mapping_path?: string
}

interface MCPCategoryModel {
  enabled: boolean
  transport_type: string
  command?: string
  args?: string[]
  env?: Record<string, string>
  url?: string
  tool_name?: string
  threshold: number
  timeout_seconds?: number
}

interface ModelScore {
  model: string
  score: number
  use_reasoning: boolean
  reasoning_description?: string
  reasoning_effort?: string
}

interface Category {
  name: string
  system_prompt?: string
  description?: string
  // model_scores can be array (Python CLI) or object (Legacy: {"gpt-4": 0.9})
  model_scores?: ModelScore[] | Record<string, number>
}

interface ToolFunction {
  name: string
  description: string
  parameters: {
    type: string
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    properties: Record<string, any>
    required?: string[]
  }
}

interface Tool {
  tool: {
    type: string
    function: ToolFunction
  }
  description: string  // Used for similarity matching (from ToolEntry)
  category?: string
  tags?: string[]
}

interface ReasoningFamily {
  type: string
  parameter: string
}

interface ModelPricing {
  currency?: string
  prompt_per_1m?: number
  completion_per_1m?: number
}

interface PIIPolicy {
  allow_by_default: boolean
  pii_types_allowed?: string[]
}

interface ModelConfigEntry {
  reasoning_family?: string
  preferred_endpoints?: string[]
  pricing?: ModelPricing
  pii_policy?: PIIPolicy
}

interface TracingConfig {
  enabled: boolean
  provider: string
  exporter: {
    type: string
    endpoint?: string
    insecure?: boolean
  }
  sampling: {
    type: string
    rate?: number
  }
  resource: {
    service_name: string
    service_version: string
    deployment_environment: string
  }
}

interface APIConfig {
  batch_classification?: {
    max_batch_size: number
    concurrency_threshold: number
    max_concurrency: number
    metrics?: {
      enabled: boolean
      detailed_goroutine_tracking?: boolean
      high_resolution_timing?: boolean
      sample_rate?: number
      duration_buckets?: number[]
      size_buckets?: number[]
    }
  }
}

interface ConfigData {
  // === Python CLI Format (new) ===
  version?: string
  listeners?: Array<{
    name: string
    address: string
    port: number
    timeout?: string
  }>
  signals?: {
    keywords?: Array<{
      name: string
      operator: 'AND' | 'OR'
      keywords: string[]
      case_sensitive: boolean
    }>
    embeddings?: Array<{
      name: string
      threshold: number
      candidates: string[]
      aggregation_method: string
    }>
    domains?: Array<{
      name: string
      description: string
      mmlu_categories?: string[]
    }>
    fact_check?: Array<{ name: string; description: string }>
    user_feedbacks?: Array<{ name: string; description: string }>
    preferences?: Array<{ name: string; description: string }>
  }
  decisions?: Array<{
    name: string
    description: string
    priority: number
    rules: {
      operator: 'AND' | 'OR'
      conditions: Array<{ type: string; name: string }>
    }
    modelRefs: Array<{ model: string; use_reasoning: boolean }>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    plugins?: Array<{ type: string; configuration: Record<string, any> }>
  }>
  providers?: {
    models: Array<{
      name: string
      reasoning_family?: string
      endpoints: Array<{
        name: string
        weight: number
        endpoint: string
        protocol: 'http' | 'https'
      }>
      access_key?: string
    }>
    default_model: string
    reasoning_families?: Record<string, ReasoningFamily>
    default_reasoning_effort?: string
  }

  // === Legacy Format (Go router) ===
  bert_model?: ModelConfig
  semantic_cache?: {
    enabled: boolean
    backend_type?: string
    similarity_threshold: number
    max_entries: number
    ttl_seconds: number
    eviction_policy?: string
    embedding_model?: string
  }
  tools?: {
    enabled: boolean
    top_k: number
    similarity_threshold: number
    tools_db_path: string
    fallback_to_empty: boolean
  }
  prompt_guard?: ModelConfig & { enabled: boolean }
  vllm_endpoints?: VLLMEndpoint[]
  classifier?: {
    category_model?: ModelConfig
    mcp_category_model?: MCPCategoryModel
    pii_model?: ModelConfig
  }
  categories?: Category[]
  default_reasoning_effort?: string
  default_model?: string
  model_config?: Record<string, ModelConfigEntry>
  reasoning_families?: Record<string, ReasoningFamily>
  api?: APIConfig
  observability?: {
    tracing?: TracingConfig
    metrics?: { enabled: boolean }
  }
  [key: string]: unknown
}

interface ConfigPageProps {
  activeSection?: ConfigSection
}

// Helper function to format threshold as percentage
const formatThreshold = (value: number): string => {
  return `${Math.round(value * 100)}%`
}

// Helper function to mask address for security
const maskAddress = (address: string): string => {
  if (address.length <= 8) {
    return '•'.repeat(address.length)
  }
  // Show first 3 and last 3 characters, mask the middle
  const start = address.substring(0, 3)
  const end = address.substring(address.length - 3)
  const middleLength = address.length - 6
  return `${start}${'•'.repeat(middleLength)}${end}`
}

const ConfigPage: React.FC<ConfigPageProps> = ({ activeSection = 'signals' }) => {
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedView, setSelectedView] = useState<'structured' | 'raw'>('structured')
  const [configFormat, setConfigFormat] = useState<ConfigFormat>('python-cli')

  // Router defaults state (from .vllm-sr/router-defaults.yaml)
  const [routerDefaults, setRouterDefaults] = useState<ConfigData | null>(null)
  const [routerDefaultsLoading, setRouterDefaultsLoading] = useState(false)

  // Tools database state
  const [toolsData, setToolsData] = useState<Tool[]>([])
  const [toolsLoading, setToolsLoading] = useState(false)
  const [toolsError, setToolsError] = useState<string | null>(null)

  // Endpoint address visibility state (for security masking)
  const [visibleAddresses, setVisibleAddresses] = useState<Set<number>>(new Set())

  // Edit modal state
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [editModalTitle, setEditModalTitle] = useState('')
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [editModalData, setEditModalData] = useState<any>(null)
  const [editModalFields, setEditModalFields] = useState<FieldConfig[]>([])
  const [editModalMode, setEditModalMode] = useState<'edit' | 'add'>('edit')
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [editModalCallback, setEditModalCallback] = useState<((data: any) => Promise<void>) | null>(null)

  useEffect(() => {
    fetchConfig()
    fetchRouterDefaults()
  }, [])

  // Fetch tools database when config is loaded
  useEffect(() => {
    if (config?.tools?.tools_db_path || routerDefaults?.tools?.tools_db_path) {
      fetchToolsDB()
    }
  }, [config?.tools?.tools_db_path, routerDefaults?.tools?.tools_db_path])

  const fetchConfig = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/router/config/all')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      setConfig(data)
      // Detect config format
      const format = detectConfigFormat(data)
      setConfigFormat(format)
      if (format === 'legacy') {
        console.warn('Legacy config format detected. Consider migrating to Python CLI format.')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch config')
      setConfig(null)
    } finally {
      setLoading(false)
    }
  }

  const fetchRouterDefaults = async () => {
    setRouterDefaultsLoading(true)
    try {
      const response = await fetch('/api/router/config/defaults')
      if (!response.ok) {
        console.warn('Router defaults not available:', response.statusText)
        setRouterDefaults(null)
        return
      }
      const data = await response.json()
      setRouterDefaults(data)
    } catch (err) {
      console.warn('Failed to fetch router defaults:', err)
      setRouterDefaults(null)
    } finally {
      setRouterDefaultsLoading(false)
    }
  }

  const fetchToolsDB = async () => {
    setToolsLoading(true)
    setToolsError(null)
    try {
      const response = await fetch('/api/tools-db')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      setToolsData(data)
    } catch (err) {
      setToolsError(err instanceof Error ? err.message : 'Failed to fetch tools database')
      setToolsData([])
    } finally {
      setToolsLoading(false)
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const saveConfig = async (updatedConfig: any) => {
    try {
      const response = await fetch('/api/router/config/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedConfig),
      })

      if (!response.ok) {
        // Try to read error message from response body
        const errorText = await response.text()
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`
        if (errorText) {
          try {
            const errorJson = JSON.parse(errorText)
            if (errorJson.error || errorJson.message) {
              errorMessage = errorJson.error || errorJson.message
            } else {
              errorMessage = errorText
            }
          } catch {
            // If not JSON, use the text as-is
            errorMessage = errorText
          }
        }
        throw new Error(errorMessage)
      }

      // Refresh config after save
      await fetchConfig()
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to save configuration')
    }
  }

  const openEditModal = (
    title: string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    data: any,
    fields: FieldConfig[],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    callback: (data: any) => Promise<void>,
    mode: 'edit' | 'add' = 'edit'
  ) => {
    setEditModalTitle(title)
    setEditModalData(data)
    setEditModalFields(fields)
    setEditModalMode(mode)
    setEditModalCallback(() => callback)
    setEditModalOpen(true)
  }

  const closeEditModal = () => {
    setEditModalOpen(false)
    setEditModalData(null)
    setEditModalFields([])
    setEditModalCallback(null)
  }

  const handleRefresh = () => {
    fetchConfig()
    fetchRouterDefaults()
  }

  // ============================================================================
  // HELPER FUNCTIONS - Normalize data access across config formats
  // ============================================================================

  // Helper: Check if using Python CLI format
  const isPythonCLI = configFormat === 'python-cli'

  // Get models - from providers.models (Python CLI) or model_config (legacy)
  interface NormalizedModel {
    name: string
    reasoning_family?: string
    endpoints: Array<{ name: string; weight: number; endpoint: string; protocol: string }>
    access_key?: string
    pricing?: {
      currency?: string
      prompt_per_1m?: number
      completion_per_1m?: number
    }
    pii_policy?: {
      allow_by_default?: boolean
      pii_types_allowed?: string[]
    }
  }

  const getModels = (): NormalizedModel[] => {
    if (isPythonCLI && config?.providers?.models) {
      return config.providers.models.map((m: NonNullable<ConfigData['providers']>['models'][number]) => ({
        name: m.name,
        reasoning_family: m.reasoning_family,
        endpoints: m.endpoints || [],
        access_key: m.access_key,
      }))
    }
    // Legacy format - convert model_config to array
    if (config?.model_config) {
      return (Object.entries(config.model_config) as [string, ModelConfigEntry][]).map(([name, cfg]) => ({
        name,
        reasoning_family: cfg.reasoning_family,
        endpoints: cfg.preferred_endpoints?.map((ep: string) => {
          const endpoint = config.vllm_endpoints?.find((e: VLLMEndpoint) => e.name === ep)
          return endpoint ? {
            name: ep,
            weight: endpoint.weight || 1,
            endpoint: `${endpoint.address}:${endpoint.port}`,
            protocol: 'http',
          } : null
        }).filter((e): e is NonNullable<typeof e> => e !== null) || [],
        access_key: undefined,
        pricing: cfg.pricing,
        pii_policy: cfg.pii_policy,
      }))
    }
    return []
  }

  // Get endpoints - from providers.models[].endpoints (Python CLI) or vllm_endpoints (legacy)
  interface NormalizedEndpoint {
    name: string
    endpoint?: string
    address?: string
    port?: number
    protocol?: string
    weight: number
    usedByModels?: string[]
  }

  const getEndpoints = (): NormalizedEndpoint[] => {
    if (isPythonCLI && config?.providers?.models) {
      // Flatten all endpoints from all models
      const endpointMap = new Map<string, NormalizedEndpoint>()
      config.providers.models.forEach((model: NonNullable<ConfigData['providers']>['models'][number]) => {
        model.endpoints?.forEach((ep: NonNullable<ConfigData['providers']>['models'][number]['endpoints'][number]) => {
          if (!endpointMap.has(ep.name)) {
            endpointMap.set(ep.name, {
              name: ep.name,
              endpoint: ep.endpoint,
              protocol: ep.protocol,
              weight: ep.weight,
              usedByModels: [model.name],
            })
          } else {
            const existing = endpointMap.get(ep.name)!
            existing.usedByModels = [...(existing.usedByModels || []), model.name]
          }
        })
      })
      return Array.from(endpointMap.values())
    }
    // Legacy format
    return (config?.vllm_endpoints || []).map((ep: VLLMEndpoint) => ({
      name: ep.name,
      address: ep.address,
      port: ep.port,
      weight: ep.weight,
    }))
  }

  // Get domains/categories - from signals.domains (Python CLI) or categories (legacy)
  const getDomains = (): Array<{ name: string; description: string; mmlu_categories?: string[] }> => {
    if (isPythonCLI && config?.signals?.domains) {
      return config.signals.domains
    }
    return config?.categories?.map((c: Category) => ({
      name: c.name,
      description: c.system_prompt || '',
      mmlu_categories: [] as string[],
    })) || []
  }

  // Get decisions/routes - from decisions (Python CLI) or categories with model_scores (legacy)
  const getDecisions = () => {
    if (isPythonCLI && config?.decisions) {
      return config.decisions
    }
    // Legacy format doesn't have decisions in the same way
    return []
  }

  // Get default model
  const getDefaultModel = (): string => {
    if (isPythonCLI) {
      return config?.providers?.default_model || ''
    }
    return config?.default_model || ''
  }

  // Get reasoning families
  const getReasoningFamilies = (): Record<string, ReasoningFamily> => {
    if (isPythonCLI) {
      return config?.providers?.reasoning_families || {}
    }
    return config?.reasoning_families || {}
  }

  // Helper: Normalize model_scores from object to array (Legacy format uses object)
  // Legacy: { "gpt-4": 0.9, "llama": 0.7 } → [{ model: "gpt-4", score: 0.9 }, ...]
  const normalizeModelScores = (modelScores: ModelScore[] | Record<string, number> | undefined): ModelScore[] => {
    if (!modelScores) return []
    // Already an array
    if (Array.isArray(modelScores)) return modelScores
    // Object format (Legacy) - convert to array
    return Object.entries(modelScores).map(([model, score]) => ({
      model,
      score: typeof score === 'number' ? score : 0,
      use_reasoning: false,
    }))
  }

  // ============================================================================
  // 1. MODELS SECTION
  // ============================================================================

  // Helper: Add endpoint (handles both formats)
  const handleAddEndpoint = (data: { name: string; endpoint: string; weight: number; protocol: string }) => {
    const newConfig = { ...config }
    if (isPythonCLI) {
      // Python CLI format: Add endpoint to the first model (or create default model)
      if (!newConfig.providers) {
        newConfig.providers = { models: [], default_model: '' }
      }
      if (!newConfig.providers.models || newConfig.providers.models.length === 0) {
        // Create a default model to hold this endpoint
        newConfig.providers.models = [{
          name: 'default-model',
          endpoints: [],
        }]
        newConfig.providers.default_model = 'default-model'
      }
      // Add to first model's endpoints
      const firstModel = newConfig.providers.models[0]
      firstModel.endpoints = [...(firstModel.endpoints || []), {
        name: data.name,
        endpoint: data.endpoint,
        weight: data.weight || 1,
        protocol: (data.protocol || 'http') as 'http' | 'https',
      }]
    } else {
      // Legacy format
      const [address, portStr] = (data.endpoint || '').split(':')
      newConfig.vllm_endpoints = [...(newConfig.vllm_endpoints || []), {
        name: data.name,
        address: address || '',
        port: parseInt(portStr) || 8000,
        weight: data.weight || 1,
        health_check_path: '/health',
      }]
    }
    return saveConfig(newConfig)
  }

  // Helper: Edit endpoint (handles both formats)
  const handleEditEndpoint = (oldName: string, data: { name: string; endpoint?: string; address?: string; port?: number; weight: number; protocol?: string }) => {
    const newConfig = { ...config }
    if (isPythonCLI && newConfig.providers?.models) {
      // Python CLI format: Find and update endpoint across all models
      newConfig.providers = { ...newConfig.providers }
      type ModelType = NonNullable<ConfigData['providers']>['models'][number]
      type EndpointType = ModelType['endpoints'][number]
      newConfig.providers.models = newConfig.providers.models.map((model: ModelType) => ({
        ...model,
        endpoints: model.endpoints?.map((ep: EndpointType) =>
          ep.name === oldName ? {
            name: data.name,
            endpoint: data.endpoint || ep.endpoint,
            weight: data.weight,
            protocol: (data.protocol || ep.protocol || 'http') as 'http' | 'https',
          } : ep
        ) || [],
      }))
    } else if (newConfig.vllm_endpoints) {
      // Legacy format
      newConfig.vllm_endpoints = newConfig.vllm_endpoints.map((ep: VLLMEndpoint) =>
        ep.name === oldName ? {
          ...ep,
          name: data.name,
          address: data.address || ep.address,
          port: data.port || ep.port,
          weight: data.weight,
        } : ep
      )
    }
    return saveConfig(newConfig)
  }

  // Helper: Delete endpoint (handles both formats)
  const handleDeleteEndpoint = (endpointName: string) => {
    const newConfig = { ...config }
    if (isPythonCLI && newConfig.providers?.models) {
      // Python CLI format: Remove endpoint from all models
      newConfig.providers = { ...newConfig.providers }
      type ModelType = NonNullable<ConfigData['providers']>['models'][number]
      type EndpointType = ModelType['endpoints'][number]
      newConfig.providers.models = newConfig.providers.models.map((model: ModelType) => ({
        ...model,
        endpoints: model.endpoints?.filter((ep: EndpointType) => ep.name !== endpointName) || [],
      }))
    } else if (newConfig.vllm_endpoints) {
      // Legacy format
      newConfig.vllm_endpoints = newConfig.vllm_endpoints.filter((ep: VLLMEndpoint) => ep.name !== endpointName)
    }
    return saveConfig(newConfig)
  }

  const renderUserDefinedEndpoints = () => {
    const endpoints = getEndpoints()
    const endpointCount = endpoints.length

    return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🔌</span>
        <h3 className={styles.sectionTitle}>User Defined Endpoints</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span className={styles.badge}>{endpointCount} endpoints</span>
          <button
            className={styles.addButton}
            onClick={() => {
              openEditModal(
                'Add New Endpoint',
                { name: '', endpoint: 'localhost:8000', weight: 1, protocol: 'http' },
                [
                  { name: 'name', label: 'Endpoint Name', type: 'text', required: true, placeholder: 'e.g., vllm-endpoint-1' },
                  { name: 'endpoint', label: 'Endpoint Address', type: 'text', required: true, placeholder: 'e.g., localhost:8000 or api.example.com' },
                  { name: 'protocol', label: 'Protocol', type: 'select', options: ['http', 'https'] },
                  { name: 'weight', label: 'Weight', type: 'number', placeholder: '1', description: 'Load balancing weight for this endpoint' }
                ],
                async (data) => {
                  await handleAddEndpoint(data)
                },
                'add'
              )
            }}
          >
            ➕ Add Endpoint
          </button>
        </div>
      </div>
      <div className={styles.sectionContent}>
        {endpoints.length > 0 ? (
          endpoints.map((endpoint, index) => (
            <div key={index} className={styles.endpointCard}>
              <div className={styles.endpointHeader}>
                <span className={styles.endpointName}>{endpoint.name}</span>
                <div className={styles.cardActions}>
                  <button
                    className={styles.editButton}
                    onClick={() => {
                      const endpointValue = endpoint.endpoint || (endpoint.address && endpoint.port ? `${endpoint.address}:${endpoint.port}` : '')
                      openEditModal(
                        `Edit Endpoint: ${endpoint.name}`,
                        { ...endpoint, endpoint: endpointValue },
                        [
                          { name: 'name', label: 'Endpoint Name', type: 'text', required: true },
                          { name: 'endpoint', label: 'Endpoint Address', type: 'text', required: true },
                          { name: 'protocol', label: 'Protocol', type: 'select', options: ['http', 'https'] },
                          { name: 'weight', label: 'Weight', type: 'number', description: 'Load balancing weight for this endpoint' }
                        ],
                        async (data) => {
                          await handleEditEndpoint(endpoint.name, data)
                        },
                        'edit'
                      )
                    }}
                  >
                    ✏️
                  </button>
                  <button
                    className={styles.deleteButton}
                    onClick={() => {
                      // Check which models use this endpoint
                      const usedByModels = endpoint.usedByModels || []

                      let confirmMessage = `Delete endpoint "${endpoint.name}"?`
                      if (usedByModels.length > 0) {
                        const modelList = usedByModels.length === 1 
                          ? `model "${usedByModels[0]}"` 
                          : `${usedByModels.length} models (${usedByModels.join(', ')})`
                        confirmMessage += `\n\n⚠️ This endpoint is currently used by ${modelList}.\nIt will be removed from all models.`
                      }

                      if (confirm(confirmMessage)) {
                        handleDeleteEndpoint(endpoint.name)
                      }
                    }}
                  >
                    🗑️
                  </button>
                </div>
              </div>
              <div className={styles.endpointDetails}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>🌐 Address</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span className={styles.configValue}>
                      {(() => {
                        const addr = endpoint.endpoint || (endpoint.address && endpoint.port ? `${endpoint.address}:${endpoint.port}` : '')
                        return visibleAddresses.has(index) ? addr : maskAddress(addr)
                      })()}
                    </span>
                    <button
                      className={styles.toggleVisibilityButton}
                      onClick={() => {
                        const newVisible = new Set(visibleAddresses)
                        if (newVisible.has(index)) {
                          newVisible.delete(index)
                        } else {
                          newVisible.add(index)
                        }
                        setVisibleAddresses(newVisible)
                      }}
                      title={visibleAddresses.has(index) ? 'Hide address' : 'Show address'}
                    >
                      {visibleAddresses.has(index) ? '👁️' : '👁️‍🗨️'}
                    </button>
                  </div>
                </div>
                {endpoint.protocol && (
                <div className={styles.configRow}>
                    <span className={styles.configLabel}>🔒 Protocol</span>
                    <span className={styles.configValue}>{endpoint.protocol.toUpperCase()}</span>
                </div>
                )}
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>⚖️ Weight</span>
                  <span className={styles.configValue}>{endpoint.weight}</span>
                </div>
                {endpoint.usedByModels && endpoint.usedByModels.length > 0 && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>🔗 Used By</span>
                    <span className={styles.configValue}>{endpoint.usedByModels.join(', ')}</span>
                  </div>
                )}
              </div>
            </div>
          ))
        ) : (
          <div className={styles.emptyState}>No endpoints configured</div>
        )}
      </div>
    </div>
  )}

  // Helper: Add model (handles both formats)
  const handleAddModel = async (data: {
    model_name: string
    reasoning_family?: string
    access_key?: string
    preferred_endpoints?: string[]
  }) => {
    const newConfig = { ...config }
    if (isPythonCLI) {
      // Python CLI format: Add to providers.models
      if (!newConfig.providers) {
        newConfig.providers = { models: [], default_model: '' }
      }
      const existingEndpoints = getEndpoints()
      const modelEndpoints = (data.preferred_endpoints || []).map(epName => {
        const existing = existingEndpoints.find(e => e.name === epName)
        return {
          name: epName,
          endpoint: existing?.endpoint || 'localhost:8000',
          weight: existing?.weight || 1,
          protocol: (existing?.protocol || 'http') as 'http' | 'https',
        }
      })
      newConfig.providers.models = [...(newConfig.providers.models || []), {
        name: data.model_name,
        reasoning_family: data.reasoning_family,
        endpoints: modelEndpoints,
        access_key: data.access_key,
      }]
      // Set as default if first model
      if (!newConfig.providers.default_model) {
        newConfig.providers.default_model = data.model_name
      }
    } else {
      // Legacy format
      if (!newConfig.model_config) {
        newConfig.model_config = {}
      }
      newConfig.model_config[data.model_name] = {
        reasoning_family: data.reasoning_family,
        preferred_endpoints: data.preferred_endpoints,
      }
    }
    await saveConfig(newConfig)
  }

  // Helper: Delete model (handles both formats)
  const handleDeleteModel = async (modelName: string) => {
    const newConfig = { ...config }
    if (isPythonCLI && newConfig.providers?.models) {
      newConfig.providers = { ...newConfig.providers }
      type ModelType = NonNullable<ConfigData['providers']>['models'][number]
      newConfig.providers.models = newConfig.providers.models.filter((m: ModelType) => m.name !== modelName)
      // Update default model if deleted
      if (newConfig.providers.default_model === modelName) {
        newConfig.providers.default_model = newConfig.providers.models[0]?.name || ''
      }
    } else if (newConfig.model_config) {
      delete newConfig.model_config[modelName]
    }
    await saveConfig(newConfig)
  }

  const renderUserDefinedModels = () => {
    const models = getModels()
    const modelCount = models.length
    const availableEndpoints = getEndpoints()
    const reasoningFamiliesObj = getReasoningFamilies()
    const reasoningFamilyNames = Object.keys(reasoningFamiliesObj)

    return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🤖</span>
        <h3 className={styles.sectionTitle}>User Defined Models</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span className={styles.badge}>{modelCount} models</span>
          <button
            className={styles.addButton}
            onClick={() => {
              // Get available reasoning families
              const reasoningFamilies = reasoningFamilyNames

              // Get available endpoints
              const endpoints = availableEndpoints.map(ep => ep.name)

              // PII types
              const piiTypes = [
                'AGE', 'CREDIT_CARD', 'DATE_TIME', 'DOMAIN_NAME',
                'EMAIL_ADDRESS', 'GPE', 'IBAN_CODE', 'IP_ADDRESS',
                'NO_PII', 'NRP', 'ORGANIZATION', 'PERSON',
                'PHONE_NUMBER', 'STREET_ADDRESS', 'US_DRIVER_LICENSE',
                'US_SSN', 'ZIP_CODE'
              ]

              openEditModal(
                'Add New Model',
                {
                  reasoning_family: reasoningFamilies[0] || '',
                  preferred_endpoints: [],
                  currency: 'USD',
                  prompt_per_1m: 0,
                  completion_per_1m: 0,
                  pii_allow_by_default: true,
                  pii_types_allowed: []
                },
                [
                  {
                    name: 'model_name',
                    label: 'Model Name',
                    type: 'text',
                    required: true,
                    placeholder: 'e.g., openai/gpt-oss-20b',
                    description: 'Unique identifier for the model'
                  },
                  {
                    name: 'reasoning_family',
                    label: 'Reasoning Family',
                    type: 'select',
                    options: reasoningFamilies,
                    description: 'Select from configured reasoning families'
                  },
                  {
                    name: 'preferred_endpoints',
                    label: 'Preferred Endpoints',
                    type: 'multiselect',
                    options: endpoints,
                    description: 'Select one or more endpoints for this model'
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
                  },
                  {
                    name: 'pii_allow_by_default',
                    label: 'PII Policy: Allow by Default',
                    type: 'boolean',
                    description: 'If enabled, all PII types are allowed unless specified below'
                  },
                  {
                    name: 'pii_types_allowed',
                    label: 'PII Types Allowed/Blocked',
                    type: 'multiselect',
                    options: piiTypes,
                    description: 'If "Allow by Default" is ON: select types to BLOCK. If OFF: select types to ALLOW'
                  }
                ],
                async (data) => {
                  await handleAddModel({
                    model_name: data.model_name,
                    reasoning_family: data.reasoning_family,
                    preferred_endpoints: data.preferred_endpoints,
                    access_key: data.access_key,
                  })
                },
                'add'
              )
            }}
          >
            ➕ Add Model
          </button>
        </div>
      </div>
      <div className={styles.sectionContent}>
        {models.length > 0 ? (
          <div className={styles.modelConfigGrid}>
            {models.map((model) => (
              <div key={model.name} className={styles.modelConfigCard}>
                <div className={styles.modelConfigHeader}>
                  <span className={styles.modelConfigName}>{model.name}</span>
                  {model.name === getDefaultModel() && (
                    <span className={styles.defaultBadge}>Default</span>
                  )}
                  <div className={styles.cardActions}>
                    <button
                      className={styles.editButton}
                      onClick={() => {
                        openEditModal(
                          `Edit Model: ${model.name}`,
                          {
                            reasoning_family: model.reasoning_family || '',
                            preferred_endpoints: model.endpoints?.map(e => e.name) || [],
                            access_key: model.access_key || '',
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
                              name: 'preferred_endpoints',
                              label: 'Endpoints',
                              type: 'multiselect',
                              options: availableEndpoints.map(e => e.name),
                              description: 'Select endpoints for this model'
                            },
                            {
                              name: 'access_key',
                              label: 'Access Key',
                              type: 'text',
                              placeholder: 'API key for this model',
                              description: 'Optional: API key for authentication'
                            }
                          ],
                          async (data) => {
                            const newConfig = { ...config }
                            if (isPythonCLI && newConfig.providers?.models) {
                              // Python CLI format: Update model in providers.models
                              newConfig.providers = { ...newConfig.providers }
                              const existingEps = getEndpoints()
                              type ModelType = NonNullable<ConfigData['providers']>['models'][number]
                              newConfig.providers.models = newConfig.providers.models.map((m: ModelType) => 
                                m.name === model.name ? {
                                  ...m,
                                  reasoning_family: data.reasoning_family,
                                  access_key: data.access_key,
                                  endpoints: (data.preferred_endpoints || []).map((epName: string) => {
                                    const existing = existingEps.find(e => e.name === epName)
                                    return {
                                      name: epName,
                                      endpoint: existing?.endpoint || 'localhost:8000',
                                      weight: existing?.weight || 1,
                                      protocol: (existing?.protocol || 'http') as 'http' | 'https',
                                }
                                  }),
                                } : m
                              )
                            } else if (newConfig.model_config) {
                              // Legacy format
                              newConfig.model_config[model.name] = {
                                ...newConfig.model_config[model.name],
                                reasoning_family: data.reasoning_family,
                                preferred_endpoints: data.preferred_endpoints,
                              }
                            }
                            await saveConfig(newConfig)
                          },
                          'edit'
                        )
                      }}
                    >
                      ✏️
                    </button>
                    <button
                      className={styles.deleteButton}
                      onClick={() => {
                        if (confirm(`Are you sure you want to delete model "${model.name}"?`)) {
                          handleDeleteModel(model.name)
                        }
                      }}
                    >
                      🗑️
                    </button>
                  </div>
                </div>
                <div className={styles.modelConfigBody}>
                  {model.reasoning_family && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>🧠 Reasoning Family</span>
                      <span className={`${styles.badge} ${styles.badgeInfo}`}>
                        {model.reasoning_family}
                      </span>
                    </div>
                  )}
                  {model.endpoints && model.endpoints.length > 0 && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>🔌 Endpoints</span>
                      <div className={styles.endpointTags}>
                        {model.endpoints.map((endpoint, idx) => (
                          <span key={idx} className={styles.endpointTag}>{endpoint.name}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {model.access_key && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>🔑 Access Key</span>
                      <span className={styles.configValue}>••••••••</span>
                    </div>
                  )}
                  {model.pricing && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>💰 Pricing</span>
                      <div className={styles.pricingContainer}>
                        <div className={styles.pricingItem}>
                          <span className={styles.pricingLabel}>Prompt</span>
                          <span className={styles.pricingValue}>
                            {model.pricing.prompt_per_1m?.toFixed(2) || '0.00'}
                          </span>
                          <span className={styles.pricingUnit}>
                            {model.pricing.currency || 'USD'}/1M
                          </span>
                        </div>
                        <div className={styles.pricingDivider}>|</div>
                        <div className={styles.pricingItem}>
                          <span className={styles.pricingLabel}>Completion</span>
                          <span className={styles.pricingValue}>
                            {model.pricing.completion_per_1m?.toFixed(2) || '0.00'}
                          </span>
                          <span className={styles.pricingUnit}>
                            {model.pricing.currency || 'USD'}/1M
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  {model.pii_policy && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>🔒 PII Policy</span>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                        <span className={`${styles.statusBadge} ${model.pii_policy.allow_by_default ? styles.statusActive : styles.statusInactive}`}>
                          {model.pii_policy.allow_by_default ? 'Allow by default' : 'Block by default'}
                        </span>
                        {model.pii_policy.pii_types_allowed && model.pii_policy.pii_types_allowed.length > 0 && (
                          <div className={styles.piiTypesTags}>
                            {model.pii_policy.pii_types_allowed.map((type: string, idx: number) => (
                              <span key={idx} className={styles.piiTypeTag}>{type}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className={styles.emptyState}>No model configurations defined</div>
        )}
      </div>
    </div>
  )}

  // ============================================================================
  // 2. PROMPT GUARD SECTION
  // ============================================================================

  const renderPIIModernBERT = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🔒</span>
        <h3 className={styles.sectionTitle}>PII Detection (ModernBERT)</h3>
        {config?.classifier?.pii_model && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit PII Detection Configuration',
                config?.classifier?.pii_model || {},
                [
                  {
                    name: 'model_id',
                    label: 'Model ID',
                    type: 'text',
                    required: true,
                    placeholder: 'e.g., answerdotai/ModernBERT-base',
                    description: 'HuggingFace model ID for PII detection'
                  },
                  {
                    name: 'threshold',
                    label: 'Detection Threshold',
                    type: 'percentage',
                    required: true,
                    placeholder: '50',
                    description: 'Confidence threshold for PII detection (0-100%)',
                    step: 1
                  },
                  {
                    name: 'use_cpu',
                    label: 'Use CPU',
                    type: 'boolean',
                    description: 'Use CPU instead of GPU for inference'
                  },
                  {
                    name: 'use_modernbert',
                    label: 'Use ModernBERT',
                    type: 'boolean',
                    description: 'Enable ModernBERT-based PII detection'
                  },
                  {
                    name: 'pii_mapping_path',
                    label: 'PII Mapping Path',
                    type: 'text',
                    placeholder: 'config/pii_mapping.json',
                    description: 'Path to PII entity mapping configuration'
                  }
                ],
                async (data) => {
                  const newConfig = { ...config }
                  if (!newConfig.classifier) newConfig.classifier = {}
                  newConfig.classifier.pii_model = data
                  await saveConfig(newConfig)
                }
              )
            }}
          >
            ✏️ Edit
          </button>
        )}
      </div>
      <div className={styles.sectionContent}>
        {config?.classifier?.pii_model ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>PII Classifier Model</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                {config.classifier.pii_model.use_cpu ? '💻 CPU' : '🎮 GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
                <span className={styles.configValue}>{config.classifier.pii_model.model_id}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
                <span className={styles.configValue}>{formatThreshold(config.classifier.pii_model.threshold)}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>ModernBERT</span>
                <span className={`${styles.statusBadge} ${config.classifier.pii_model.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
                  {config.classifier.pii_model.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
                </span>
              </div>
              {config.classifier.pii_model.pii_mapping_path && (
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Mapping Path</span>
                  <span className={styles.configValue}>{config.classifier.pii_model.pii_mapping_path}</span>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className={styles.emptyState}>PII detection not configured</div>
        )}
      </div>
    </div>
  )

  const renderJailbreakModernBERT = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🛡️</span>
        <h3 className={styles.sectionTitle}>Jailbreak Detection (ModernBERT)</h3>
        {config?.prompt_guard && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Jailbreak Detection Configuration',
                config?.prompt_guard || {},
                [
                  {
                    name: 'enabled',
                    label: 'Enable Jailbreak Detection',
                    type: 'boolean',
                    description: 'Enable or disable jailbreak detection'
                  },
                  {
                    name: 'model_id',
                    label: 'Model ID',
                    type: 'text',
                    required: true,
                    placeholder: 'e.g., answerdotai/ModernBERT-base',
                    description: 'HuggingFace model ID for jailbreak detection'
                  },
                  {
                    name: 'threshold',
                    label: 'Detection Threshold',
                    type: 'percentage',
                    required: true,
                    placeholder: '50',
                    description: 'Confidence threshold for jailbreak detection (0-100%)',
                    step: 1
                  },
                  {
                    name: 'use_cpu',
                    label: 'Use CPU',
                    type: 'boolean',
                    description: 'Use CPU instead of GPU for inference'
                  },
                  {
                    name: 'use_modernbert',
                    label: 'Use ModernBERT',
                    type: 'boolean',
                    description: 'Enable ModernBERT-based jailbreak detection'
                  },
                  {
                    name: 'jailbreak_mapping_path',
                    label: 'Jailbreak Mapping Path',
                    type: 'text',
                    placeholder: 'config/jailbreak_mapping.json',
                    description: 'Path to jailbreak pattern mapping configuration'
                  }
                ],
                async (data) => {
                  const newConfig = { ...config }
                  newConfig.prompt_guard = data
                  await saveConfig(newConfig)
                }
              )
            }}
          >
            ✏️ Edit
          </button>
        )}
      </div>
      <div className={styles.sectionContent}>
        {config?.prompt_guard ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>Jailbreak Protection</span>
              <span className={`${styles.statusBadge} ${config.prompt_guard.enabled ? styles.statusActive : styles.statusInactive}`}>
                {config.prompt_guard.enabled ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            {config.prompt_guard.enabled && (
              <div className={styles.modelCardBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Model ID</span>
                  <span className={styles.configValue}>{config.prompt_guard.model_id}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(config.prompt_guard.threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Use CPU</span>
                  <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                    {config.prompt_guard.use_cpu ? '💻 CPU' : '🎮 GPU'}
                  </span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>ModernBERT</span>
                  <span className={`${styles.statusBadge} ${config.prompt_guard.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
                    {config.prompt_guard.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                {config.prompt_guard.jailbreak_mapping_path && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Mapping Path</span>
                    <span className={styles.configValue}>{config.prompt_guard.jailbreak_mapping_path}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className={styles.emptyState}>Jailbreak detection not configured</div>
        )}
      </div>
    </div>
  )

  // ============================================================================
  // 3. SIMILARITY CACHE SECTION
  // ============================================================================

  const renderSimilarityBERT = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>⚡</span>
        <h3 className={styles.sectionTitle}>Similarity BERT Configuration</h3>
        {config?.bert_model && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Similarity BERT Configuration',
                config?.bert_model || {},
                [
                  {
                    name: 'model_id',
                    label: 'Model ID',
                    type: 'text',
                    required: true,
                    placeholder: 'e.g., sentence-transformers/all-MiniLM-L6-v2',
                    description: 'HuggingFace model ID for semantic similarity'
                  },
                  {
                    name: 'threshold',
                    label: 'Similarity Threshold',
                    type: 'percentage',
                    required: true,
                    placeholder: '80',
                    description: 'Minimum similarity score for cache hits (0-100%)',
                    step: 1
                  },
                  {
                    name: 'use_cpu',
                    label: 'Use CPU',
                    type: 'boolean',
                    description: 'Use CPU instead of GPU for inference'
                  }
                ],
                async (data) => {
                  const newConfig = { ...config }
                  newConfig.bert_model = data
                  await saveConfig(newConfig)
                }
              )
            }}
          >
            ✏️ Edit
          </button>
        )}
      </div>
      <div className={styles.sectionContent}>
        {config?.bert_model ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>BERT Model (Semantic Similarity)</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                {config.bert_model.use_cpu ? '💻 CPU' : '🎮 GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
                <span className={styles.configValue}>{config.bert_model.model_id}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
                <span className={styles.configValue}>{formatThreshold(config.bert_model.threshold)}</span>
              </div>
            </div>
          </div>
        ) : (
          <div className={styles.emptyState}>BERT model not configured</div>
        )}

        {config?.semantic_cache && (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Semantic Cache</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span className={`${styles.statusBadge} ${config.semantic_cache.enabled ? styles.statusActive : styles.statusInactive}`}>
                  {config.semantic_cache.enabled ? '✓ Enabled' : '✗ Disabled'}
                </span>
                <button
                  className={styles.sectionEditButton}
                  onClick={() => {
                    openEditModal(
                      'Edit Semantic Cache Configuration',
                      config?.semantic_cache || {},
                      [
                        {
                          name: 'enabled',
                          label: 'Enable Semantic Cache',
                          type: 'boolean',
                          description: 'Enable or disable semantic caching'
                        },
                        {
                          name: 'backend_type',
                          label: 'Backend Type',
                          type: 'select',
                          options: ['memory', 'redis', 'memcached'],
                          description: 'Cache backend storage type'
                        },
                        {
                          name: 'similarity_threshold',
                          label: 'Similarity Threshold',
                          type: 'percentage',
                          required: true,
                          placeholder: '90',
                          description: 'Minimum similarity score for cache hits (0-100%)',
                          step: 1
                        },
                        {
                          name: 'max_entries',
                          label: 'Max Entries',
                          type: 'number',
                          placeholder: '10000',
                          description: 'Maximum number of cached entries'
                        },
                        {
                          name: 'ttl_seconds',
                          label: 'TTL (seconds)',
                          type: 'number',
                          placeholder: '3600',
                          description: 'Time-to-live for cached entries'
                        },
                        {
                          name: 'eviction_policy',
                          label: 'Eviction Policy',
                          type: 'select',
                          options: ['lru', 'lfu', 'fifo'],
                          description: 'Cache eviction policy when max entries reached'
                        }
                      ],
                      async (data) => {
                        const newConfig = { ...config }
                        newConfig.semantic_cache = data
                        await saveConfig(newConfig)
                      }
                    )
                  }}
                >
                  ✏️ Edit
                </button>
              </div>
            </div>
            {config.semantic_cache.enabled && (
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Backend Type</span>
                  <span className={styles.configValue}>{config.semantic_cache.backend_type || 'memory'}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Similarity Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(config.semantic_cache.similarity_threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Max Entries</span>
                  <span className={styles.configValue}>{config.semantic_cache.max_entries}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>TTL</span>
                  <span className={styles.configValue}>{config.semantic_cache.ttl_seconds}s</span>
                </div>
                {config.semantic_cache.eviction_policy && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Eviction Policy</span>
                    <span className={styles.configValue}>{config.semantic_cache.eviction_policy}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )

  // ============================================================================
  // 4. INTELLIGENT ROUTING SECTION
  // ============================================================================

  const renderClassifyBERT = () => {
    const hasInTree = config?.classifier?.category_model
    const hasOutTree = config?.classifier?.mcp_category_model?.enabled

    return (
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionIcon}>🎯</span>
          <h3 className={styles.sectionTitle}>Classify BERT Model</h3>
        </div>
        <div className={styles.sectionContent}>
          {/* In-tree Classifier */}
          {hasInTree && config?.classifier?.category_model && (
            <div className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelCardTitle}>In-tree Category Classifier</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                    {config.classifier.category_model.use_cpu ? '💻 CPU' : '🎮 GPU'}
                  </span>
                  <button
                    className={styles.editButton}
                    onClick={() => {
                      openEditModal(
                        'Edit In-tree Category Classifier',
                        config?.classifier?.category_model || {},
                        [
                          {
                            name: 'model_id',
                            label: 'Model ID',
                            type: 'text',
                            required: true,
                            placeholder: 'e.g., answerdotai/ModernBERT-base',
                            description: 'HuggingFace model ID for category classification'
                          },
                          {
                            name: 'threshold',
                            label: 'Classification Threshold',
                            type: 'percentage',
                            required: true,
                            placeholder: '70',
                            description: 'Confidence threshold for category classification (0-100%)',
                            step: 1
                          },
                          {
                            name: 'use_cpu',
                            label: 'Use CPU',
                            type: 'boolean',
                            description: 'Use CPU instead of GPU for inference'
                          },
                          {
                            name: 'use_modernbert',
                            label: 'Use ModernBERT',
                            type: 'boolean',
                            description: 'Enable ModernBERT-based classification'
                          },
                          {
                            name: 'category_mapping_path',
                            label: 'Category Mapping Path',
                            type: 'text',
                            placeholder: 'config/category_mapping.json',
                            description: 'Path to category mapping configuration'
                          }
                        ],
                        async (data) => {
                          const newConfig = { ...config }
                          if (!newConfig.classifier) newConfig.classifier = {}
                          newConfig.classifier.category_model = data
                          await saveConfig(newConfig)
                        }
                      )
                    }}
                  >
                    ✏️
                  </button>
                </div>
              </div>
              <div className={styles.modelCardBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Type</span>
                  <span className={`${styles.badge} ${styles.badgeInfo}`}>Built-in ModernBERT</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Model ID</span>
                  <span className={styles.configValue}>{config.classifier.category_model.model_id}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(config.classifier.category_model.threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>ModernBERT</span>
                  <span className={`${styles.statusBadge} ${config.classifier.category_model.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
                    {config.classifier.category_model.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                {config.classifier.category_model.category_mapping_path && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Mapping Path</span>
                    <span className={styles.configValue}>{config.classifier.category_model.category_mapping_path}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Out-tree Classifier (MCP) */}
          {hasOutTree && config?.classifier?.mcp_category_model && (
            <div className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelCardTitle}>Out-tree Category Classifier (MCP)</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span className={`${styles.statusBadge} ${styles.statusActive}`}>✓ Enabled</span>
                  <button
                    className={styles.editButton}
                    onClick={() => {
                      openEditModal(
                        'Edit Out-tree MCP Category Classifier',
                        config?.classifier?.mcp_category_model || {},
                        [
                          {
                            name: 'enabled',
                            label: 'Enable MCP Classifier',
                            type: 'boolean',
                            description: 'Enable or disable MCP-based classification'
                          },
                          {
                            name: 'transport_type',
                            label: 'Transport Type',
                            type: 'select',
                            options: ['stdio', 'http'],
                            required: true,
                            description: 'MCP transport protocol type'
                          },
                          {
                            name: 'command',
                            label: 'Command',
                            type: 'text',
                            placeholder: 'e.g., python mcp_server.py',
                            description: 'Command to start MCP server (for stdio transport)'
                          },
                          {
                            name: 'args',
                            label: 'Arguments (JSON)',
                            type: 'json',
                            placeholder: '["--port", "8080"]',
                            description: 'Command line arguments as JSON array'
                          },
                          {
                            name: 'env',
                            label: 'Environment Variables (JSON)',
                            type: 'json',
                            placeholder: '{"API_KEY": "xxx"}',
                            description: 'Environment variables as JSON object'
                          },
                          {
                            name: 'url',
                            label: 'URL',
                            type: 'text',
                            placeholder: 'http://localhost:8080',
                            description: 'MCP server URL (for http transport)'
                          },
                          {
                            name: 'tool_name',
                            label: 'Tool Name',
                            type: 'text',
                            placeholder: 'classify_category',
                            description: 'Name of the MCP tool to call'
                          },
                          {
                            name: 'threshold',
                            label: 'Classification Threshold',
                            type: 'percentage',
                            required: true,
                            placeholder: '70',
                            description: 'Confidence threshold for classification (0-100%)',
                            step: 1
                          },
                          {
                            name: 'timeout_seconds',
                            label: 'Timeout (seconds)',
                            type: 'number',
                            placeholder: '30',
                            description: 'Request timeout in seconds'
                          }
                        ],
                        async (data) => {
                          const newConfig = { ...config }
                          if (!newConfig.classifier) newConfig.classifier = {}
                          newConfig.classifier.mcp_category_model = data
                          await saveConfig(newConfig)
                        }
                      )
                    }}
                  >
                    ✏️
                  </button>
                </div>
              </div>
              <div className={styles.modelCardBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Type</span>
                  <span className={`${styles.badge} ${styles.badgeInfo}`}>MCP Protocol</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Transport Type</span>
                  <span className={styles.configValue}>{config.classifier.mcp_category_model.transport_type}</span>
                </div>
                {config.classifier.mcp_category_model.command && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Command</span>
                    <span className={styles.configValue}>{config.classifier.mcp_category_model.command}</span>
                  </div>
                )}
                {config.classifier.mcp_category_model.url && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>URL</span>
                    <span className={styles.configValue}>{config.classifier.mcp_category_model.url}</span>
                  </div>
                )}
                {config.classifier.mcp_category_model.tool_name && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Tool Name</span>
                    <span className={styles.configValue}>{config.classifier.mcp_category_model.tool_name}</span>
                  </div>
                )}
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(config.classifier.mcp_category_model.threshold)}</span>
                </div>
                {config.classifier.mcp_category_model.timeout_seconds && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Timeout</span>
                    <span className={styles.configValue}>{config.classifier.mcp_category_model.timeout_seconds}s</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {!hasInTree && !hasOutTree && (
            <div className={styles.emptyState}>No category classifier configured</div>
          )}
        </div>
      </div>
    )
  }

  const renderCategories = () => {
    // Get domains/categories from both formats
    const domains = getDomains()
    const decisions = getDecisions()
    const defaultModel = getDefaultModel()

    return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>📊</span>
        <h3 className={styles.sectionTitle}>{isPythonCLI ? 'Domains & Decisions' : 'Categories Configuration'}</h3>
        <span className={styles.badge}>{domains.length} {isPythonCLI ? 'domains' : 'categories'}</span>
      </div>
      <div className={styles.sectionContent}>
        {/* Core Settings at the top */}
        <div className={styles.coreSettingsInline}>
          <div className={styles.inlineConfigRow}>
            <span className={styles.inlineConfigLabel}>🎯 Default Model:</span>
            <span className={styles.inlineConfigValue}>{defaultModel || 'N/A'}</span>
          </div>
          {!isPythonCLI && (
          <div className={styles.inlineConfigRow}>
            <span className={styles.inlineConfigLabel}>⚡ Default Reasoning Effort:</span>
            <span className={`${styles.badge} ${styles[`badge${config?.default_reasoning_effort || 'medium'}`]}`}>
              {config?.default_reasoning_effort || 'medium'}
            </span>
          </div>
          )}
        </div>

        {/* Python CLI format - show domains and decisions separately */}
        {isPythonCLI ? (
          <>
            {/* Domains Section */}
            <h4 className={styles.subsectionTitle}>📁 Domains</h4>
            {domains.length > 0 ? (
              <div className={styles.categoryGridTwoColumn}>
                {domains.map((domain, index) => (
                  <div key={index} className={styles.categoryCard}>
                    <div className={styles.categoryHeader}>
                      <span className={styles.categoryName}>{domain.name}</span>
                      <button
                        className={styles.editButton}
                        onClick={() => {
                          openEditModal(
                            `Edit Domain: ${domain.name}`,
                            { description: domain.description || '' },
                            [
                              {
                                name: 'description',
                                label: 'Description',
                                type: 'textarea',
                                placeholder: 'Describe this domain...',
                                description: 'What types of queries belong to this domain'
                              }
                            ],
                            async (data) => {
                              const newConfig = { ...config }
                              if (newConfig.signals?.domains) {
                                newConfig.signals.domains[index] = {
                                  ...domain,
                                  description: data.description,
                                }
                              }
                              await saveConfig(newConfig)
                            }
                          )
                        }}
                      >
                        ✏️
                      </button>
                    </div>
                    {domain.description && (
                      <p className={styles.categoryDescription}>{domain.description}</p>
                    )}
                    {domain.mmlu_categories && domain.mmlu_categories.length > 0 && (
                      <div className={styles.tagsContainer}>
                        {domain.mmlu_categories.map((cat: string, idx: number) => (
                          <span key={idx} className={styles.tag}>{cat}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className={styles.emptyState}>No domains configured</div>
            )}

            {/* Decisions Section */}
            <h4 className={styles.subsectionTitle}>🔀 Decisions (Routing Rules)</h4>
            {decisions.length > 0 ? (
              <div className={styles.categoryGridTwoColumn}>
                {decisions.map((decision, index) => (
                  <div key={index} className={styles.categoryCard}>
                    <div className={styles.categoryHeader}>
                      <span className={styles.categoryName}>{decision.name}</span>
                      <span className={`${styles.badge} ${styles.badgeInfo}`}>Priority: {decision.priority}</span>
                    </div>
                    {decision.description && (
                      <p className={styles.categoryDescription}>{decision.description}</p>
                    )}
                    {decision.rules && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Rules</span>
                        <span className={styles.configValue}>
                          {decision.rules.conditions?.length || 0} conditions ({decision.rules.operator})
                        </span>
                      </div>
                    )}
                    {decision.modelRefs && decision.modelRefs.length > 0 && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Models</span>
                        <div className={styles.endpointTags}>
                          {decision.modelRefs.map((ref: { model: string; use_reasoning?: boolean }, idx: number) => (
                            <span key={idx} className={styles.endpointTag}>
                              {ref.model} {ref.use_reasoning && '⚡'}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className={styles.emptyState}>No decisions configured</div>
            )}
          </>
        ) : (
          /* Legacy format - categories with model scores */
          config?.categories && config.categories.length > 0 ? (
          <div className={styles.categoryGridTwoColumn}>
            {config.categories.map((category, index) => {
              // Normalize model_scores (handles both object and array formats)
              const normalizedScores = normalizeModelScores(category.model_scores)
              // Get reasoning info from best model (first model score)
              const bestModel = normalizedScores[0]
              const useReasoning = bestModel?.use_reasoning || false
              const reasoningEffort = bestModel?.reasoning_effort || 'medium'
              const reasoningDescription = bestModel?.reasoning_description || ''

              return (
              <div key={index} className={styles.categoryCard}>
                <div className={styles.categoryHeader}>
                  <span className={styles.categoryName}>{category.name}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    {useReasoning && (
                      <span className={`${styles.reasoningBadge} ${styles[`reasoning${reasoningEffort}`]}`}>
                        ⚡ {reasoningEffort}
                      </span>
                    )}
                    <button
                      className={styles.editButton}
                      onClick={() => {
                        openEditModal(
                          `Edit Category: ${category.name}`,
                          {
                            system_prompt: category.system_prompt || ''
                          },
                          [
                            {
                              name: 'system_prompt',
                              label: 'System Prompt',
                              type: 'textarea',
                              placeholder: 'Enter system prompt for this category...',
                              description: 'Instructions for the model when handling this category'
                            }
                          ],
                          async (data) => {
                            const newConfig = { ...config }
                            if (newConfig.categories) {
                              newConfig.categories[index] = {
                                ...category,
                                ...data
                              }
                            }
                            await saveConfig(newConfig)
                          }
                        )
                      }}
                    >
                      ✏️
                    </button>
                  </div>
                </div>

                {/* System Prompt */}
                {category.system_prompt && (
                  <div className={styles.systemPromptSection}>
                    <div className={styles.systemPromptLabel}>💬 System Prompt</div>
                    <div className={styles.systemPromptText}>{category.system_prompt}</div>
                  </div>
                )}

                {reasoningDescription && (
                  <p className={styles.categoryDescription}>{reasoningDescription}</p>
                )}

                <div className={styles.categoryModels}>
                  <div className={styles.categoryModelsHeader}>
                    <span>Model Scores</span>
                    <button
                      className={styles.addModelButton}
                      onClick={() => {
                        // Get available models from model_config
                        const availableModels = config?.model_config
                          ? Object.keys(config.model_config)
                          : []

                        openEditModal(
                          `Add Model to ${category.name}`,
                          {
                            model: availableModels[0] || '',
                            score: 0.5,
                            use_reasoning: false
                          },
                          [
                            {
                              name: 'model',
                              label: 'Model',
                              type: 'select',
                              options: availableModels,
                              required: true,
                              description: 'Select from configured models'
                            },
                            {
                              name: 'score',
                              label: 'Score',
                              type: 'number',
                              required: true,
                              placeholder: '0.5',
                              description: 'Model score (0-1)'
                            },
                            {
                              name: 'use_reasoning',
                              label: 'Use Reasoning',
                              type: 'boolean',
                              description: 'Enable reasoning for this model in this category'
                            }
                          ],
                          async (data) => {
                            const newConfig = { ...config }
                            if (newConfig.categories) {
                              const updatedCategory = { ...category }
                              // Convert to array format if needed (Legacy uses object)
                              const scores = normalizeModelScores(updatedCategory.model_scores)
                              scores.push(data)
                              updatedCategory.model_scores = scores
                              newConfig.categories[index] = updatedCategory
                            }
                            await saveConfig(newConfig)
                          },
                          'add'
                        )
                      }}
                    >
                      ➕
                    </button>
                  </div>
                  {normalizedScores.length > 0 ? (
                    normalizedScores.map((modelScore, modelIdx) => (
                      <div key={modelIdx} className={styles.modelScoreRow}>
                        <span className={styles.modelScoreName}>
                          {modelScore.model}
                          {modelScore.use_reasoning && <span className={styles.reasoningIcon}>🧠</span>}
                        </span>
                        <div className={styles.scoreBar}>
                          <div
                            className={styles.scoreBarFill}
                            style={{ width: `${(modelScore.score ?? 0) * 100}%` }}
                          ></div>
                          <span className={styles.scoreText}>{((modelScore.score ?? 0) * 100).toFixed(0)}%</span>
                        </div>
                        <div className={styles.modelScoreActions}>
                          <button
                            className={styles.editButton}
                            onClick={() => {
                              // Get available models from model_config
                              const availableModels = config?.model_config
                                ? Object.keys(config.model_config)
                                : []

                              openEditModal(
                                `Edit Model: ${modelScore.model}`,
                                { ...modelScore },
                                [
                                  {
                                    name: 'model',
                                    label: 'Model',
                                    type: 'select',
                                    options: availableModels,
                                    required: true,
                                    description: 'Select from configured models'
                                  },
                                  {
                                    name: 'score',
                                    label: 'Score',
                                    type: 'number',
                                    required: true,
                                    placeholder: '0.5',
                                    description: 'Model score (0-1)'
                                  },
                                  {
                                    name: 'use_reasoning',
                                    label: 'Use Reasoning',
                                    type: 'boolean',
                                    description: 'Enable reasoning for this model in this category'
                                  }
                                ],
                                async (data) => {
                                  // For legacy object format, we need to convert back
                                  const newConfig = { ...config }
                                  if (newConfig.categories) {
                                    const updatedCategory = { ...category }
                                    // Convert to array format for consistency
                                    const scores = normalizeModelScores(updatedCategory.model_scores)
                                    scores[modelIdx] = data
                                    updatedCategory.model_scores = scores
                                    newConfig.categories[index] = updatedCategory
                                  }
                                  await saveConfig(newConfig)
                                }
                              )
                            }}
                          >
                            ✏️
                          </button>
                          <button
                            className={styles.deleteButton}
                            onClick={() => {
                              if (confirm(`Remove model "${modelScore.model}" from this category?`)) {
                                const newConfig = { ...config }
                                if (newConfig.categories) {
                                  const updatedCategory = { ...category }
                                  // Convert to array format for consistency
                                  const scores = normalizeModelScores(updatedCategory.model_scores)
                                  scores.splice(modelIdx, 1)
                                  updatedCategory.model_scores = scores
                                  newConfig.categories[index] = updatedCategory
                                }
                                saveConfig(newConfig)
                              }
                            }}
                          >
                            🗑️
                          </button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className={styles.emptyModelScores}>No models configured for this category</div>
                  )}
                </div>
              </div>
            )})}
          </div>
        ) : (
          <div className={styles.emptyState}>No categories configured</div>
          )
        )}
      </div>
    </div>
  )}

  const renderReasoningFamilies = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🧠</span>
        <h3 className={styles.sectionTitle}>Reasoning Families</h3>
        <span className={styles.badge}>{config?.reasoning_families ? Object.keys(config.reasoning_families).length : 0} families</span>
      </div>
      <div className={styles.sectionContent}>
        {config?.reasoning_families && Object.keys(config.reasoning_families).length > 0 ? (
          <div className={styles.reasoningFamiliesGrid}>
            {Object.entries(config.reasoning_families).map(([familyName, familyConfig]) => (
              <div key={familyName} className={styles.reasoningFamilyCard}>
                <div className={styles.reasoningFamilyHeader}>
                  <span className={styles.reasoningFamilyName}>{familyName}</span>
                  <button
                    className={styles.editButton}
                    onClick={() => {
                      openEditModal(
                        `Edit Reasoning Family: ${familyName}`,
                        { ...familyConfig },
                        [
                          {
                            name: 'type',
                            label: 'Type',
                            type: 'select',
                            options: ['openai', 'anthropic', 'google', 'custom'],
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
                          const newConfig = { ...config }
                          if (newConfig.reasoning_families) {
                            newConfig.reasoning_families[familyName] = data
                          }
                          await saveConfig(newConfig)
                        }
                      )
                    }}
                  >
                    ✏️
                  </button>
                </div>
                <div className={styles.reasoningFamilyBody}>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Type</span>
                    <span className={styles.configValue}>{familyConfig.type}</span>
                  </div>
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Parameter</span>
                    <span className={styles.configValue}><code>{familyConfig.parameter}</code></span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className={styles.emptyState}>No reasoning families configured</div>
        )}
      </div>
    </div>
  )

  // ============================================================================
  // 5. TOOLS SELECTION SECTION
  // ============================================================================

  const renderToolsConfiguration = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🔧</span>
        <h3 className={styles.sectionTitle}>Tools Configuration</h3>
        {config?.tools && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Tools Configuration',
                config?.tools || {},
                [
                  {
                    name: 'enabled',
                    label: 'Enable Tool Auto-Selection',
                    type: 'boolean',
                    description: 'Enable automatic tool selection based on similarity'
                  },
                  {
                    name: 'top_k',
                    label: 'Top K',
                    type: 'number',
                    placeholder: '3',
                    description: 'Number of top similar tools to select'
                  },
                  {
                    name: 'similarity_threshold',
                    label: 'Similarity Threshold',
                    type: 'percentage',
                    placeholder: '70',
                    description: 'Minimum similarity score for tool selection (0-100%)',
                    step: 1
                  },
                  {
                    name: 'fallback_to_empty',
                    label: 'Fallback to Empty',
                    type: 'boolean',
                    description: 'Return empty list if no tools meet threshold'
                  },
                  {
                    name: 'tools_db_path',
                    label: 'Tools Database Path',
                    type: 'text',
                    placeholder: 'config/tools_db.json',
                    description: 'Path to tools database JSON file'
                  }
                ],
                async (data) => {
                  const newConfig = { ...config }
                  newConfig.tools = data
                  await saveConfig(newConfig)
                }
              )
            }}
          >
            ✏️ Edit
          </button>
        )}
      </div>
      <div className={styles.sectionContent}>
        {config?.tools ? (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Tool Auto-Selection</span>
              <span className={`${styles.statusBadge} ${config.tools.enabled ? styles.statusActive : styles.statusInactive}`}>
                {config.tools.enabled ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            {config.tools.enabled && (
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Top K</span>
                  <span className={styles.configValue}>{config.tools.top_k}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Similarity Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(config.tools.similarity_threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Fallback to Empty</span>
                  <span className={styles.configValue}>{config.tools.fallback_to_empty ? 'Yes' : 'No'}</span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className={styles.emptyState}>Tools configuration not available</div>
        )}
      </div>
    </div>
  )

  const renderToolsDB = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🗄️</span>
        <h3 className={styles.sectionTitle}>Tools Database</h3>
        {toolsData.length > 0 && <span className={styles.badge}>{toolsData.length} tools</span>}
      </div>
      <div className={styles.sectionContent}>
        {config?.tools?.tools_db_path ? (
          <>
            <div className={styles.featureCard}>
              <div className={styles.featureHeader}>
                <span className={styles.featureTitle}>Database Path</span>
              </div>
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Path</span>
                  <span className={styles.configValue}>{config.tools.tools_db_path}</span>
                </div>
              </div>
            </div>

            {toolsLoading && <div className={styles.loadingState}>Loading tools...</div>}
            {toolsError && <div className={styles.errorState}>Error loading tools: {toolsError}</div>}

            {!toolsLoading && !toolsError && toolsData.length > 0 && (
                <div className={styles.toolsGrid}>
                  {toolsData.map((tool, index) => (
                    <div key={index} className={styles.toolCard}>
                      <div className={styles.toolHeader}>
                        <span className={styles.toolName}>{tool.tool.function.name}</span>
                        {tool.category && (
                          <span className={`${styles.badge} ${styles.badgeInfo}`}>{tool.category}</span>
                        )}
                      </div>

                      {/* Function Description */}
                      <div className={styles.toolFunctionDescription}>
                        <strong>Function:</strong> {tool.tool.function.description}
                      </div>

                      {/* Similarity Description (used for matching) */}
                      {tool.description && tool.description !== tool.tool.function.description && (
                        <div className={styles.toolSimilarityDescription}>
                          <div className={styles.similarityDescriptionLabel}>🔍 Similarity Keywords</div>
                          <div className={styles.similarityDescriptionText}>{tool.description}</div>
                        </div>
                      )}

                      {/* Parameters */}
                      {tool.tool.function.parameters.properties && (
                        <div className={styles.toolParameters}>
                          <div className={styles.toolParametersHeader}>Parameters:</div>
                          {Object.entries(tool.tool.function.parameters.properties).map(([paramName, paramInfo]: [string, any]) => (
                            <div key={paramName} className={styles.toolParameter}>
                              <div>
                                <span className={styles.parameterName}>
                                  {paramName}
                                  {tool.tool.function.parameters.required?.includes(paramName) && (
                                    <span className={styles.requiredBadge}>*</span>
                                  )}
                                </span>
                                <span className={styles.parameterType}>{paramInfo.type}</span>
                              </div>
                              {paramInfo.description && (
                                <div className={styles.parameterDescription}>{paramInfo.description}</div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Tags */}
                      {tool.tags && tool.tags.length > 0 && (
                        <div className={styles.toolTags}>
                          {tool.tags.map((tag, idx) => (
                            <span key={idx} className={styles.toolTag}>{tag}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
            )}
          </>
        ) : (
          <div className={styles.emptyState}>Tools database path not configured</div>
        )}
      </div>
    </div>
  )

  // ============================================================================
  // 6. OBSERVABILITY SECTION
  // ============================================================================

  const renderObservabilityTracing = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🔍</span>
        <h3 className={styles.sectionTitle}>Distributed Tracing</h3>
        {config?.observability?.tracing && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Distributed Tracing Configuration',
                config?.observability?.tracing || {},
                [
                  {
                    name: 'enabled',
                    label: 'Enable Tracing',
                    type: 'boolean',
                    description: 'Enable distributed tracing'
                  },
                  {
                    name: 'provider',
                    label: 'Provider',
                    type: 'select',
                    options: ['jaeger', 'zipkin', 'otlp'],
                    description: 'Tracing provider'
                  },
                  {
                    name: 'exporter',
                    label: 'Exporter Configuration (JSON)',
                    type: 'json',
                    placeholder: '{"type": "otlp", "endpoint": "http://localhost:4318"}',
                    description: 'Exporter configuration as JSON object'
                  },
                  {
                    name: 'sampling',
                    label: 'Sampling Configuration (JSON)',
                    type: 'json',
                    placeholder: '{"type": "probabilistic", "rate": 0.1}',
                    description: 'Sampling configuration as JSON object'
                  },
                  {
                    name: 'resource',
                    label: 'Resource Configuration (JSON)',
                    type: 'json',
                    placeholder: '{"service_name": "semantic-router", "service_version": "1.0.0", "deployment_environment": "production"}',
                    description: 'Resource attributes as JSON object'
                  }
                ],
                async (data) => {
                  const newConfig = { ...config }
                  if (!newConfig.observability) newConfig.observability = {}
                  newConfig.observability.tracing = data
                  await saveConfig(newConfig)
                }
              )
            }}
          >
            ✏️ Edit
          </button>
        )}
      </div>
      <div className={styles.sectionContent}>
        {config?.observability?.tracing ? (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Tracing Status</span>
              <span className={`${styles.statusBadge} ${config.observability.tracing.enabled ? styles.statusActive : styles.statusInactive}`}>
                {config.observability.tracing.enabled ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            {config.observability.tracing.enabled && (
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Provider</span>
                  <span className={styles.configValue}>{config.observability.tracing.provider}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Exporter Type</span>
                  <span className={styles.configValue}>{config.observability.tracing.exporter.type}</span>
                </div>
                {config.observability.tracing.exporter.endpoint && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Endpoint</span>
                    <span className={styles.configValue}>{config.observability.tracing.exporter.endpoint}</span>
                  </div>
                )}
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Sampling Type</span>
                  <span className={styles.configValue}>{config.observability.tracing.sampling.type}</span>
                </div>
                {config.observability.tracing.sampling.rate !== undefined && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Sampling Rate</span>
                    <span className={styles.configValue}>{(config.observability.tracing.sampling.rate * 100).toFixed(0)}%</span>
                  </div>
                )}
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Service Name</span>
                  <span className={styles.configValue}>{config.observability.tracing.resource.service_name}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Service Version</span>
                  <span className={styles.configValue}>{config.observability.tracing.resource.service_version}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Environment</span>
                  <span className={`${styles.badge} ${styles[`badge${config.observability.tracing.resource.deployment_environment}`]}`}>
                    {config.observability.tracing.resource.deployment_environment}
                  </span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className={styles.emptyState}>Tracing not configured</div>
        )}
      </div>
    </div>
  )

  // ============================================================================
  // 7. CLASSIFICATION API SECTION
  // ============================================================================

  const renderClassificationAPI = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🔌</span>
        <h3 className={styles.sectionTitle}>Batch Classification API</h3>
        {config?.api?.batch_classification && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Batch Classification API Configuration',
                config?.api?.batch_classification || {},
                [
                  {
                    name: 'max_batch_size',
                    label: 'Max Batch Size',
                    type: 'number',
                    required: true,
                    placeholder: '100',
                    description: 'Maximum number of items in a single batch'
                  },
                  {
                    name: 'concurrency_threshold',
                    label: 'Concurrency Threshold',
                    type: 'number',
                    placeholder: '10',
                    description: 'Threshold to trigger concurrent processing'
                  },
                  {
                    name: 'max_concurrency',
                    label: 'Max Concurrency',
                    type: 'number',
                    placeholder: '5',
                    description: 'Maximum number of concurrent batch processes'
                  },
                  {
                    name: 'metrics',
                    label: 'Metrics Configuration (JSON)',
                    type: 'json',
                    placeholder: '{"enabled": true, "sample_rate": 0.1, "detailed_goroutine_tracking": false, "high_resolution_timing": true}',
                    description: 'Metrics collection configuration as JSON object'
                  }
                ],
                async (data) => {
                  const newConfig = { ...config }
                  if (!newConfig.api) newConfig.api = {}
                  newConfig.api.batch_classification = data
                  await saveConfig(newConfig)
                }
              )
            }}
          >
            ✏️ Edit
          </button>
        )}
      </div>
      <div className={styles.sectionContent}>
        {config?.api?.batch_classification ? (
          <>
            <div className={styles.featureCard}>
              <div className={styles.featureHeader}>
                <span className={styles.featureTitle}>Batch Configuration</span>
              </div>
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Max Batch Size</span>
                  <span className={styles.configValue}>{config.api.batch_classification.max_batch_size}</span>
                </div>
                {config.api.batch_classification.concurrency_threshold !== undefined && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Concurrency Threshold</span>
                    <span className={styles.configValue}>{config.api.batch_classification.concurrency_threshold}</span>
                  </div>
                )}
                {config.api.batch_classification.max_concurrency !== undefined && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Max Concurrency</span>
                    <span className={styles.configValue}>{config.api.batch_classification.max_concurrency}</span>
                  </div>
                )}
              </div>
            </div>

            {config.api.batch_classification.metrics && (
              <div className={styles.featureCard}>
                <div className={styles.featureHeader}>
                  <span className={styles.featureTitle}>Metrics Collection</span>
                  <span className={`${styles.statusBadge} ${config.api.batch_classification.metrics.enabled ? styles.statusActive : styles.statusInactive}`}>
                    {config.api.batch_classification.metrics.enabled ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                {config.api.batch_classification.metrics.enabled && (
                  <div className={styles.featureBody}>
                    {config.api.batch_classification.metrics.sample_rate !== undefined && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Sample Rate</span>
                        <span className={styles.configValue}>{(config.api.batch_classification.metrics.sample_rate * 100).toFixed(0)}%</span>
                      </div>
                    )}
                    {config.api.batch_classification.metrics.detailed_goroutine_tracking !== undefined && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Goroutine Tracking</span>
                        <span className={styles.configValue}>{config.api.batch_classification.metrics.detailed_goroutine_tracking ? 'Yes' : 'No'}</span>
                      </div>
                    )}
                    {config.api.batch_classification.metrics.high_resolution_timing !== undefined && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>High Resolution Timing</span>
                        <span className={styles.configValue}>{config.api.batch_classification.metrics.high_resolution_timing ? 'Yes' : 'No'}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <div className={styles.emptyState}>Batch classification API not configured</div>
        )}
      </div>
    </div>
  )

  // ============================================================================
  // SECTION PANEL RENDERS - Aligned with Python CLI config structure
  // ============================================================================

  // Signals Section - Keywords, Embeddings, Domains, Preferences (config.yaml)
  const renderSignalsSection = () => {
    const signals = config?.signals
    const keywordsCount = signals?.keywords?.length || 0
    const embeddingsCount = signals?.embeddings?.length || 0
    const domainsCount = signals?.domains?.length || 0
    const preferencesCount = signals?.preferences?.length || 0
    const factCheckCount = signals?.fact_check?.length || 0
    const feedbacksCount = signals?.user_feedbacks?.length || 0

    return (
      <div className={styles.sectionPanel}>
        {/* Keywords */}
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>🔑</span>
            <h3 className={styles.sectionTitle}>Keywords</h3>
            <span className={styles.badge}>{keywordsCount} signals</span>
          </div>
          <div className={styles.sectionContent}>
            {keywordsCount > 0 ? (
              <div className={styles.categoryGridTwoColumn}>
                {signals?.keywords?.map((kw, idx) => (
                  <div key={idx} className={styles.categoryCard}>
                    <div className={styles.categoryHeader}>
                      <span className={styles.categoryName}>{kw.name}</span>
                      <span className={`${styles.badge} ${styles.badgeInfo}`}>{kw.operator}</span>
                    </div>
                    <div className={styles.tagsContainer}>
                      {kw.keywords.map((word, i) => (
                        <span key={i} className={styles.tag}>{word}</span>
                      ))}
                    </div>
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>Case Sensitive</span>
                      <span className={styles.configValue}>{kw.case_sensitive ? 'Yes' : 'No'}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className={styles.emptyState}>No keyword signals configured</div>
            )}
          </div>
        </div>

        {/* Embeddings */}
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>🧬</span>
            <h3 className={styles.sectionTitle}>Embeddings</h3>
            <span className={styles.badge}>{embeddingsCount} signals</span>
          </div>
          <div className={styles.sectionContent}>
            {embeddingsCount > 0 ? (
              <div className={styles.categoryGridTwoColumn}>
                {signals?.embeddings?.map((emb, idx) => (
                  <div key={idx} className={styles.categoryCard}>
                    <div className={styles.categoryHeader}>
                      <span className={styles.categoryName}>{emb.name}</span>
                      <span className={`${styles.badge} ${styles.badgeSuccess}`}>θ≥{emb.threshold}</span>
                    </div>
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>Aggregation</span>
                      <span className={styles.configValue}>{emb.aggregation_method}</span>
                    </div>
                    <div className={styles.tagsContainer}>
                      {emb.candidates.map((c, i) => (
                        <span key={i} className={styles.tag} title={c}>{c.substring(0, 30)}...</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className={styles.emptyState}>No embedding signals configured</div>
            )}
          </div>
        </div>

        {/* Domains */}
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>📁</span>
            <h3 className={styles.sectionTitle}>Domains</h3>
            <span className={styles.badge}>{domainsCount} domains</span>
          </div>
          <div className={styles.sectionContent}>
            {domainsCount > 0 ? (
              <div className={styles.categoryGridTwoColumn}>
                {signals?.domains?.map((domain, idx) => (
                  <div key={idx} className={styles.categoryCard}>
                    <div className={styles.categoryHeader}>
                      <span className={styles.categoryName}>{domain.name}</span>
                      <button
                        className={styles.editButton}
                        onClick={() => {
                          openEditModal(
                            `Edit Domain: ${domain.name}`,
                            { description: domain.description || '' },
                            [
                              {
                                name: 'description',
                                label: 'Description',
                                type: 'textarea',
                                placeholder: 'Describe this domain...',
                              }
                            ],
                            async (data) => {
                              const newConfig = { ...config }
                              if (newConfig.signals?.domains) {
                                newConfig.signals.domains[idx] = { ...domain, description: data.description }
                              }
                              await saveConfig(newConfig)
                            }
                          )
                        }}
                      >
                        ✏️
                      </button>
                    </div>
                    {domain.description && (
                      <p className={styles.categoryDescription}>{domain.description}</p>
                    )}
                    {domain.mmlu_categories && domain.mmlu_categories.length > 0 && (
                      <div className={styles.tagsContainer}>
                        {domain.mmlu_categories.map((cat: string, i: number) => (
                          <span key={i} className={styles.tag}>{cat}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className={styles.emptyState}>No domains configured</div>
            )}
          </div>
        </div>

        {/* Preferences */}
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>⚙️</span>
            <h3 className={styles.sectionTitle}>Preferences</h3>
            <span className={styles.badge}>{preferencesCount} preferences</span>
          </div>
          <div className={styles.sectionContent}>
            {preferencesCount > 0 ? (
              <div className={styles.categoryGridTwoColumn}>
                {signals?.preferences?.map((pref, idx) => (
                  <div key={idx} className={styles.categoryCard}>
                    <div className={styles.categoryHeader}>
                      <span className={styles.categoryName}>{pref.name}</span>
                    </div>
                    {pref.description && (
                      <p className={styles.categoryDescription}>{pref.description}</p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className={styles.emptyState}>No preferences configured</div>
            )}
          </div>
        </div>

        {/* Fact Check */}
        {factCheckCount > 0 && (
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionIcon}>✓</span>
              <h3 className={styles.sectionTitle}>Fact Check Signals</h3>
              <span className={styles.badge}>{factCheckCount} signals</span>
            </div>
            <div className={styles.sectionContent}>
              <div className={styles.categoryGridTwoColumn}>
                {signals?.fact_check?.map((fc, idx) => (
                  <div key={idx} className={styles.categoryCard}>
                    <div className={styles.categoryHeader}>
                      <span className={styles.categoryName}>{fc.name}</span>
                    </div>
                    <p className={styles.categoryDescription}>{fc.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* User Feedbacks */}
        {feedbacksCount > 0 && (
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionIcon}>💬</span>
              <h3 className={styles.sectionTitle}>User Feedback Signals</h3>
              <span className={styles.badge}>{feedbacksCount} signals</span>
            </div>
            <div className={styles.sectionContent}>
              <div className={styles.categoryGridTwoColumn}>
                {signals?.user_feedbacks?.map((uf, idx) => (
                  <div key={idx} className={styles.categoryCard}>
                    <div className={styles.categoryHeader}>
                      <span className={styles.categoryName}>{uf.name}</span>
                    </div>
                    <p className={styles.categoryDescription}>{uf.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Legacy format notice */}
        {!isPythonCLI && (
          <div className={styles.section}>
            <div className={styles.emptyState}>
              ⚠️ Signals are only available in Python CLI config format. 
              Current config uses legacy format - use "Intelligent Routing" features instead.
            </div>
          </div>
        )}
      </div>
    )
  }

  // Decisions Section - Routing rules with priorities (config.yaml)
  const renderDecisionsSection = () => {
    const decisions = config?.decisions || []
    const defaultModel = getDefaultModel()

    return (
      <div className={styles.sectionPanel}>
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>🔀</span>
            <h3 className={styles.sectionTitle}>Routing Decisions</h3>
            <span className={styles.badge}>{decisions.length} rules</span>
          </div>
          <div className={styles.sectionContent}>
            {/* Default Model Info */}
            <div className={styles.coreSettingsInline}>
              <div className={styles.inlineConfigRow}>
                <span className={styles.inlineConfigLabel}>🎯 Default Model:</span>
                <span className={styles.inlineConfigValue}>{defaultModel || 'N/A'}</span>
              </div>
              <div className={styles.inlineConfigRow}>
                <span className={styles.inlineConfigLabel}>⚡ Default Reasoning:</span>
                <span className={`${styles.badge} ${styles[`badge${config?.providers?.default_reasoning_effort || 'medium'}`]}`}>
                  {config?.providers?.default_reasoning_effort || 'medium'}
                </span>
              </div>
            </div>

            {isPythonCLI && decisions.length > 0 ? (
              <div className={styles.categoryGridTwoColumn}>
                {decisions.map((decision, idx) => (
                  <div key={idx} className={styles.categoryCard}>
                    <div className={styles.categoryHeader}>
                      <span className={styles.categoryName}>{decision.name}</span>
                      <span className={`${styles.badge} ${styles.badgeInfo}`}>P{decision.priority}</span>
                    </div>
                    {decision.description && (
                      <p className={styles.categoryDescription}>{decision.description}</p>
                    )}
                    
                    {/* Rules */}
                    {decision.rules && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Rules ({decision.rules.operator})</span>
                        <div className={styles.tagsContainer}>
                          {decision.rules.conditions?.map((cond, i) => (
                            <span key={i} className={styles.tag}>{cond.type}: {cond.name}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Model References */}
                    {decision.modelRefs && decision.modelRefs.length > 0 && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Models</span>
                        <div className={styles.endpointTags}>
                          {decision.modelRefs.map((ref, i) => (
                            <span key={i} className={styles.endpointTag}>
                              {ref.model.split('/').pop()} {ref.use_reasoning && '⚡'}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Plugins */}
                    {decision.plugins && decision.plugins.length > 0 && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Plugins</span>
                        <div className={styles.tagsContainer}>
                          {decision.plugins.map((plugin, i) => (
                            <span key={i} className={styles.tag}>🔌 {plugin.type}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : !isPythonCLI ? (
              <div className={styles.emptyState}>
                ⚠️ Decisions are only available in Python CLI config format.
                Current config uses legacy format - see "Categories" in legacy mode.
              </div>
            ) : (
              <div className={styles.emptyState}>No routing decisions configured</div>
            )}
          </div>
        </div>

        {/* Reasoning Families */}
        {renderReasoningFamilies()}
      </div>
    )
  }

  // Models Section - Provider models and endpoints (config.yaml)
  const renderModelsSection = () => (
    <div className={styles.sectionPanel}>
      {renderUserDefinedModels()}
      {renderUserDefinedEndpoints()}
    </div>
  )

  // Router Configuration Section - System defaults from router-defaults.yaml
  const renderRouterConfigSection = () => (
    <div className={styles.sectionPanel}>
      {/* Source indicator */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionIcon}>📄</span>
          <h3 className={styles.sectionTitle}>Configuration Source</h3>
          {routerDefaultsLoading && <span className={styles.badge}>Loading...</span>}
        </div>
        <div className={styles.sectionContent}>
          <div className={styles.coreSettingsInline}>
            <div className={styles.inlineConfigRow}>
              <span className={styles.inlineConfigLabel}>📁 Source File:</span>
              <span className={styles.inlineConfigValue}>
                {routerDefaults && Object.keys(routerDefaults).length > 0 
                  ? '.vllm-sr/router-defaults.yaml' 
                  : 'config.yaml (fallback)'}
              </span>
            </div>
            {routerDefaults && Object.keys(routerDefaults).length > 0 && (
              <div className={styles.inlineConfigRow}>
                <span className={styles.inlineConfigLabel}>ℹ️ Note:</span>
                <span className={styles.configValue} style={{ fontSize: '0.85em', opacity: 0.8 }}>
                  Router defaults are system settings that apply across all configurations
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Semantic Cache */}
      {renderSimilarityBERT()}
      
      {/* Prompt Guard */}
      {renderPIIModernBERT()}
      {renderJailbreakModernBERT()}
      
      {/* Classifier */}
      {renderClassifyBERT()}
      
      {/* Tools */}
      {renderToolsConfiguration()}
      {renderToolsDB()}
      
      {/* Observability */}
      {renderObservabilityTracing()}
      
      {/* Classification API */}
      {renderClassificationAPI()}
      
      {/* Legacy Categories (for backward compatibility) */}
      {!isPythonCLI && renderCategories()}
    </div>
  )

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'signals':
        return renderSignalsSection()
      case 'decisions':
        return renderDecisionsSection()
      case 'models':
        return renderModelsSection()
      case 'router-config':
        return renderRouterConfigSection()
      default:
        return renderSignalsSection()
    }
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h2 className={styles.title}>⚙️ Configuration</h2>
          <div className={styles.viewToggle}>
            <button
              className={`${styles.toggleButton} ${selectedView === 'structured' ? styles.active : ''}`}
              onClick={() => setSelectedView('structured')}
            >
              📋 Structured
            </button>
            <button
              className={`${styles.toggleButton} ${selectedView === 'raw' ? styles.active : ''}`}
              onClick={() => setSelectedView('raw')}
            >
              💻 Raw YAML
            </button>
          </div>
        </div>
        <button onClick={handleRefresh} className={styles.button} disabled={loading}>
          🔄 Refresh
        </button>
      </div>

      <div className={styles.content}>
        {loading && (
          <div className={styles.loading}>
            <div className={styles.spinner}></div>
            <p>Loading configuration...</p>
          </div>
        )}

        {error && !loading && (
          <div className={styles.error}>
            <span className={styles.errorIcon}>⚠️</span>
            <div>
              <h3>Error Loading Config</h3>
              <p>{error}</p>
            </div>
          </div>
        )}

        {config && !loading && !error && (
          <>
            {selectedView === 'structured' ? (
              <div className={styles.contentArea}>
                {renderActiveSection()}
              </div>
            ) : (
              <pre className={styles.codeBlock}>
                <code>{JSON.stringify(config, null, 2)}</code>
              </pre>
            )}
          </>
        )}
      </div>

      {/* Edit Modal */}
      <EditModal
        isOpen={editModalOpen}
        onClose={closeEditModal}
        onSave={editModalCallback || (async () => {})}
        title={editModalTitle}
        data={editModalData}
        fields={editModalFields}
        mode={editModalMode}
      />
    </div>
  )
}

export default ConfigPage
