import React, { useState, useEffect } from 'react'
import styles from './ConfigPage.module.css'
import { ConfigSection } from '../components/ConfigNav'
import EditModal, { FieldConfig } from '../components/EditModal'

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
  model_scores: ModelScore[]
}

interface ToolFunction {
  name: string
  description: string
  parameters: {
    type: string
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
  bert_model?: ModelConfig
  semantic_cache?: {
    enabled: boolean
    backend_type?: string
    similarity_threshold: number
    max_entries: number
    ttl_seconds: number
    eviction_policy?: string
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

const ConfigPage: React.FC<ConfigPageProps> = ({ activeSection = 'models' }) => {
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedView, setSelectedView] = useState<'structured' | 'raw'>('structured')

  // Tools database state
  const [toolsData, setToolsData] = useState<Tool[]>([])
  const [toolsLoading, setToolsLoading] = useState(false)
  const [toolsError, setToolsError] = useState<string | null>(null)

  // Endpoint address visibility state (for security masking)
  const [visibleAddresses, setVisibleAddresses] = useState<Set<number>>(new Set())

  // Edit modal state
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [editModalTitle, setEditModalTitle] = useState('')
  const [editModalData, setEditModalData] = useState<any>(null)
  const [editModalFields, setEditModalFields] = useState<FieldConfig[]>([])
  const [editModalMode, setEditModalMode] = useState<'edit' | 'add'>('edit')
  const [editModalCallback, setEditModalCallback] = useState<((data: any) => Promise<void>) | null>(null)

  useEffect(() => {
    fetchConfig()
  }, [])

  // Fetch tools database when config is loaded
  useEffect(() => {
    if (config?.tools?.tools_db_path) {
      fetchToolsDB()
    }
  }, [config?.tools?.tools_db_path])

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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch config')
      setConfig(null)
    } finally {
      setLoading(false)
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

  // @ts-ignore - Will be used when edit buttons are added
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
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      // Refresh config after save
      await fetchConfig()
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to save configuration')
    }
  }

  // @ts-ignore - Will be used when edit buttons are added
  const openEditModal = (
    title: string,
    data: any,
    fields: FieldConfig[],
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
  }

  // ============================================================================
  // 1. MODELS SECTION
  // ============================================================================

  const renderUserDefinedEndpoints = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🔌</span>
        <h3 className={styles.sectionTitle}>User Defined Endpoints</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span className={styles.badge}>{config?.vllm_endpoints?.length || 0} endpoints</span>
          <button
            className={styles.addButton}
            onClick={() => {
              openEditModal(
                'Add New Endpoint',
                { name: '', address: '', port: 8000, weight: 1 },
                [
                  { name: 'name', label: 'Endpoint Name', type: 'text', required: true, placeholder: 'e.g., vllm-endpoint-1' },
                  { name: 'address', label: 'Address', type: 'text', required: true, placeholder: 'e.g., localhost' },
                  { name: 'port', label: 'Port', type: 'number', required: true, placeholder: '8000' },
                  { name: 'weight', label: 'Weight', type: 'number', placeholder: '1', description: 'Load balancing weight for this endpoint' }
                ],
                async (data) => {
                  const newConfig = { ...config }
                  if (!newConfig.vllm_endpoints) newConfig.vllm_endpoints = []
                  newConfig.vllm_endpoints.push(data)
                  await saveConfig(newConfig)
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
        {config?.vllm_endpoints && config.vllm_endpoints.length > 0 ? (
          config.vllm_endpoints.map((endpoint, index) => (
            <div key={index} className={styles.endpointCard}>
              <div className={styles.endpointHeader}>
                <span className={styles.endpointName}>{endpoint.name}</span>
                <div className={styles.cardActions}>
                  <button
                    className={styles.editButton}
                    onClick={() => {
                      openEditModal(
                        `Edit Endpoint: ${endpoint.name}`,
                        { ...endpoint },
                        [
                          { name: 'name', label: 'Endpoint Name', type: 'text', required: true },
                          { name: 'address', label: 'Address', type: 'text', required: true },
                          { name: 'port', label: 'Port', type: 'number', required: true },
                          { name: 'weight', label: 'Weight', type: 'number', description: 'Load balancing weight for this endpoint' }
                        ],
                        async (data) => {
                          const newConfig = { ...config }
                          if (newConfig.vllm_endpoints) {
                            newConfig.vllm_endpoints[index] = data
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
                      if (confirm(`Are you sure you want to delete endpoint "${endpoint.name}"?`)) {
                        const newConfig = { ...config }
                        if (newConfig.vllm_endpoints) {
                          newConfig.vllm_endpoints.splice(index, 1)
                        }
                        saveConfig(newConfig)
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
                      {visibleAddresses.has(index) ? endpoint.address : maskAddress(endpoint.address)}
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
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>🔌 Port</span>
                  <span className={styles.configValue}>{endpoint.port}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>⚖️ Weight</span>
                  <span className={styles.configValue}>{endpoint.weight}</span>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className={styles.emptyState}>No endpoints configured</div>
        )}
      </div>
    </div>
  )

  const renderUserDefinedModels = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🤖</span>
        <h3 className={styles.sectionTitle}>User Defined Models</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span className={styles.badge}>{config?.model_config ? Object.keys(config.model_config).length : 0} models</span>
          <button
            className={styles.addButton}
            onClick={() => {
              // Get available reasoning families
              const reasoningFamilies = config?.reasoning_families
                ? Object.keys(config.reasoning_families)
                : []

              // Get available endpoints
              const endpoints = config?.vllm_endpoints
                ? config.vllm_endpoints.map(ep => ep.name)
                : []

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
                  const newConfig = { ...config }
                  if (!newConfig.model_config) newConfig.model_config = {}
                  const { model_name, currency, prompt_per_1m, completion_per_1m, pii_allow_by_default, pii_types_allowed, ...otherData } = data

                  // Build model data
                  const modelData: any = { ...otherData }

                  // Add pricing if any value is set
                  if (currency || prompt_per_1m || completion_per_1m) {
                    modelData.pricing = {
                      currency: currency || 'USD',
                      prompt_per_1m: prompt_per_1m || 0,
                      completion_per_1m: completion_per_1m || 0
                    }
                  }

                  // Add PII policy
                  modelData.pii_policy = {
                    allow_by_default: pii_allow_by_default,
                    pii_types_allowed: pii_types_allowed || []
                  }

                  newConfig.model_config[model_name] = modelData
                  newConfig.model_config[model_name] = modelData
                  await saveConfig(newConfig)
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
        {config?.model_config && Object.keys(config.model_config).length > 0 ? (
          <div className={styles.modelConfigGrid}>
            {Object.entries(config.model_config).map(([modelName, modelConfig]) => (
              <div key={modelName} className={styles.modelConfigCard}>
                <div className={styles.modelConfigHeader}>
                  <span className={styles.modelConfigName}>{modelName}</span>
                  <div className={styles.cardActions}>
                    <button
                      className={styles.editButton}
                      onClick={() => {
                        // Get available reasoning families
                        const reasoningFamilies = config?.reasoning_families
                          ? Object.keys(config.reasoning_families)
                          : []

                        // Get available endpoints
                        const endpoints = config?.vllm_endpoints
                          ? config.vllm_endpoints.map(ep => ep.name)
                          : []

                        // PII types
                        const piiTypes = [
                          'AGE', 'CREDIT_CARD', 'DATE_TIME', 'DOMAIN_NAME',
                          'EMAIL_ADDRESS', 'GPE', 'IBAN_CODE', 'IP_ADDRESS',
                          'NO_PII', 'NRP', 'ORGANIZATION', 'PERSON',
                          'PHONE_NUMBER', 'STREET_ADDRESS', 'US_DRIVER_LICENSE',
                          'US_SSN', 'ZIP_CODE'
                        ]

                        // Flatten model config for editing
                        const editData = {
                          reasoning_family: modelConfig.reasoning_family || '',
                          preferred_endpoints: modelConfig.preferred_endpoints || [],
                          currency: modelConfig.pricing?.currency || 'USD',
                          prompt_per_1m: modelConfig.pricing?.prompt_per_1m || 0,
                          completion_per_1m: modelConfig.pricing?.completion_per_1m || 0,
                          pii_allow_by_default: modelConfig.pii_policy?.allow_by_default ?? true,
                          pii_types_allowed: modelConfig.pii_policy?.pii_types_allowed || []
                        }

                        openEditModal(
                          `Edit Model: ${modelName}`,
                          editData,
                          [
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
                            const newConfig = { ...config }
                            if (newConfig.model_config) {
                              const { currency, prompt_per_1m, completion_per_1m, pii_allow_by_default, pii_types_allowed, ...otherData } = data

                              // Build model data
                              const modelData: any = { ...otherData }

                              // Add pricing if any value is set
                              if (currency || prompt_per_1m || completion_per_1m) {
                                modelData.pricing = {
                                  currency: currency || 'USD',
                                  prompt_per_1m: prompt_per_1m || 0,
                                  completion_per_1m: completion_per_1m || 0
                                }
                              }

                              // Add PII policy
                              modelData.pii_policy = {
                                allow_by_default: pii_allow_by_default,
                                pii_types_allowed: pii_types_allowed || []
                              }

                              newConfig.model_config[modelName] = modelData
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
                        if (confirm(`Are you sure you want to delete model "${modelName}"?`)) {
                          const newConfig = { ...config }
                          if (newConfig.model_config) {
                            delete newConfig.model_config[modelName]
                          }
                          saveConfig(newConfig)
                        }
                      }}
                    >
                      🗑️
                    </button>
                  </div>
                </div>
                <div className={styles.modelConfigBody}>
                  {modelConfig.reasoning_family && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>🧠 Reasoning Family</span>
                      <span className={`${styles.badge} ${styles.badgeInfo}`}>
                        {modelConfig.reasoning_family}
                      </span>
                    </div>
                  )}
                  {modelConfig.preferred_endpoints && modelConfig.preferred_endpoints.length > 0 && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>🔌 Preferred Endpoints</span>
                      <div className={styles.endpointTags}>
                        {modelConfig.preferred_endpoints.map((endpoint, idx) => (
                          <span key={idx} className={styles.endpointTag}>{endpoint}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {modelConfig.pricing && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>💰 Pricing</span>
                      <div className={styles.pricingContainer}>
                        <div className={styles.pricingItem}>
                          <span className={styles.pricingLabel}>Prompt</span>
                          <span className={styles.pricingValue}>
                            {modelConfig.pricing.prompt_per_1m?.toFixed(2) || '0.00'}
                          </span>
                          <span className={styles.pricingUnit}>
                            {modelConfig.pricing.currency || 'USD'}/1M
                          </span>
                        </div>
                        <div className={styles.pricingDivider}>|</div>
                        <div className={styles.pricingItem}>
                          <span className={styles.pricingLabel}>Completion</span>
                          <span className={styles.pricingValue}>
                            {modelConfig.pricing.completion_per_1m?.toFixed(2) || '0.00'}
                          </span>
                          <span className={styles.pricingUnit}>
                            {modelConfig.pricing.currency || 'USD'}/1M
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  {modelConfig.pii_policy && (
                    <div className={styles.configRow}>
                      <span className={styles.configLabel}>🔒 PII Policy</span>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                        <span className={`${styles.statusBadge} ${modelConfig.pii_policy.allow_by_default ? styles.statusActive : styles.statusInactive}`}>
                          {modelConfig.pii_policy.allow_by_default ? 'Allow by default' : 'Block by default'}
                        </span>
                        {modelConfig.pii_policy.pii_types_allowed && modelConfig.pii_policy.pii_types_allowed.length > 0 && (
                          <div className={styles.piiTypesTags}>
                            {modelConfig.pii_policy.pii_types_allowed.map((type, idx) => (
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
  )

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

  const renderCategories = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>📊</span>
        <h3 className={styles.sectionTitle}>Categories Configuration</h3>
        <span className={styles.badge}>{config?.categories?.length || 0} categories</span>
      </div>
      <div className={styles.sectionContent}>
        {/* Core Settings at the top */}
        <div className={styles.coreSettingsInline}>
          <div className={styles.inlineConfigRow}>
            <span className={styles.inlineConfigLabel}>🎯 Default Model:</span>
            <span className={styles.inlineConfigValue}>{config?.default_model || 'N/A'}</span>
          </div>
          <div className={styles.inlineConfigRow}>
            <span className={styles.inlineConfigLabel}>⚡ Default Reasoning Effort:</span>
            <span className={`${styles.badge} ${styles[`badge${config?.default_reasoning_effort || 'medium'}`]}`}>
              {config?.default_reasoning_effort || 'medium'}
            </span>
          </div>
        </div>

        {config?.categories && config.categories.length > 0 ? (
          <div className={styles.categoryGridTwoColumn}>
            {config.categories.map((category, index) => {
              // Get reasoning info from best model (first model score)
              const bestModel = category.model_scores?.[0]
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
                              if (!updatedCategory.model_scores) {
                                updatedCategory.model_scores = []
                              }
                              updatedCategory.model_scores.push(data)
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
                  {category.model_scores && category.model_scores.length > 0 ? (
                    category.model_scores.map((modelScore, modelIdx) => (
                      <div key={modelIdx} className={styles.modelScoreRow}>
                        <span className={styles.modelScoreName}>
                          {modelScore.model}
                          {modelScore.use_reasoning && <span className={styles.reasoningIcon}>🧠</span>}
                        </span>
                        <div className={styles.scoreBar}>
                          <div
                            className={styles.scoreBarFill}
                            style={{ width: `${modelScore.score * 100}%` }}
                          ></div>
                          <span className={styles.scoreText}>{(modelScore.score * 100).toFixed(0)}%</span>
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
                                  const newConfig = { ...config }
                                  if (newConfig.categories) {
                                    const updatedCategory = { ...category }
                                    updatedCategory.model_scores[modelIdx] = data
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
                                  updatedCategory.model_scores.splice(modelIdx, 1)
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
        )}
      </div>
    </div>
  )

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
  // SECTION PANEL RENDERS
  // ============================================================================

  const renderModelsSection = () => (
    <div className={styles.sectionPanel}>
      {renderUserDefinedModels()}
      {renderUserDefinedEndpoints()}
    </div>
  )

  const renderPromptGuardSection = () => (
    <div className={styles.sectionPanel}>
      {renderPIIModernBERT()}
      {renderJailbreakModernBERT()}
    </div>
  )

  const renderSimilarityCacheSection = () => (
    <div className={styles.sectionPanel}>
      {renderSimilarityBERT()}
    </div>
  )

  const renderIntelligentRoutingSection = () => (
    <div className={styles.sectionPanel}>
      {renderClassifyBERT()}
      {renderCategories()}
      {renderReasoningFamilies()}
    </div>
  )

  const renderToolsSelectionSection = () => (
    <div className={styles.sectionPanel}>
      {renderToolsConfiguration()}
      {renderToolsDB()}
    </div>
  )

  const renderObservabilitySection = () => (
    <div className={styles.sectionPanel}>
      {renderObservabilityTracing()}
    </div>
  )

  const renderClassificationAPISection = () => (
    <div className={styles.sectionPanel}>
      {renderClassificationAPI()}
    </div>
  )

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'models':
        return renderModelsSection()
      case 'prompt-guard':
        return renderPromptGuardSection()
      case 'similarity-cache':
        return renderSimilarityCacheSection()
      case 'intelligent-routing':
        return renderIntelligentRoutingSection()
      case 'tools-selection':
        return renderToolsSelectionSection()
      case 'observability':
        return renderObservabilitySection()
      case 'classification-api':
        return renderClassificationAPISection()
      default:
        return renderModelsSection()
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
