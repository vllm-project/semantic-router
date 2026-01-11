import React, { useState, useEffect } from 'react'
import styles from './ConfigPage.module.css'
import { ConfigSection } from '../components/ConfigNav'
import EditModal, { FieldConfig } from '../components/EditModal'
import ViewModal, { ViewSection } from '../components/ViewModal'
import { DataTable, Column } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import EndpointsEditor, { Endpoint } from '../components/EndpointsEditor'
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

// Removed maskAddress - no longer needed after removing endpoint visibility toggle

const ConfigPage: React.FC<ConfigPageProps> = ({ activeSection = 'signals' }) => {
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [configFormat, setConfigFormat] = useState<ConfigFormat>('python-cli')

  // Router defaults state (from .vllm-sr/router-defaults.yaml)
  const [routerDefaults, setRouterDefaults] = useState<ConfigData | null>(null)
  const [routerDefaultsLoading, setRouterDefaultsLoading] = useState(false)

  // Tools database state
  const [toolsData, setToolsData] = useState<Tool[]>([])
  const [toolsLoading, setToolsLoading] = useState(false)
  const [toolsError, setToolsError] = useState<string | null>(null)

  // Removed visibleAddresses state - no longer needed

  // Edit modal state
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [editModalTitle, setEditModalTitle] = useState('')
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [editModalData, setEditModalData] = useState<any>(null)
  const [editModalFields, setEditModalFields] = useState<FieldConfig[]>([])
  const [editModalMode, setEditModalMode] = useState<'edit' | 'add'>('edit')
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [editModalCallback, setEditModalCallback] = useState<((data: any) => Promise<void>) | null>(null)

  // View modal state
  const [viewModalOpen, setViewModalOpen] = useState(false)
  const [viewModalTitle, setViewModalTitle] = useState('')
  const [viewModalSections, setViewModalSections] = useState<ViewSection[]>([])
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [viewModalEditCallback, setViewModalEditCallback] = useState<(() => void) | null>(null)

  // Search state
  const [decisionsSearch, setDecisionsSearch] = useState('')
  const [signalsSearch, setSignalsSearch] = useState('')
  const [modelsSearch, setModelsSearch] = useState('')

  // Expandable rows state for models
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set())

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

  // Get effective config value - check router defaults first, then main config
  // Utility for merging config sources, will be used in render functions
  const getEffectiveConfig = (key: string) => {
    // For router defaults sections, prefer routerDefaults
    if (routerDefaults && routerDefaults[key] !== undefined) {
      return routerDefaults[key]
    }
    return config?.[key]
  }
  // Mark as used to avoid linting error
  void getEffectiveConfig

  // ============================================================================
  // HELPER FUNCTIONS - Normalize data access across config formats
  // ============================================================================

  // Helper: Check if using Python CLI format
  const isPythonCLI = configFormat === 'python-cli'

  // Effective router config - merges routerDefaults (system settings) with config (fallback)
  // For Python CLI: system settings like bert_model, tools, prompt_guard come from routerDefaults
  // For Legacy: these settings are in config.yaml directly
  const routerConfig = {
    bert_model: routerDefaults?.bert_model ?? config?.bert_model,
    semantic_cache: routerDefaults?.semantic_cache ?? config?.semantic_cache,
    tools: routerDefaults?.tools ?? config?.tools,
    prompt_guard: routerDefaults?.prompt_guard ?? config?.prompt_guard,
    classifier: routerDefaults?.classifier ?? config?.classifier,
    observability: routerDefaults?.observability ?? config?.observability,
    api: routerDefaults?.api ?? config?.api,
  }

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
  // 2. PROMPT GUARD SECTION
  // ============================================================================

  const renderPIIModernBERT = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>🔒</span>
        <h3 className={styles.sectionTitle}>PII Detection (ModernBERT)</h3>
        {routerConfig.classifier?.pii_model && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit PII Detection Configuration',
                routerConfig.classifier?.pii_model || {},
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
        {routerConfig.classifier?.pii_model ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>PII Classifier Model</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                {routerConfig.classifier.pii_model.use_cpu ? '💻 CPU' : '🎮 GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
                <span className={styles.configValue}>{routerConfig.classifier.pii_model.model_id}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
                <span className={styles.configValue}>{formatThreshold(routerConfig.classifier.pii_model.threshold)}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>ModernBERT</span>
                <span className={`${styles.statusBadge} ${routerConfig.classifier.pii_model.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
                  {routerConfig.classifier.pii_model.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
                </span>
              </div>
              {routerConfig.classifier.pii_model.pii_mapping_path && (
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Mapping Path</span>
                  <span className={styles.configValue}>{routerConfig.classifier.pii_model.pii_mapping_path}</span>
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
        {routerConfig.prompt_guard && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Jailbreak Detection Configuration',
                routerConfig.prompt_guard || {},
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
        {routerConfig.prompt_guard ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>Jailbreak Protection</span>
              <span className={`${styles.statusBadge} ${routerConfig.prompt_guard.enabled ? styles.statusActive : styles.statusInactive}`}>
                {routerConfig.prompt_guard.enabled ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            {routerConfig.prompt_guard.enabled && (
              <div className={styles.modelCardBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Model ID</span>
                  <span className={styles.configValue}>{routerConfig.prompt_guard.model_id}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(routerConfig.prompt_guard.threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Use CPU</span>
                  <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                    {routerConfig.prompt_guard.use_cpu ? '💻 CPU' : '🎮 GPU'}
                  </span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>ModernBERT</span>
                  <span className={`${styles.statusBadge} ${routerConfig.prompt_guard.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
                    {routerConfig.prompt_guard.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                {routerConfig.prompt_guard.jailbreak_mapping_path && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Mapping Path</span>
                    <span className={styles.configValue}>{routerConfig.prompt_guard.jailbreak_mapping_path}</span>
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

  const renderSimilarityBERT = () => {
    return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.sectionIcon}>⚡</span>
        <h3 className={styles.sectionTitle}>Similarity BERT Configuration</h3>
        {routerConfig.bert_model && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Similarity BERT Configuration',
                routerConfig.bert_model || {},
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
        {routerConfig.bert_model ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>BERT Model (Semantic Similarity)</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                {routerConfig.bert_model.use_cpu ? '💻 CPU' : '🎮 GPU'}
              </span>
            </div>
            <div className={styles.modelCardBody}>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Model ID</span>
                <span className={styles.configValue}>{routerConfig.bert_model.model_id}</span>
              </div>
              <div className={styles.configRow}>
                <span className={styles.configLabel}>Threshold</span>
                <span className={styles.configValue}>{formatThreshold(routerConfig.bert_model.threshold)}</span>
              </div>
            </div>
          </div>
        ) : (
          <div className={styles.emptyState}>BERT model not configured</div>
        )}

        {routerConfig.semantic_cache && (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Semantic Cache</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span className={`${styles.statusBadge} ${routerConfig.semantic_cache?.enabled ? styles.statusActive : styles.statusInactive}`}>
                  {routerConfig.semantic_cache?.enabled ? '✓ Enabled' : '✗ Disabled'}
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
            {routerConfig.semantic_cache?.enabled && (
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Backend Type</span>
                  <span className={styles.configValue}>{routerConfig.semantic_cache?.backend_type || 'memory'}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Similarity Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(routerConfig.semantic_cache?.similarity_threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Max Entries</span>
                  <span className={styles.configValue}>{routerConfig.semantic_cache?.max_entries}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>TTL</span>
                  <span className={styles.configValue}>{routerConfig.semantic_cache?.ttl_seconds}s</span>
                </div>
                {routerConfig.semantic_cache?.eviction_policy && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Eviction Policy</span>
                    <span className={styles.configValue}>{routerConfig.semantic_cache.eviction_policy}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )}

  // ============================================================================
  // 4. INTELLIGENT ROUTING SECTION
  // ============================================================================

  const renderClassifyBERT = () => {
    const hasInTree = routerConfig.classifier?.category_model
    const hasOutTree = routerConfig.classifier?.mcp_category_model?.enabled

    return (
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionIcon}>🎯</span>
          <h3 className={styles.sectionTitle}>Classify BERT Model</h3>
        </div>
        <div className={styles.sectionContent}>
          {/* In-tree Classifier */}
          {hasInTree && routerConfig.classifier?.category_model && (
            <div className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelCardTitle}>In-tree Category Classifier</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                    {routerConfig.classifier.category_model.use_cpu ? '💻 CPU' : '🎮 GPU'}
                  </span>
                  <button
                    className={styles.editButton}
                    onClick={() => {
                      openEditModal(
                        'Edit In-tree Category Classifier',
                        routerConfig.classifier?.category_model || {},
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
                  <span className={styles.configValue}>{routerConfig.classifier?.category_model?.model_id}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(routerConfig.classifier?.category_model?.threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>ModernBERT</span>
                  <span className={`${styles.statusBadge} ${routerConfig.classifier?.category_model?.use_modernbert ? styles.statusActive : styles.statusInactive}`}>
                    {routerConfig.classifier?.category_model?.use_modernbert ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                {routerConfig.classifier?.category_model?.category_mapping_path && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Mapping Path</span>
                    <span className={styles.configValue}>{routerConfig.classifier.category_model.category_mapping_path}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Out-tree Classifier (MCP) */}
          {hasOutTree && routerConfig.classifier?.mcp_category_model && (
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
                        routerConfig.classifier?.mcp_category_model || {},
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
                  <span className={styles.configValue}>{routerConfig.classifier?.mcp_category_model?.transport_type}</span>
                </div>
                {routerConfig.classifier?.mcp_category_model?.command && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Command</span>
                    <span className={styles.configValue}>{routerConfig.classifier.mcp_category_model.command}</span>
                  </div>
                )}
                {routerConfig.classifier?.mcp_category_model?.url && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>URL</span>
                    <span className={styles.configValue}>{routerConfig.classifier?.mcp_category_model?.url}</span>
                  </div>
                )}
                {routerConfig.classifier?.mcp_category_model?.tool_name && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Tool Name</span>
                    <span className={styles.configValue}>{routerConfig.classifier.mcp_category_model.tool_name}</span>
                  </div>
                )}
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(routerConfig.classifier?.mcp_category_model?.threshold)}</span>
                </div>
                {routerConfig.classifier?.mcp_category_model?.timeout_seconds && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Timeout</span>
                    <span className={styles.configValue}>{routerConfig.classifier.mcp_category_model.timeout_seconds}s</span>
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
        {routerConfig.tools && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Tools Configuration',
                routerConfig.tools || {},
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
        {routerConfig.tools ? (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Tool Auto-Selection</span>
              <span className={`${styles.statusBadge} ${routerConfig.tools.enabled ? styles.statusActive : styles.statusInactive}`}>
                {routerConfig.tools.enabled ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            {routerConfig.tools.enabled && (
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Top K</span>
                  <span className={styles.configValue}>{routerConfig.tools.top_k}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Similarity Threshold</span>
                  <span className={styles.configValue}>{formatThreshold(routerConfig.tools.similarity_threshold)}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Fallback to Empty</span>
                  <span className={styles.configValue}>{routerConfig.tools.fallback_to_empty ? 'Yes' : 'No'}</span>
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
        {routerConfig.observability?.tracing && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Distributed Tracing Configuration',
                routerConfig.observability?.tracing || {},
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
        {routerConfig.observability?.tracing ? (
          <div className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <span className={styles.featureTitle}>Tracing Status</span>
              <span className={`${styles.statusBadge} ${routerConfig.observability.tracing.enabled ? styles.statusActive : styles.statusInactive}`}>
                {routerConfig.observability.tracing.enabled ? '✓ Enabled' : '✗ Disabled'}
              </span>
            </div>
            {routerConfig.observability.tracing.enabled && (
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Provider</span>
                  <span className={styles.configValue}>{routerConfig.observability.tracing.provider}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Exporter Type</span>
                  <span className={styles.configValue}>{routerConfig.observability?.tracing?.exporter?.type}</span>
                </div>
                {routerConfig.observability?.tracing?.exporter?.endpoint && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Endpoint</span>
                    <span className={styles.configValue}>{routerConfig.observability.tracing.exporter.endpoint}</span>
                  </div>
                )}
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Sampling Type</span>
                  <span className={styles.configValue}>{routerConfig.observability?.tracing?.sampling?.type}</span>
                </div>
                {routerConfig.observability?.tracing?.sampling?.rate !== undefined && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Sampling Rate</span>
                    <span className={styles.configValue}>{((routerConfig.observability?.tracing?.sampling?.rate ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                )}
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Service Name</span>
                  <span className={styles.configValue}>{routerConfig.observability?.tracing?.resource?.service_name}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Service Version</span>
                  <span className={styles.configValue}>{routerConfig.observability?.tracing?.resource?.service_version}</span>
                </div>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Environment</span>
                  <span className={`${styles.badge} ${styles[`badge${routerConfig.observability?.tracing?.resource?.deployment_environment ?? ''}`]}`}>
                    {routerConfig.observability?.tracing?.resource?.deployment_environment}
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
        {routerConfig.api?.batch_classification && (
          <button
            className={styles.sectionEditButton}
            onClick={() => {
              openEditModal(
                'Edit Batch Classification API Configuration',
                routerConfig.api?.batch_classification || {},
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
        {routerConfig.api?.batch_classification ? (
          <>
            <div className={styles.featureCard}>
              <div className={styles.featureHeader}>
                <span className={styles.featureTitle}>Batch Configuration</span>
              </div>
              <div className={styles.featureBody}>
                <div className={styles.configRow}>
                  <span className={styles.configLabel}>Max Batch Size</span>
                  <span className={styles.configValue}>{routerConfig.api.batch_classification.max_batch_size}</span>
                </div>
                {routerConfig.api.batch_classification.concurrency_threshold !== undefined && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Concurrency Threshold</span>
                    <span className={styles.configValue}>{routerConfig.api.batch_classification.concurrency_threshold}</span>
                  </div>
                )}
                {routerConfig.api.batch_classification.max_concurrency !== undefined && (
                  <div className={styles.configRow}>
                    <span className={styles.configLabel}>Max Concurrency</span>
                    <span className={styles.configValue}>{routerConfig.api.batch_classification.max_concurrency}</span>
                  </div>
                )}
              </div>
            </div>

            {routerConfig.api?.batch_classification?.metrics && (
              <div className={styles.featureCard}>
                <div className={styles.featureHeader}>
                  <span className={styles.featureTitle}>Metrics Collection</span>
                  <span className={`${styles.statusBadge} ${routerConfig.api.batch_classification.metrics.enabled ? styles.statusActive : styles.statusInactive}`}>
                    {routerConfig.api.batch_classification.metrics.enabled ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                {routerConfig.api.batch_classification.metrics.enabled && (
                  <div className={styles.featureBody}>
                    {routerConfig.api.batch_classification.metrics.sample_rate !== undefined && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Sample Rate</span>
                        <span className={styles.configValue}>{((routerConfig.api.batch_classification.metrics.sample_rate ?? 0) * 100).toFixed(0)}%</span>
                      </div>
                    )}
                    {routerConfig.api.batch_classification.metrics.detailed_goroutine_tracking !== undefined && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>Goroutine Tracking</span>
                        <span className={styles.configValue}>{routerConfig.api.batch_classification.metrics.detailed_goroutine_tracking ? 'Yes' : 'No'}</span>
                      </div>
                    )}
                    {routerConfig.api.batch_classification.metrics.high_resolution_timing !== undefined && (
                      <div className={styles.configRow}>
                        <span className={styles.configLabel}>High Resolution Timing</span>
                        <span className={styles.configValue}>{routerConfig.api.batch_classification.metrics.high_resolution_timing ? 'Yes' : 'No'}</span>
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

    // Unified signal type
    type SignalType = 'Keywords' | 'Embeddings' | 'Domain' | 'Preference' | 'Fact Check' | 'User Feedback'

    interface UnifiedSignal {
      name: string
      type: SignalType
      summary: string
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      rawData: any
    }

    // Flatten all signals into a unified array
    const allSignals: UnifiedSignal[] = []

    // Keywords
    signals?.keywords?.forEach(kw => {
      allSignals.push({
        name: kw.name,
        type: 'Keywords',
        summary: `${kw.operator}, ${kw.keywords.length} keywords${kw.case_sensitive ? ', case-sensitive' : ''}`,
        rawData: kw
      })
    })

    // Embeddings
    signals?.embeddings?.forEach(emb => {
      allSignals.push({
        name: emb.name,
        type: 'Embeddings',
        summary: `Threshold: ${Math.round(emb.threshold * 100)}%, ${emb.candidates.length} items, ${emb.aggregation_method}`,
        rawData: emb
      })
    })

    // Domains
    signals?.domains?.forEach(domain => {
      const categoryCount = domain.mmlu_categories?.length || 0
      allSignals.push({
        name: domain.name,
        type: 'Domain',
        summary: categoryCount > 0 ? `${categoryCount} MMLU categories` : (domain.description || 'No description'),
        rawData: domain
      })
    })

    // Preferences
    signals?.preferences?.forEach(pref => {
      allSignals.push({
        name: pref.name,
        type: 'Preference',
        summary: pref.description || 'No description',
        rawData: pref
      })
    })

    // Fact Check
    signals?.fact_check?.forEach(fc => {
      allSignals.push({
        name: fc.name,
        type: 'Fact Check',
        summary: fc.description || 'No description',
        rawData: fc
      })
    })

    // User Feedbacks
    signals?.user_feedbacks?.forEach(uf => {
      allSignals.push({
        name: uf.name,
        type: 'User Feedback',
        summary: uf.description || 'No description',
        rawData: uf
      })
    })

    // Filter signals based on search
    const filteredSignals = allSignals.filter(signal =>
      signal.name.toLowerCase().includes(signalsSearch.toLowerCase()) ||
      signal.type.toLowerCase().includes(signalsSearch.toLowerCase()) ||
      signal.summary.toLowerCase().includes(signalsSearch.toLowerCase())
    )

    // Define table columns
    const signalsColumns: Column<UnifiedSignal>[] = [
      {
        key: 'name',
        header: 'Name',
        sortable: true,
        render: (row) => <span style={{ fontWeight: 600 }}>{row.name}</span>
      },
      {
        key: 'type',
        header: 'Type',
        width: '140px',
        sortable: true,
        render: (row) => {
          const typeColors: Record<SignalType, string> = {
            'Keywords': 'rgba(118, 185, 0, 0.15)',
            'Embeddings': 'rgba(0, 212, 255, 0.15)',
            'Domain': 'rgba(147, 51, 234, 0.15)',
            'Preference': 'rgba(234, 179, 8, 0.15)',
            'Fact Check': 'rgba(34, 197, 94, 0.15)',
            'User Feedback': 'rgba(236, 72, 153, 0.15)'
          }
          return (
            <span className={styles.badge} style={{ background: typeColors[row.type] }}>
              {row.type}
            </span>
          )
        }
      },
      {
        key: 'summary',
        header: 'Summary',
        render: (row) => (
          <span style={{
            fontSize: '0.875rem',
            color: 'var(--color-text-secondary)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            display: 'block'
          }}>
            {row.summary}
          </span>
        )
      }
    ]

    // Handle view signal
    const handleViewSignal = (signal: UnifiedSignal) => {
      const sections: ViewSection[] = []

      // Basic info section
      sections.push({
        title: 'Basic Information',
        fields: [
          { label: 'Name', value: signal.name },
          { label: 'Type', value: signal.type },
          { label: 'Summary', value: signal.summary, fullWidth: true }
        ]
      })

      // Type-specific details
      if (signal.type === 'Keywords') {
        sections.push({
          title: 'Keywords Configuration',
          fields: [
            { label: 'Operator', value: signal.rawData.operator },
            { label: 'Case Sensitive', value: signal.rawData.case_sensitive ? 'Yes' : 'No' },
            {
              label: 'Keywords',
              value: (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {signal.rawData.keywords.map((kw: string, i: number) => (
                    <span key={i} style={{
                      padding: '0.25rem 0.75rem',
                      background: 'rgba(118, 185, 0, 0.1)',
                      borderRadius: '4px',
                      fontSize: '0.875rem',
                      fontFamily: 'var(--font-mono)'
                    }}>
                      {kw}
                    </span>
                  ))}
                </div>
              ),
              fullWidth: true
            }
          ]
        })
      } else if (signal.type === 'Embeddings') {
        sections.push({
          title: 'Embeddings Configuration',
          fields: [
            { label: 'Threshold', value: `${Math.round(signal.rawData.threshold * 100)}%` },
            { label: 'Aggregation Method', value: signal.rawData.aggregation_method },
            {
              label: 'Candidates',
              value: (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {signal.rawData.candidates.map((c: string, i: number) => (
                    <div key={i} style={{
                      padding: '0.5rem',
                      background: 'rgba(0, 212, 255, 0.1)',
                      borderRadius: '4px',
                      fontSize: '0.875rem'
                    }}>
                      {c}
                    </div>
                  ))}
                </div>
              ),
              fullWidth: true
            }
          ]
        })
      } else if (signal.type === 'Domain') {
        sections.push({
          title: 'Domain Configuration',
          fields: [
            { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
            {
              label: 'MMLU Categories',
              value: signal.rawData.mmlu_categories?.length ? (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {signal.rawData.mmlu_categories.map((cat: string, i: number) => (
                    <span key={i} style={{
                      padding: '0.25rem 0.75rem',
                      background: 'rgba(147, 51, 234, 0.1)',
                      borderRadius: '4px',
                      fontSize: '0.875rem'
                    }}>
                      {cat}
                    </span>
                  ))}
                </div>
              ) : 'No categories',
              fullWidth: true
            }
          ]
        })
      } else {
        // Preference, Fact Check, User Feedback
        sections.push({
          title: 'Details',
          fields: [
            { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
          ]
        })
      }

      setViewModalTitle(`Signal: ${signal.name}`)
      setViewModalSections(sections)
      setViewModalEditCallback(() => () => handleEditSignal(signal))
      setViewModalOpen(true)
    }

    // Handle edit signal (placeholder)
    const handleEditSignal = (signal: UnifiedSignal) => {
      setViewModalOpen(false)
      // TODO: Implement edit functionality
      console.log('Edit signal:', signal)
    }

    // Handle delete signal (placeholder)
    const handleDeleteSignal = (signal: UnifiedSignal) => {
      if (confirm(`Are you sure you want to delete signal "${signal.name}"?`)) {
        // TODO: Implement delete functionality
        console.log('Delete signal:', signal)
      }
    }

    return (
      <div className={styles.sectionPanel}>
        <TableHeader
          title="Signals"
          count={allSignals.length}
          searchPlaceholder="Search signals..."
          searchValue={signalsSearch}
          onSearchChange={setSignalsSearch}
          onAdd={() => console.log('Add signal')}
          addButtonText="Add Signal"
        />

        {isPythonCLI ? (
          <DataTable
            columns={signalsColumns}
            data={filteredSignals}
            keyExtractor={(row) => `${row.type}-${row.name}`}
            onView={handleViewSignal}
            onEdit={handleEditSignal}
            onDelete={handleDeleteSignal}
            emptyMessage={signalsSearch ? 'No signals match your search' : 'No signals configured'}
          />
        ) : (
          <div className={styles.emptyState}>
            Signals are only available in Python CLI config format.
            Current config uses legacy format - use "Intelligent Routing" features instead.
          </div>
        )}
      </div>
    )
  }

  // Decisions Section - Routing rules with priorities (config.yaml)
  const renderDecisionsSection = () => {
    const decisions = config?.decisions || []
    const defaultModel = getDefaultModel()

    // Filter decisions based on search
    const filteredDecisions = decisions.filter(decision =>
      decision.name.toLowerCase().includes(decisionsSearch.toLowerCase()) ||
      decision.description?.toLowerCase().includes(decisionsSearch.toLowerCase())
    )

    // Define table columns
    type DecisionRow = NonNullable<ConfigData['decisions']>[number]
    const decisionsColumns: Column<DecisionRow>[] = [
      {
        key: 'name',
        header: 'Name',
        sortable: true,
        render: (row) => <span style={{ fontWeight: 600 }}>{row.name}</span>
      },
      {
        key: 'priority',
        header: 'Priority',
        width: '100px',
        align: 'center',
        sortable: true,
        render: (row) => (
          <span className={styles.badge} style={{ background: 'rgba(0, 212, 255, 0.15)', color: 'var(--color-accent-cyan)' }}>
            P{row.priority}
          </span>
        )
      },
      {
        key: 'conditions',
        header: 'Conditions',
        width: '150px',
        render: (row) => {
          const count = row.rules?.conditions?.length || 0
          return <span>{count} {count === 1 ? 'condition' : 'conditions'}</span>
        }
      },
      {
        key: 'models',
        header: 'Models',
        width: '150px',
        render: (row) => {
          const count = row.modelRefs?.length || 0
          return <span>{count} {count === 1 ? 'model' : 'models'}</span>
        }
      }
    ]

    // Handle view decision
    const handleViewDecision = (decision: DecisionRow) => {
      const sections: ViewSection[] = [
        {
          title: 'Basic Information',
          fields: [
            { label: 'Name', value: decision.name },
            { label: 'Priority', value: `P${decision.priority}` },
            { label: 'Description', value: decision.description || 'N/A', fullWidth: true }
          ]
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
                    <div key={i} style={{
                      padding: '0.5rem',
                      background: 'rgba(118, 185, 0, 0.1)',
                      borderRadius: '4px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.875rem'
                    }}>
                      {cond.type}: {cond.name}
                    </div>
                  ))}
                </div>
              ) : 'No conditions',
              fullWidth: true
            }
          ]
        },
        {
          title: 'Models',
          fields: [
            {
              label: 'Model References',
              value: decision.modelRefs?.length ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {decision.modelRefs.map((ref, i) => (
                    <div key={i} style={{
                      padding: '0.5rem',
                      background: 'rgba(0, 212, 255, 0.1)',
                      borderRadius: '4px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.875rem'
                    }}>
                      {ref.model} {ref.use_reasoning && '(with reasoning)'}
                    </div>
                  ))}
                </div>
              ) : 'No models',
              fullWidth: true
            }
          ]
        }
      ]

      if (decision.plugins && decision.plugins.length > 0) {
        sections.push({
          title: 'Plugins',
          fields: [
            {
              label: 'Configured Plugins',
              value: (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {decision.plugins.map((plugin, i) => (
                    <div key={i} style={{
                      padding: '0.5rem',
                      background: 'rgba(147, 51, 234, 0.1)',
                      borderRadius: '4px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.875rem'
                    }}>
                      {plugin.type}
                    </div>
                  ))}
                </div>
              ),
              fullWidth: true
            }
          ]
        })
      }

      setViewModalTitle(`Decision: ${decision.name}`)
      setViewModalSections(sections)
      setViewModalEditCallback(() => () => handleEditDecision(decision))
      setViewModalOpen(true)
    }

    // Handle edit decision (placeholder for now)
    const handleEditDecision = (decision: DecisionRow) => {
      setViewModalOpen(false)
      // TODO: Implement edit functionality
      console.log('Edit decision:', decision)
    }

    // Handle delete decision (placeholder for now)
    const handleDeleteDecision = (decision: DecisionRow) => {
      if (confirm(`Are you sure you want to delete decision "${decision.name}"?`)) {
        // TODO: Implement delete functionality
        console.log('Delete decision:', decision)
      }
    }

    return (
      <div className={styles.sectionPanel}>
        {/* Default Model Info */}
        <div className={styles.coreSettingsInline}>
          <div className={styles.inlineConfigRow}>
            <span className={styles.inlineConfigLabel}>Default Model:</span>
            <span className={styles.inlineConfigValue}>{defaultModel || 'N/A'}</span>
          </div>
          <div className={styles.inlineConfigRow}>
            <span className={styles.inlineConfigLabel}>Default Reasoning:</span>
            <span className={`${styles.badge} ${styles[`badge${config?.providers?.default_reasoning_effort || 'medium'}`]}`}>
              {config?.providers?.default_reasoning_effort || 'medium'}
            </span>
          </div>
        </div>

        {/* Decisions Table */}
        <TableHeader
          title="Routing Decisions"
          count={decisions.length}
          searchPlaceholder="Search decisions..."
          searchValue={decisionsSearch}
          onSearchChange={setDecisionsSearch}
          onAdd={() => console.log('Add decision')}
          addButtonText="Add Decision"
        />

        {isPythonCLI ? (
          <DataTable
            columns={decisionsColumns}
            data={filteredDecisions}
            keyExtractor={(row) => row.name}
            onView={handleViewDecision}
            onEdit={handleEditDecision}
            onDelete={handleDeleteDecision}
            emptyMessage={decisionsSearch ? 'No decisions match your search' : 'No routing decisions configured'}
          />
        ) : (
          <div className={styles.emptyState}>
            Decisions are only available in Python CLI config format.
            Current config uses legacy format - see "Categories" in legacy mode.
          </div>
        )}

        {/* Reasoning Families */}
        {renderReasoningFamilies()}
      </div>
    )
  }

  // Models Section - Provider models and endpoints (config.yaml)
  const renderModelsSection = () => {
    const models = getModels()

    // Filter models based on search
    const filteredModels = models.filter(model =>
      model.name.toLowerCase().includes(modelsSearch.toLowerCase()) ||
      model.reasoning_family?.toLowerCase().includes(modelsSearch.toLowerCase())
    )

    // Define model columns
    type ModelRow = NormalizedModel
    const modelColumns: Column<ModelRow>[] = [
      {
        key: 'name',
        header: 'Model Name',
        sortable: true,
        render: (row) => (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontWeight: 600 }}>{row.name}</span>
            {row.name === getDefaultModel() && (
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
        width: '180px',
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
        width: '120px',
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
        width: '150px',
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

    // Render expanded row (endpoints table)
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
                    {ep.endpoint || 'N/A'}
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
                      {ep.protocol || 'http'}
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

    // Handle view model
    const handleViewModel = (model: ModelRow) => {
      const sections: ViewSection[] = [
        {
          title: 'Basic Information',
          fields: [
            { label: 'Model Name', value: model.name },
            { label: 'Reasoning Family', value: model.reasoning_family || 'N/A' },
            { label: 'Is Default', value: model.name === getDefaultModel() ? 'Yes' : 'No' }
          ]
        }
      ]

      if (model.endpoints && model.endpoints.length > 0) {
        sections.push({
          title: `Endpoints (${model.endpoints.length})`,
          fields: [
            {
              label: 'Configured Endpoints',
              value: (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {model.endpoints.map((ep, i) => {
                    const isHttps = ep.protocol === 'https'
                    return (
                      <div key={i} style={{
                        padding: '0.75rem',
                        background: 'rgba(0, 212, 255, 0.05)',
                        border: '1px solid rgba(0, 212, 255, 0.15)',
                        borderRadius: '4px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                            {ep.name}
                          </div>
                          <div style={{
                            fontSize: '0.875rem',
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--color-text-secondary)'
                          }}>
                            {ep.endpoint}
                          </div>
                        </div>
                        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                          <span style={{
                            fontSize: '0.75rem',
                            textTransform: 'uppercase',
                            fontWeight: 600,
                            padding: '0.25rem 0.5rem',
                            borderRadius: '3px',
                            background: isHttps ? 'rgba(34, 197, 94, 0.15)' : 'rgba(234, 179, 8, 0.15)',
                            color: isHttps ? 'rgb(34, 197, 94)' : 'rgb(234, 179, 8)'
                          }}>
                            {isHttps ? 'HTTPS' : 'HTTP'}
                          </span>
                          <span style={{
                            fontSize: '0.875rem',
                            color: 'var(--color-text-secondary)'
                          }}>
                            Weight: {ep.weight}
                          </span>
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

      if (model.pii_policy) {
        sections.push({
          title: 'PII Policy',
          fields: [
            { label: 'Allow by Default', value: model.pii_policy.allow_by_default ? 'Yes' : 'No' },
            {
              label: model.pii_policy.allow_by_default ? 'Blocked Types' : 'Allowed Types',
              value: model.pii_policy.pii_types_allowed?.length ? (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {model.pii_policy.pii_types_allowed.map((type, i) => (
                    <span key={i} style={{
                      padding: '0.25rem 0.75rem',
                      background: 'rgba(236, 72, 153, 0.1)',
                      borderRadius: '4px',
                      fontSize: '0.875rem',
                      fontFamily: 'var(--font-mono)'
                    }}>
                      {type}
                    </span>
                  ))}
                </div>
              ) : 'None',
              fullWidth: true
            }
          ]
        })
      }

      if (model.access_key) {
        sections.push({
          title: 'Authentication',
          fields: [
            { label: 'Access Key', value: '••••••••' }
          ]
        })
      }

      setViewModalTitle(`Model: ${model.name}`)
      setViewModalSections(sections)
      setViewModalEditCallback(() => () => handleEditModel(model))
      setViewModalOpen(true)
    }

    // Handle add model
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const handleAddModel = () => {
      const reasoningFamiliesObj = getReasoningFamilies()
      const reasoningFamilyNames = Object.keys(reasoningFamiliesObj)

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
          model_name: '',
          reasoning_family: reasoningFamilyNames[0] || '',
          access_key: '',
          endpoints: [{
            name: 'endpoint-1',
            endpoint: 'localhost:8000',
            protocol: 'http' as const,
            weight: 1
          }],
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
            label: 'Access Key',
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
          // Endpoints are already validated by EndpointsEditor
          const endpoints = data.endpoints || []

          const newConfig = { ...config }

          if (isPythonCLI && newConfig.providers) {
            newConfig.providers = { ...newConfig.providers }
            if (!newConfig.providers.models) {
              newConfig.providers.models = []
            }
            const newModel: any = {
              name: data.model_name,
              reasoning_family: data.reasoning_family,
              access_key: data.access_key,
              endpoints: endpoints,
              pricing: {
                currency: data.currency,
                prompt_per_1m: parseFloat(data.prompt_per_1m) || 0,
                completion_per_1m: parseFloat(data.completion_per_1m) || 0
              },
              pii_policy: {
                allow_by_default: data.pii_allow_by_default,
                pii_types_allowed: data.pii_types_allowed
              }
            }
            newConfig.providers.models.push(newModel)
          } else {
            // Legacy format
            if (!newConfig.model_config) {
              newConfig.model_config = {}
            }
            newConfig.model_config[data.model_name] = {
              reasoning_family: data.reasoning_family,
              preferred_endpoints: endpoints.map((ep: any) => ep.name),
              pricing: {
                currency: data.currency,
                prompt_per_1m: parseFloat(data.prompt_per_1m) || 0,
                completion_per_1m: parseFloat(data.completion_per_1m) || 0
              },
              pii_policy: {
                allow_by_default: data.pii_allow_by_default,
                pii_types_allowed: data.pii_types_allowed
              }
            }
          }
          await saveConfig(newConfig)
        },
        'add'
      )
    }

    // Handle edit model
    const handleEditModel = (model: ModelRow) => {
      setViewModalOpen(false)

      const reasoningFamiliesObj = getReasoningFamilies()
      const reasoningFamilyNames = Object.keys(reasoningFamiliesObj)

      // PII types
      const piiTypes = [
        'AGE', 'CREDIT_CARD', 'DATE_TIME', 'DOMAIN_NAME',
        'EMAIL_ADDRESS', 'GPE', 'IBAN_CODE', 'IP_ADDRESS',
        'NO_PII', 'NRP', 'ORGANIZATION', 'PERSON',
        'PHONE_NUMBER', 'STREET_ADDRESS', 'US_DRIVER_LICENSE',
        'US_SSN', 'ZIP_CODE'
      ]

      openEditModal(
        `Edit Model: ${model.name}`,
        {
          reasoning_family: model.reasoning_family || '',
          access_key: model.access_key || '',
          // Endpoints
          endpoints: model.endpoints || [],
          // Pricing
          currency: model.pricing?.currency || 'USD',
          prompt_per_1m: model.pricing?.prompt_per_1m || 0,
          completion_per_1m: model.pricing?.completion_per_1m || 0,
          // PII Policy
          pii_allow_by_default: model.pii_policy?.allow_by_default ?? true,
          pii_types_allowed: model.pii_policy?.pii_types_allowed || []
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
            label: 'Access Key',
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

          // Endpoints are already validated by EndpointsEditor
          const endpoints = data.endpoints || []

          if (isPythonCLI && newConfig.providers?.models) {
            newConfig.providers = { ...newConfig.providers }
            type ModelType = NonNullable<ConfigData['providers']>['models'][number]
            newConfig.providers.models = newConfig.providers.models.map((m: ModelType) =>
              m.name === model.name ? {
                ...m,
                reasoning_family: data.reasoning_family,
                access_key: data.access_key,
                endpoints: endpoints,
                pricing: {
                  currency: data.currency,
                  prompt_per_1m: parseFloat(data.prompt_per_1m) || 0,
                  completion_per_1m: parseFloat(data.completion_per_1m) || 0
                },
                pii_policy: {
                  allow_by_default: data.pii_allow_by_default,
                  pii_types_allowed: data.pii_types_allowed
                }
              } : m
            )
          } else if (newConfig.model_config) {
            // Legacy format
            newConfig.model_config[model.name] = {
              ...newConfig.model_config[model.name],
              reasoning_family: data.reasoning_family,
              preferred_endpoints: endpoints.map((ep: any) => ep.name),
              pricing: {
                currency: data.currency,
                prompt_per_1m: parseFloat(data.prompt_per_1m) || 0,
                completion_per_1m: parseFloat(data.completion_per_1m) || 0
              },
              pii_policy: {
                allow_by_default: data.pii_allow_by_default,
                pii_types_allowed: data.pii_types_allowed
              }
            }
          }
          await saveConfig(newConfig)
        },
        'edit'
      )
    }

    // Handle delete model
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const handleDeleteModel = (model: ModelRow) => {
      if (confirm(`Are you sure you want to delete model "${model.name}"?`)) {
        handleDeleteModelAction(model.name)
      }
    }

    const handleDeleteModelAction = async (modelName: string) => {
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

    // Toggle expand
    const handleToggleExpand = (model: ModelRow) => {
      const newExpanded = new Set(expandedModels)
      if (newExpanded.has(model.name)) {
        newExpanded.delete(model.name)
      } else {
        newExpanded.add(model.name)
      }
      setExpandedModels(newExpanded)
    }

    return (
      <div className={styles.sectionPanel}>
        {/* Models Table */}
        <TableHeader
          title="Models"
          count={models.length}
          searchPlaceholder="Search models..."
          searchValue={modelsSearch}
          onSearchChange={setModelsSearch}
          onAdd={handleAddModel}
          addButtonText="Add Model"
        />

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
        />
      </div>
    )
  }

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
          <div className={styles.contentArea}>
            {renderActiveSection()}
          </div>
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

      {/* View Modal */}
      <ViewModal
        isOpen={viewModalOpen}
        onClose={() => setViewModalOpen(false)}
        onEdit={viewModalEditCallback || undefined}
        title={viewModalTitle}
        sections={viewModalSections}
      />
    </div>
  )
}

export default ConfigPage
