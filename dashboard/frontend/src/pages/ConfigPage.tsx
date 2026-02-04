import React, { useState, useEffect } from 'react'
import styles from './ConfigPage.module.css'
import { ConfigSection } from '../components/ConfigNav'
import EditModal, { FieldConfig } from '../components/EditModal'
import ViewModal, { ViewSection } from '../components/ViewModal'
import { DataTable, Column } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import EndpointsEditor, { Endpoint } from '../components/EndpointsEditor'
import { useReadonly } from '../contexts/ReadonlyContext'
import {
  ConfigFormat,
  detectConfigFormat,
  DecisionConditionType
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

interface ModelConfigEntry {
  reasoning_family?: string
  preferred_endpoints?: string[]
  pricing?: ModelPricing
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
    language?: Array<{ name: string }>
    latency?: Array<{ name: string; tpot_percentile?: number; ttft_percentile?: number; description?: string }>
    context?: Array<{ name: string; min_tokens: string; max_tokens: string; description?: string }>
    complexity?: Array<{
      name: string
      threshold: number
      hard: { candidates: string[] }
      easy: { candidates: string[] }
      description?: string
      composer?: {
        operator: 'AND' | 'OR'
        conditions: Array<{ type: string; name: string }>
      }
    }>
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

type SignalType = 'Keywords' | 'Embeddings' | 'Domain' | 'Preference' | 'Fact Check' | 'User Feedback' | 'Language' | 'Latency' | 'Context' | 'Complexity'
type DecisionConfig = NonNullable<ConfigData['decisions']>[number]

interface DecisionFormState {
  name: string
  description: string
  priority: number
  operator: 'AND' | 'OR'
  conditions: { type: string; name: string }[]
  modelRefs: { model: string; use_reasoning: boolean }[]
  plugins: { type: string; configuration: string | Record<string, unknown> }[]
}

interface AddSignalFormState {
  type: SignalType
  name: string
  description: string
  operator: 'AND' | 'OR'
  keywords: string
  case_sensitive: boolean
  threshold: number
  candidates: string
  aggregation_method: string
  mmlu_categories: string
  tpot_percentile?: number
  ttft_percentile?: number
  min_tokens?: string
  max_tokens?: string
  complexity_threshold?: number
  hard_candidates?: string
  easy_candidates?: string
  composer_operator?: 'AND' | 'OR'
  composer_conditions?: string
}

// Helper function to format threshold as percentage
const formatThreshold = (value: number): string => {
  return `${Math.round(value * 100)}%`
}

// Removed maskAddress - no longer needed after removing endpoint visibility toggle

const ConfigPage: React.FC<ConfigPageProps> = ({ activeSection = 'signals' }) => {
  const { isReadonly } = useReadonly()
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [configFormat, setConfigFormat] = useState<ConfigFormat>('python-cli')

  // Router defaults state (from .vllm-sr/router-defaults.yaml)
  const [routerDefaults, setRouterDefaults] = useState<ConfigData | null>(null)

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
    // Prevent save in read-only mode
    if (isReadonly) {
      throw new Error('Dashboard is in read-only mode. Configuration editing is disabled.')
    }

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

  const listInputToArray = (input: string) => input
    .split(/[\n,]/)
    .map(item => item.trim())
    .filter(Boolean)

  const removeSignalByName = (cfg: ConfigData, type: SignalType, targetName: string) => {
    // match by type and name to remove the signal from the config
    if (!cfg.signals) cfg.signals = {}

    switch (type) {
      case 'Keywords':
        cfg.signals.keywords = (cfg.signals.keywords || []).filter(s => s.name !== targetName)
        break
      case 'Embeddings':
        cfg.signals.embeddings = (cfg.signals.embeddings || []).filter(s => s.name !== targetName)
        break
      case 'Domain':
        cfg.signals.domains = (cfg.signals.domains || []).filter(s => s.name !== targetName)
        break
      case 'Preference':
        cfg.signals.preferences = (cfg.signals.preferences || []).filter(s => s.name !== targetName)
        break
      case 'Fact Check':
        cfg.signals.fact_check = (cfg.signals.fact_check || []).filter(s => s.name !== targetName)
        break
      case 'User Feedback':
        cfg.signals.user_feedbacks = (cfg.signals.user_feedbacks || []).filter(s => s.name !== targetName)
        break
      case 'Language':
        cfg.signals.language = (cfg.signals.language || []).filter(s => s.name !== targetName)
        break
      case 'Latency':
        cfg.signals.latency = (cfg.signals.latency || []).filter(s => s.name !== targetName)
        break
      case 'Context':
        cfg.signals.context = (cfg.signals.context || []).filter(s => s.name !== targetName)
        break
      case 'Complexity':
        cfg.signals.complexity = (cfg.signals.complexity || []).filter(s => s.name !== targetName)
        break
      default:
        break
    }
  }

  const removeDecisionByName = (cfg: ConfigData, targetName: string) => {
    cfg.decisions = (cfg.decisions || []).filter(d => d.name !== targetName)
  }


  const handleDeleteDecision = async (decision: DecisionConfig) => {
    if (!confirm(`Are you sure you want to delete decision "${decision.name}"?`)) {
      return
    }

    if (!config || !isPythonCLI) {
      alert('Deleting decisions is only supported for Python CLI configs.')
      return
    }

    const newConfig: ConfigData = { ...config }
    removeDecisionByName(newConfig, decision.name)
    await saveConfig(newConfig)
  }

  const handleCloseViewModal = () => {
    setViewModalOpen(false)
    setViewModalTitle('')
    setViewModalSections([])
    setViewModalEditCallback(null)
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
  }

  const getModels = (): NormalizedModel[] => {
    if (isPythonCLI && config?.providers?.models) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return config.providers.models.map((m: any): NormalizedModel => {
        const model: NormalizedModel = {
          name: m.name,
          reasoning_family: m.reasoning_family,
          endpoints: m.endpoints || [],
          access_key: m.access_key,
          pricing: m.pricing,
        }
        return model
      })
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
        pricing: cfg.pricing
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
        <h3 className={styles.sectionTitle}>PII Detection (ModernBERT)</h3>
        {routerConfig.classifier?.pii_model && !isReadonly && (
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
            Edit
          </button>
        )}
      </div>
      <div className={styles.sectionContent}>
        {routerConfig.classifier?.pii_model ? (
          <div className={styles.modelCard}>
            <div className={styles.modelCardHeader}>
              <span className={styles.modelCardTitle}>PII Classifier Model</span>
              <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                {routerConfig.classifier.pii_model.use_cpu ? 'CPU' : 'GPU'}
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
        <h3 className={styles.sectionTitle}>Jailbreak Detection (ModernBERT)</h3>
        {routerConfig.prompt_guard && !isReadonly && (
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
            Edit
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
                    {routerConfig.prompt_guard.use_cpu ? 'CPU' : 'GPU'}
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
          <h3 className={styles.sectionTitle}>Similarity BERT Configuration</h3>
          {routerConfig.bert_model && !isReadonly && (
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
              Edit
            </button>
          )}
        </div>
        <div className={styles.sectionContent}>
          {routerConfig.bert_model ? (
            <div className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelCardTitle}>BERT Model (Semantic Similarity)</span>
                <span className={`${styles.statusBadge} ${styles.statusActive}`}>
                  {routerConfig.bert_model.use_cpu ? 'CPU' : 'GPU'}
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
                  {!isReadonly && (
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
                    Edit
                  </button>
                  )}
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
    )
  }

  // ============================================================================
  // 4. INTELLIGENT ROUTING SECTION
  // ============================================================================

  const renderClassifyBERT = () => {
    const hasInTree = routerConfig.classifier?.category_model
    const hasOutTree = routerConfig.classifier?.mcp_category_model?.enabled

    return (
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
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
                    {routerConfig.classifier.category_model.use_cpu ? 'CPU' : 'GPU'}
                  </span>
                  {!isReadonly && (
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

                  </button>
                  )}
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
                  {!isReadonly && (
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

                  </button>
                  )}
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
          <h3 className={styles.sectionTitle}>{isPythonCLI ? 'Domains & Decisions' : 'Categories Configuration'}</h3>
          <span className={styles.badge}>{domains.length} {isPythonCLI ? 'domains' : 'categories'}</span>
        </div>
        <div className={styles.sectionContent}>
          {/* Core Settings at the top */}
          <div className={styles.coreSettingsInline}>
            <div className={styles.inlineConfigRow}>
              <span className={styles.inlineConfigLabel}>Default Model:</span>
              <span className={styles.inlineConfigValue}>{defaultModel || 'N/A'}</span>
            </div>
            {!isPythonCLI && (
              <div className={styles.inlineConfigRow}>
                <span className={styles.inlineConfigLabel}>Default Reasoning Effort:</span>
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
              <h4 className={styles.subsectionTitle}>Domains</h4>
              {domains.length > 0 ? (
                <div className={styles.categoryGridTwoColumn}>
                  {domains.map((domain, index) => (
                    <div key={index} className={styles.categoryCard}>
                      <div className={styles.categoryHeader}>
                        <span className={styles.categoryName}>{domain.name}</span>
                        {!isReadonly && (
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

                          </button>
                        )}
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
              <h4 className={styles.subsectionTitle}>Decisions (Routing Rules)</h4>
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
                                {ref.model} {ref.use_reasoning && ''}
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
                              {reasoningEffort}
                            </span>
                          )}
                          {!isReadonly && (
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

                            </button>
                          )}
                        </div>
                      </div>

                      {/* System Prompt */}
                      {category.system_prompt && (
                        <div className={styles.systemPromptSection}>
                          <div className={styles.systemPromptLabel}>System Prompt</div>
                          <div className={styles.systemPromptText}>{category.system_prompt}</div>
                        </div>
                      )}

                      {reasoningDescription && (
                        <p className={styles.categoryDescription}>{reasoningDescription}</p>
                      )}

                      <div className={styles.categoryModels}>
                        <div className={styles.categoryModelsHeader}>
                          <span>Model Scores</span>
                          {!isReadonly && (
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

                            </button>
                          )}
                        </div>
                        {normalizedScores.length > 0 ? (
                          normalizedScores.map((modelScore, modelIdx) => (
                            <div key={modelIdx} className={styles.modelScoreRow}>
                              <span className={styles.modelScoreName}>
                                {modelScore.model}
                                {modelScore.use_reasoning && <span className={styles.reasoningIcon}></span>}
                              </span>
                              <div className={styles.scoreBar}>
                                <div
                                  className={styles.scoreBarFill}
                                  style={{ width: `${(modelScore.score ?? 0) * 100}%` }}
                                ></div>
                                <span className={styles.scoreText}>{((modelScore.score ?? 0) * 100).toFixed(0)}%</span>
                              </div>
                              <div className={styles.modelScoreActions}>
                                {!isReadonly && (
                                  <>
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

                                    </button>
                                  </>
                                )}
                              </div>
                            </div>
                          ))
                        ) : (
                          <div className={styles.emptyModelScores}>No models configured for this category</div>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className={styles.emptyState}>No categories configured</div>
            )
          )}
        </div>
      </div>
    )
  }



  // ============================================================================
  // 5. TOOLS SELECTION SECTION
  // ============================================================================

  const renderToolsConfiguration = () => (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>Tools Configuration</h3>
        {routerConfig.tools && !isReadonly && (
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
            Edit
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
                        <div className={styles.similarityDescriptionLabel}>Similarity Keywords</div>
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
        <h3 className={styles.sectionTitle}>Distributed Tracing</h3>
        {routerConfig.observability?.tracing && !isReadonly && (
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
            Edit
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
        <h3 className={styles.sectionTitle}>Batch Classification API</h3>
        {routerConfig.api?.batch_classification && !isReadonly && (
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
            Edit
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

    // Language
    signals?.language?.forEach(lang => {
      allSignals.push({
        name: lang.name,
        type: 'Language',
        summary: 'Language detection rule',
        rawData: lang
      })
    })

    // Latency
    signals?.latency?.forEach(lat => {
      const parts: string[] = []
      if (lat.tpot_percentile) {
        parts.push(`TPOT: ${lat.tpot_percentile}th percentile`)
      }
      if (lat.ttft_percentile) {
        parts.push(`TTFT: ${lat.ttft_percentile}th percentile`)
      }
      allSignals.push({
        name: lat.name,
        type: 'Latency',
        summary: parts.length > 0 ? parts.join(', ') : 'Latency signal',
        rawData: lat
      })
    })

    // Context
    signals?.context?.forEach(ctx => {
      allSignals.push({
        name: ctx.name,
        type: 'Context',
        summary: `${ctx.min_tokens} to ${ctx.max_tokens} tokens`,
        rawData: ctx
      })
    })

    // Complexity
    signals?.complexity?.forEach(comp => {
      const hardCount = comp.hard?.candidates?.length || 0
      const easyCount = comp.easy?.candidates?.length || 0
      allSignals.push({
        name: comp.name,
        type: 'Complexity',
        summary: `Threshold: ${comp.threshold}, ${hardCount} hard / ${easyCount} easy candidates`,
        rawData: comp
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
            'User Feedback': 'rgba(236, 72, 153, 0.15)',
            'Language': 'rgba(59, 130, 246, 0.15)',
            'Latency': 'rgba(168, 85, 247, 0.15)',
            'Context': 'rgba(251, 146, 60, 0.15)',
            'Complexity': 'rgba(66, 153, 225, 0.15)'
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
      } else if (signal.type === 'Language') {
        sections.push({
          title: 'Language Signal',
          fields: [
            { label: 'Language Code', value: signal.rawData.name || 'N/A', fullWidth: true },
            { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
          ]
        })
      } else if (signal.type === 'Latency') {
        sections.push({
          title: 'Latency Signal',
          fields: [
            { label: 'TPOT Percentile', value: signal.rawData.tpot_percentile ? `${signal.rawData.tpot_percentile}th percentile` : 'N/A', fullWidth: true },
            { label: 'TTFT Percentile', value: signal.rawData.ttft_percentile ? `${signal.rawData.ttft_percentile}th percentile` : 'N/A', fullWidth: true },
            { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
          ]
        })
      } else if (signal.type === 'Context') {
        sections.push({
          title: 'Context Signal',
          fields: [
            { label: 'Min Tokens', value: signal.rawData.min_tokens || 'N/A', fullWidth: true },
            { label: 'Max Tokens', value: signal.rawData.max_tokens || 'N/A', fullWidth: true },
            { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
          ]
        })
      } else if (signal.type === 'Complexity') {
        const fields: Array<{ label: string; value: React.ReactNode; fullWidth?: boolean }> = [
          { label: 'Threshold', value: signal.rawData.threshold?.toString() || 'N/A', fullWidth: true }
        ]

        // Add composer if present
        if (signal.rawData.composer) {
          fields.push({
            label: 'Composer',
            value: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div><strong>Operator:</strong> {signal.rawData.composer.operator}</div>
                <div><strong>Conditions:</strong></div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginLeft: '1rem' }}>
                  {signal.rawData.composer.conditions.map((cond: { type: string; name: string }, i: number) => (
                    <div key={i} style={{
                      padding: '0.5rem',
                      background: 'rgba(255, 165, 0, 0.1)',
                      borderRadius: '4px',
                      fontSize: '0.875rem',
                      fontFamily: 'var(--font-mono)'
                    }}>
                      {cond.type}: {cond.name}
                    </div>
                  ))}
                </div>
              </div>
            ),
            fullWidth: true
          })
        }

        fields.push(
          {
            label: 'Hard Candidates',
            value: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                {(signal.rawData.hard?.candidates || []).map((c: string, i: number) => (
                  <div key={i}>• {c}</div>
                ))}
              </div>
            ),
            fullWidth: true
          },
          {
            label: 'Easy Candidates',
            value: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                {(signal.rawData.easy?.candidates || []).map((c: string, i: number) => (
                  <div key={i}>• {c}</div>
                ))}
              </div>
            ),
            fullWidth: true
          },
          { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
        )

        sections.push({
          title: 'Complexity Signal',
          fields
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

    const openSignalEditor = (mode: 'add' | 'edit', signal?: UnifiedSignal) => {
      setViewModalOpen(false)
      const defaultForm: AddSignalFormState = {
        type: 'Keywords',
        name: '',
        description: '',
        operator: 'AND',
        keywords: '',
        case_sensitive: false,
        threshold: 0.8,
        candidates: '',
        aggregation_method: 'mean',
        mmlu_categories: '',
        tpot_percentile: undefined,
        ttft_percentile: undefined,
        min_tokens: '0',
        max_tokens: '8K',
        complexity_threshold: 0.1,
        hard_candidates: '',
        easy_candidates: '',
        composer_operator: 'AND',
        composer_conditions: ''
      }

      const initialData: AddSignalFormState = mode === 'edit' && signal ? {
        type: signal.type,
        name: signal.name,
        description: signal.rawData.description || '',
        operator: signal.rawData.operator || 'AND',
        keywords: (signal.rawData.keywords || []).join('\n'),
        case_sensitive: !!signal.rawData.case_sensitive,
        threshold: signal.rawData.threshold ?? 0.8,
        candidates: (signal.rawData.candidates || []).join('\n'),
        aggregation_method: signal.rawData.aggregation_method || 'mean',
        mmlu_categories: (signal.rawData.mmlu_categories || []).join('\n'),
        tpot_percentile: signal.rawData.tpot_percentile,
        ttft_percentile: signal.rawData.ttft_percentile,
        min_tokens: signal.rawData.min_tokens || '0',
        max_tokens: signal.rawData.max_tokens || '8K',
        complexity_threshold: signal.rawData.threshold ?? 0.1,
        hard_candidates: (signal.rawData.hard?.candidates || []).join('\n'),
        easy_candidates: (signal.rawData.easy?.candidates || []).join('\n'),
        composer_operator: signal.rawData.composer?.operator || 'AND',
        composer_conditions: signal.rawData.composer?.conditions?.map((c: { type: string; name: string }) => `${c.type}:${c.name}`).join('\n') || ''
      } : defaultForm


      const conditionallyHideFieldExceptType = (type: SignalType) => {
        return (formData: AddSignalFormState) => formData.type !== type
      }

      const keywordFields: FieldConfig[] = [
        {
          name: 'operator',
          label: 'Operator (keywords only)',
          type: 'select',
          options: ['AND', 'OR'],
          description: 'Used when type is Keywords',
          shouldHide: conditionallyHideFieldExceptType('Keywords')
        },
        {
          name: 'case_sensitive',
          label: 'Case Sensitive (keywords only)',
          type: 'boolean',
          description: 'Whether keyword matching is case sensitive',
          shouldHide: conditionallyHideFieldExceptType('Keywords')
        },
        {
          name: 'keywords',
          label: 'Keywords',
          type: 'textarea',
          placeholder: 'Comma or newline separated keywords',
          shouldHide: conditionallyHideFieldExceptType('Keywords')
        },
      ]


      const embeddingFields: FieldConfig[] = [{
        name: 'threshold',
        label: 'Threshold (embeddings only)',
        type: 'number',
        min: 0,
        max: 1,
        step: 0.01,
        placeholder: '0.80',
        shouldHide: conditionallyHideFieldExceptType('Embeddings')
      },
      {
        name: 'aggregation_method',
        label: 'Aggregation Method (embeddings only)',
        type: 'text',
        placeholder: 'mean',
        shouldHide: conditionallyHideFieldExceptType('Embeddings')
      },
      {
        name: 'candidates',
        label: 'Candidates (embeddings only)',
        type: 'textarea',
        placeholder: 'One candidate per line or comma separated',
        shouldHide: conditionallyHideFieldExceptType('Embeddings')
      }]

      const domainFields: FieldConfig[] = [
        {
          name: 'mmlu_categories',
          label: 'MMLU Categories (domains only)',
          type: 'textarea',
          placeholder: 'Comma or newline separated categories',
          shouldHide: conditionallyHideFieldExceptType('Domain')
        }
      ]

      const latencyFields: FieldConfig[] = [
        {
          name: 'tpot_percentile',
          label: 'TPOT Percentile (latency only)',
          type: 'number',
          min: 1,
          max: 100,
          step: 1,
          placeholder: '10',
          description: 'TPOT (Time Per Output Token) percentile bucket (1-100). Example: 10 = 10th percentile (top 10% fastest TPOT). Works with any number of observations, adapts to model performance. ⚠️ RECOMMENDED: Use both TPOT and TTFT for comprehensive latency evaluation.',
          shouldHide: conditionallyHideFieldExceptType('Latency')
        },
        {
          name: 'ttft_percentile',
          label: 'TTFT Percentile (latency only)',
          type: 'number',
          min: 1,
          max: 100,
          step: 1,
          placeholder: '10',
          description: 'TTFT (Time To First Token) percentile bucket (1-100). Example: 10 = 10th percentile (top 10% fastest TTFT). Works with any number of observations, adapts to model performance. ⚠️ RECOMMENDED: Use both TPOT and TTFT for comprehensive latency evaluation. At least one of TPOT or TTFT percentile must be set.',
          shouldHide: conditionallyHideFieldExceptType('Latency')
        }
      ]

      const contextFields: FieldConfig[] = [
        {
          name: 'min_tokens',
          label: 'Minimum Tokens (context only)',
          type: 'text',
          placeholder: 'e.g., 0, 8K, 1M',
          description: 'Minimum token count (supports K/M suffixes)',
          shouldHide: conditionallyHideFieldExceptType('Context')
        },
        {
          name: 'max_tokens',
          label: 'Maximum Tokens (context only)',
          type: 'text',
          placeholder: 'e.g., 8K, 1024K',
          description: 'Maximum token count (supports K/M suffixes)',
          shouldHide: conditionallyHideFieldExceptType('Context')
        }
      ]

      const complexityFields: FieldConfig[] = [
        {
          name: 'complexity_threshold',
          label: 'Threshold (complexity only)',
          type: 'number',
          placeholder: 'e.g., 0.1',
          description: 'Similarity difference threshold for hard/easy classification',
          shouldHide: conditionallyHideFieldExceptType('Complexity')
        },
        {
          name: 'composer_operator',
          label: 'Composer Operator (complexity only)',
          type: 'select',
          options: ['AND', 'OR'],
          description: 'Logical operator for composer conditions (recommended to filter based on other signals)',
          shouldHide: conditionallyHideFieldExceptType('Complexity')
        },
        {
          name: 'composer_conditions',
          label: 'Composer Conditions (complexity only)',
          type: 'textarea',
          placeholder: 'One condition per line in format type:name, e.g.:\ndomain:computer_science\nkeyword:coding',
          description: 'Filter this complexity signal based on other signals (RECOMMENDED). Format: type:name per line',
          shouldHide: conditionallyHideFieldExceptType('Complexity')
        },
        {
          name: 'hard_candidates',
          label: 'Hard Candidates (complexity only)',
          type: 'textarea',
          placeholder: 'One candidate per line, e.g.:\ndesign distributed system\nimplement consensus algorithm',
          description: 'Phrases representing hard/complex queries',
          shouldHide: conditionallyHideFieldExceptType('Complexity')
        },
        {
          name: 'easy_candidates',
          label: 'Easy Candidates (complexity only)',
          type: 'textarea',
          placeholder: 'One candidate per line, e.g.:\nprint hello world\nloop through array',
          description: 'Phrases representing easy/simple queries',
          shouldHide: conditionallyHideFieldExceptType('Complexity')
        }
      ]

      const fields: FieldConfig[] = [
        {
          name: 'type',
          label: 'Type',
          type: 'select',
          options: ['Keywords', 'Embeddings', 'Domain', 'Preference', 'Fact Check', 'User Feedback', 'Language', 'Latency', 'Context', 'Complexity'],
          required: true,
          description: 'Fields are validated based on the selected type.'
        },
        {
          name: 'name',
          label: 'Name',
          type: 'text',
          required: true,
          placeholder: 'Enter a unique signal name here'
        },
        {
          name: 'description',
          label: 'Description',
          type: 'textarea',
          placeholder: 'Optional description for this signal'
        },
        ...keywordFields,
        ...embeddingFields,
        ...domainFields,
        ...latencyFields,
        ...contextFields,
        ...complexityFields,
      ]

      const saveSignal = async (formData: AddSignalFormState) => {
        if (!config) {
          throw new Error('Configuration not loaded yet.')
        }

        if (!isPythonCLI) {
          throw new Error('Editing signals is only supported for Python CLI configs.')
        }

        const name = (formData.name || '').trim()
        if (!name) {
          throw new Error('Name is required.')
        }

        const type = formData.type as SignalType
        if (!type) {
          throw new Error('Type is required.')
        }

        const newConfig: ConfigData = { ...config }
        if (!newConfig.signals) newConfig.signals = {}

        if (mode === 'edit' && signal) {
          removeSignalByName(newConfig, signal.type, signal.name)
        }

        // type specific validations
        switch (type) {
          case 'Keywords': {
            const keywords = listInputToArray(formData.keywords || '')
            if (keywords.length === 0) {
              throw new Error('Please provide at least one keyword.')
            }
            newConfig.signals.keywords = [
              ...(newConfig.signals.keywords || []),
              {
                name,
                operator: formData.operator,
                keywords,
                case_sensitive: !!formData.case_sensitive
              }
            ]
            break
          }
          case 'Embeddings': {
            const candidates = listInputToArray(formData.candidates || '')
            if (candidates.length === 0) {
              throw new Error('Please provide at least one candidate string.')
            }
            const threshold = Number.isFinite(formData.threshold)
              ? Math.max(0, Math.min(1, formData.threshold))
              : 0
            newConfig.signals.embeddings = [
              ...(newConfig.signals.embeddings || []),
              {
                name,
                threshold,
                candidates,
                aggregation_method: formData.aggregation_method || 'mean'
              }
            ]
            break
          }
          case 'Domain': {
            const mmlu_categories = listInputToArray(formData.mmlu_categories || '')
            newConfig.signals.domains = [
              ...(newConfig.signals.domains || []),
              {
                name,
                description: formData.description,
                mmlu_categories
              }
            ]
            break
          }
          case 'Preference': {
            newConfig.signals.preferences = [
              ...(newConfig.signals.preferences || []),
              {
                name,
                description: formData.description
              }
            ]
            break
          }
          case 'Fact Check': {
            newConfig.signals.fact_check = [
              ...(newConfig.signals.fact_check || []),
              {
                name,
                description: formData.description
              }
            ]
            break
          }
          case 'User Feedback': {
            newConfig.signals.user_feedbacks = [
              ...(newConfig.signals.user_feedbacks || []),
              {
                name,
                description: formData.description
              }
            ]
            break
          }
          case 'Language': {
            newConfig.signals.language = [
              ...(newConfig.signals.language || []),
              {
                name
              }
            ]
            break
          }
          case 'Latency': {
            const tpot_percentile = formData.tpot_percentile
            const ttft_percentile = formData.ttft_percentile

            // Validate: at least one of tpot_percentile or ttft_percentile must be set
            if ((tpot_percentile === undefined || tpot_percentile <= 0) && (ttft_percentile === undefined || ttft_percentile <= 0)) {
              throw new Error('Either TPOT Percentile or TTFT Percentile must be set for latency signals (or both)')
            }
            if (tpot_percentile !== undefined && (tpot_percentile < 1 || tpot_percentile > 100)) {
              throw new Error('TPOT Percentile must be between 1 and 100')
            }
            if (ttft_percentile !== undefined && (ttft_percentile < 1 || ttft_percentile > 100)) {
              throw new Error('TTFT Percentile must be between 1 and 100')
            }

            const latencySignal: { name: string; description?: string; tpot_percentile?: number; ttft_percentile?: number } = {
              name,
            }
            if (formData.description) {
              latencySignal.description = formData.description
            }
            if (tpot_percentile !== undefined && tpot_percentile > 0) {
              latencySignal.tpot_percentile = tpot_percentile
            }
            if (ttft_percentile !== undefined && ttft_percentile > 0) {
              latencySignal.ttft_percentile = ttft_percentile
            }

            newConfig.signals.latency = [
              ...(newConfig.signals.latency || []),
              latencySignal
            ]
            break
          }
          case 'Context': {
            const min_tokens = (formData.min_tokens || '0').trim()
            const max_tokens = (formData.max_tokens || '8K').trim()
            if (!min_tokens || !max_tokens) {
              throw new Error('Both min_tokens and max_tokens are required.')
            }
            newConfig.signals.context = [
              ...(newConfig.signals.context || []),
              {
                name,
                min_tokens,
                max_tokens,
                description: formData.description || undefined
              }
            ]
            break
          }
          case 'Complexity': {
            const complexity_threshold = formData.complexity_threshold ?? 0.1
            const hard_candidates = (formData.hard_candidates || '').trim()
            const easy_candidates = (formData.easy_candidates || '').trim()

            if (!hard_candidates || !easy_candidates) {
              throw new Error('Both hard and easy candidates are required.')
            }

            const hardList = hard_candidates.split('\n').map(c => c.trim()).filter(c => c.length > 0)
            const easyList = easy_candidates.split('\n').map(c => c.trim()).filter(c => c.length > 0)

            if (hardList.length === 0 || easyList.length === 0) {
              throw new Error('Both hard and easy candidates must have at least one entry.')
            }

            // Parse composer conditions if provided
            const composer_conditions_str = (formData.composer_conditions || '').trim()
            let composer = undefined
            if (composer_conditions_str) {
              const conditions = composer_conditions_str
                .split('\n')
                .map(line => line.trim())
                .filter(line => line.length > 0)
                .map(line => {
                  const parts = line.split(':')
                  if (parts.length !== 2) {
                    throw new Error(`Invalid composer condition format: "${line}". Expected format: type:name`)
                  }
                  return {
                    type: parts[0].trim(),
                    name: parts[1].trim()
                  }
                })

              if (conditions.length > 0) {
                composer = {
                  operator: formData.composer_operator || 'AND',
                  conditions
                }
              }
            }

            newConfig.signals.complexity = [
              ...(newConfig.signals.complexity || []),
              {
                name,
                threshold: complexity_threshold,
                hard: {
                  candidates: hardList
                },
                easy: {
                  candidates: easyList
                },
                description: formData.description || undefined,
                ...(composer && { composer })
              }
            ]
            break
          }
          default:
            throw new Error('Unsupported signal type.')
        }

        await saveConfig(newConfig)
      }

      openEditModal(
        mode === 'add' ? 'Add Signal' : `Edit Signal: ${signal?.name}`,
        initialData,
        fields,
        saveSignal,
        mode
      )
    }

    const handleEditSignal = (signal: UnifiedSignal) => {
      openSignalEditor('edit', signal)
    }

    // Handle delete signal
    const handleDeleteSignal = async (signal: UnifiedSignal) => {
      if (confirm(`Are you sure you want to delete signal "${signal.name}"?`)) {
        if (!config || !isPythonCLI) {
          alert('Deleting signals is only supported for Python CLI configs.')
          return
        }

        const newConfig: ConfigData = { ...config }
        removeSignalByName(newConfig, signal.type, signal.name)

        await saveConfig(newConfig)
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
          onAdd={() => openSignalEditor('add')}
          addButtonText="Add Signal"
          disabled={isReadonly}
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
            readonly={isReadonly}
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

    const openDecisionEditor = (mode: 'add' | 'edit', decision?: DecisionRow) => {
      setViewModalOpen(false)
      const conditionTypeOptions = ['keyword', 'domain', 'preference', 'user_feedback', 'embedding', 'language', 'latency'] as const

      const getConditionNameOptions = (type?: DecisionConditionType) => {
        // derive condition name options based on signals configured
        switch (type) {
          case 'keyword':
            return config?.signals?.keywords?.map((k) => k.name) || []
          case 'domain':
            return config?.signals?.domains?.map((d) => d.name) || []
          case 'preference':
            return config?.signals?.preferences?.map((p) => p.name) || []
          case 'user_feedback':
            return config?.signals?.user_feedbacks?.map((u) => u.name) || []
          case 'embedding':
            return config?.signals?.embeddings?.map((e) => e.name) || []
          case 'latency':
            return config?.signals?.latency?.map((l) => l.name) || []
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
        modelRefs: [{ model: '', use_reasoning: false }],
        plugins: []
      }

      const initialData: DecisionFormState = mode === 'edit' && decision ? {
        name: decision.name,
        description: decision.description || '',
        priority: decision.priority ?? 1,
        operator: decision.rules?.operator || 'AND',
        conditions: (decision.rules?.conditions || []).map((cond) => ({
          type: cond.type,
          name: cond.name
        })),
        modelRefs: (decision.modelRefs || []).map((ref) => ({
          model: ref.model,
          use_reasoning: !!ref.use_reasoning
        })),
        plugins: (decision.plugins || []).map((plugin) => ({
          type: plugin.type,
          configuration: JSON.stringify(plugin.configuration || {}, null, 2)
        }))
      } : defaultForm

      const renderConditionsEditor = (
        value: DecisionFormState['conditions'],
        onChange: (value: DecisionFormState['conditions']) => void
      ) => {
        const rows = (Array.isArray(value) ? value : []).length ? value : [{ type: 'keyword', name: '' }]

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
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {rows.map((cond, idx) => (
              <div
                key={idx}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr auto',
                  gap: '0.5rem',
                  alignItems: 'center'
                }}
              >
                <select
                  value={cond?.type || conditionTypeOptions[0]}
                  onChange={(e) => updateItem(idx, 'type', e.target.value)}
                  style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                >
                  {conditionTypeOptions.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
                <select
                  value={cond?.name || ''}
                  onChange={(e) => updateItem(idx, 'name', e.target.value)}
                  style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                >
                  <option value="" disabled>Select name</option>
                  {getConditionNameOptions(cond?.type as DecisionConditionType).map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                  {getConditionNameOptions(cond?.type as DecisionConditionType).length === 0 && (
                    <option value="" disabled>No matching signals</option>
                  )}
                </select>
                <button
                  type="button"
                  onClick={() => removeItem(idx)}
                  style={{
                    padding: '0.5rem 0.75rem',
                    borderRadius: 6,
                    border: '1px solid var(--color-border)',
                    background: 'transparent',
                    color: 'var(--color-text)'
                  }}
                >
                  Remove
                </button>
              </div>
            ))}
            <button
              type="button"
              onClick={addItem}
              style={{
                width: 'fit-content',
                padding: '0.5rem 0.75rem',
                borderRadius: 6,
                border: '1px solid var(--color-border)',
                background: 'transparent',
                color: 'var(--color-text)'
              }}
            >
              Add Condition
            </button>
          </div>
        )
      }

      const renderModelRefsEditor = (
        value: DecisionFormState['modelRefs'],
        onChange: (value: DecisionFormState['modelRefs']) => void
      ) => {
        const rows = (Array.isArray(value) ? value : []).length ? value : [{ model: '', use_reasoning: false }]

        const updateItem = (index: number, key: 'model' | 'use_reasoning', val: string | boolean) => {
          const next = rows.map((item, idx) => idx === index ? { ...item, [key]: val } : item)
          onChange(next)
        }

        const removeItem = (index: number) => {
          const next = rows.filter((_, idx) => idx !== index)
          onChange(next.length ? next : [{ model: '', use_reasoning: false }])
        }

        const addItem = () => onChange([...rows, { model: '', use_reasoning: false }])

        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {rows.map((ref, idx) => (
              <div
                key={idx}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr auto auto',
                  gap: '0.5rem',
                  alignItems: 'center'
                }}
              >
                <input
                  type="text"
                  value={ref?.model || ''}
                  onChange={(e) => updateItem(idx, 'model', e.target.value)}
                  placeholder="Model name (e.g. gpt-4o)"
                  style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                />
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--color-text-secondary)' }}>
                  <input
                    type="checkbox"
                    checked={!!ref?.use_reasoning}
                    onChange={(e) => updateItem(idx, 'use_reasoning', e.target.checked)}
                  />
                  Use reasoning
                </label>
                <button
                  type="button"
                  onClick={() => removeItem(idx)}
                  style={{
                    padding: '0.5rem 0.75rem',
                    borderRadius: 6,
                    border: '1px solid var(--color-border)',
                    background: 'transparent',
                    color: 'var(--color-text)'
                  }}
                >
                  Remove
                </button>
              </div>
            ))}
            <button
              type="button"
              onClick={addItem}
              style={{
                width: 'fit-content',
                padding: '0.5rem 0.75rem',
                borderRadius: 6,
                border: '1px solid var(--color-border)',
                background: 'transparent',
                color: 'var(--color-text)'
              }}
            >
              Add Model Reference
            </button>
          </div>
        )
      }

      const renderPluginsEditor = (
        value: DecisionFormState['plugins'],
        onChange: (value: DecisionFormState['plugins']) => void
      ) => {
        const rows = Array.isArray(value) ? value : []

        const updateItem = (index: number, key: 'type' | 'configuration', val: string | Record<string, unknown>) => {
          const next = rows.map((item, idx) => idx === index ? { ...item, [key]: val } : item)
          onChange(next)
        }

        const removeItem = (index: number) => {
          const next = rows.filter((_, idx) => idx !== index)
          onChange(next)
        }

        const addItem = () => onChange([...rows, { type: '', configuration: '' }])

        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {rows.map((plugin, idx) => (
              <div
                key={idx}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr',
                  gap: '0.5rem',
                  padding: '0.75rem',
                  borderRadius: 8,
                  border: '1px solid var(--color-border)'
                }}
              >
                <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '0.5rem', alignItems: 'center' }}>
                  <input
                    type="text"
                    value={plugin?.type || ''}
                    onChange={(e) => updateItem(idx, 'type', e.target.value)}
                    placeholder="Plugin type (e.g. logging)"
                    style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                  />
                  <button
                    type="button"
                    onClick={() => removeItem(idx)}
                    style={{
                      padding: '0.5rem 0.75rem',
                      borderRadius: 6,
                      border: '1px solid var(--color-border)',
                      background: 'transparent',
                      color: 'var(--color-text)'
                    }}
                  >
                    Remove
                  </button>
                </div>
                <textarea
                  value={typeof plugin?.configuration === 'string' ? plugin.configuration : JSON.stringify(plugin?.configuration || {}, null, 2)}
                  onChange={(e) => updateItem(idx, 'configuration', e.target.value)}
                  placeholder='Configuration JSON (optional)'
                  rows={4}
                  style={{
                    padding: '0.55rem 0.75rem',
                    borderRadius: 6,
                    border: '1px solid var(--color-border)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.9rem'
                  }}
                />
              </div>
            ))}
            <button
              type="button"
              onClick={addItem}
              style={{
                width: 'fit-content',
                padding: '0.5rem 0.75rem',
                borderRadius: 6,
                border: '1px solid var(--color-border)',
                background: 'transparent',
                color: 'var(--color-text)'
              }}
            >
              Add Plugin
            </button>
          </div>
        )
      }

      const fields: FieldConfig[] = [
        {
          name: 'name',
          label: 'Name',
          type: 'text',
          required: true,
          placeholder: 'Enter a unique decision name'
        },
        {
          name: 'description',
          label: 'Description',
          type: 'textarea',
          placeholder: 'What does this decision route?'
        },
        {
          name: 'priority',
          label: 'Priority',
          type: 'number',
          min: 0,
          placeholder: '1'
        },
        {
          name: 'operator',
          label: 'Rules Operator',
          type: 'select',
          options: ['AND', 'OR'],
          required: true
        },
        {
          name: 'conditions',
          label: 'Conditions',
          type: 'custom',
          description: 'Add routing conditions (type and name).',
          customRender: renderConditionsEditor
        },
        {
          name: 'modelRefs',
          label: 'Model References',
          type: 'custom',
          description: 'Set target models and whether to enable reasoning.',
          customRender: renderModelRefsEditor
        },
        {
          name: 'plugins',
          label: 'Plugins',
          type: 'custom',
          description: 'Optional plugins applied to this decision.',
          customRender: renderPluginsEditor
        }
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

        const normalizedConditions = (formData.conditions || []).filter((c) => (c?.type || '').trim() || (c?.name || '').trim())
        const conditions = normalizedConditions.map((c, idx) => {
          const type = (c?.type || '').trim()
          const name = (c?.name || '').trim()
          if (!type || !name) {
            throw new Error(`Condition #${idx + 1} needs both type and name.`)
          }
          return { type, name }
        })

        const normalizedModelRefs = (formData.modelRefs || []).filter((m) => (m?.model || '').trim())
        const modelRefs = normalizedModelRefs.map((m, idx) => {
          const model = (m?.model || '').trim()
          if (!model) {
            throw new Error(`Model reference #${idx + 1} is missing a model name.`)
          }
          return { model, use_reasoning: !!m?.use_reasoning }
        })

        const normalizedPlugins = (formData.plugins || []).filter((p) => {
          const hasType = (p?.type || '').trim()
          const hasConfigString = typeof p?.configuration === 'string' && (p.configuration as string).trim()
          const hasConfigObject = p?.configuration && typeof p.configuration === 'object'
          return hasType || hasConfigString || hasConfigObject
        })

        const plugins = normalizedPlugins.map((p, idx) => {
          const type = (p?.type || '').trim()
          if (!type) {
            throw new Error(`Plugin #${idx + 1} must include a type.`)
          }

          let configuration: Record<string, unknown> = {}
          if (typeof p?.configuration === 'string') {
            const trimmed = p.configuration.trim()
            if (trimmed) {
              try {
                configuration = JSON.parse(trimmed)
              } catch {
                throw new Error(`Plugin #${idx + 1} configuration must be valid JSON.`)
              }
            }
          } else if (p?.configuration && typeof p.configuration === 'object') {
            configuration = p.configuration as Record<string, unknown>
          }

          return { type, configuration }
        })

        const newDecision: DecisionConfig = {
          name,
          description: formData.description,
          priority: priority || 0,
          rules: {
            operator: formData.operator,
            conditions
          },
          modelRefs,
          plugins
        }

        const newConfig: ConfigData = { ...config }
        newConfig.decisions = [...(newConfig.decisions || [])]

        if (mode === 'edit' && decision) {
          removeDecisionByName(newConfig, decision.name)
        }

        newConfig.decisions.push(newDecision)
        await saveConfig(newConfig)
      }

      openEditModal(
        mode === 'add' ? 'Add Decision' : `Edit Decision: ${decision?.name}`,
        initialData,
        fields,
        saveDecision,
        mode
      )
    }

    const handleEditDecision = (decision: DecisionRow) => {
      openDecisionEditor('edit', decision)
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
          onAdd={() => openDecisionEditor('add')}
          addButtonText="Add Decision"
          disabled={isReadonly}
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
            readonly={isReadonly}
          />
        ) : (
          <div className={styles.emptyState}>
            Decisions are only available in Python CLI config format.
            Current config uses legacy format - see "Categories" in legacy mode.
          </div>
        )}
      </div>
    )
  }

  // Models Section - Provider models and endpoints (config.yaml)
  const renderModelsSection = () => {
    const models = getModels()
    const reasoningFamilies = getReasoningFamilies()

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

    // Reasoning Families handlers
    const handleViewReasoningFamily = (familyName: string) => {
      const familyConfig = reasoningFamilies[familyName]
      if (!familyConfig) return

      const sections: ViewSection[] = [
        {
          title: 'Configuration',
          fields: [
            { label: 'Family Name', value: familyName },
            { label: 'Type', value: familyConfig.type },
            { label: 'Parameter', value: familyConfig.parameter }
          ]
        }
      ]

      setViewModalTitle(`Reasoning Family: ${familyName}`)
      setViewModalSections(sections)
      setViewModalEditCallback(() => () => handleEditReasoningFamily(familyName))
      setViewModalOpen(true)
    }

    const handleEditReasoningFamily = (familyName: string) => {
      const familyConfig = reasoningFamilies[familyName]
      if (!familyConfig) return

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
          const newConfig = { ...config }
          if (isPythonCLI && newConfig.providers) {
            newConfig.providers = { ...newConfig.providers }
            if (!newConfig.providers.reasoning_families) {
              newConfig.providers.reasoning_families = {}
            }
            newConfig.providers.reasoning_families[familyName] = data
          } else if (newConfig.reasoning_families) {
            newConfig.reasoning_families[familyName] = data
          }
          await saveConfig(newConfig)
        }
      )
    }

    const handleAddReasoningFamily = () => {
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

          const newConfig = { ...config }
          if (isPythonCLI && newConfig.providers) {
            newConfig.providers = { ...newConfig.providers }
            if (!newConfig.providers.reasoning_families) {
              newConfig.providers.reasoning_families = {}
            }
            newConfig.providers.reasoning_families[familyName] = data
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
      if (!confirm(`Are you sure you want to delete reasoning family "${familyName}"?`)) {
        return
      }

      const newConfig = { ...config }
      if (isPythonCLI && newConfig.providers?.reasoning_families) {
        newConfig.providers = { ...newConfig.providers }
        newConfig.providers.reasoning_families = { ...newConfig.providers.reasoning_families }
        delete newConfig.providers.reasoning_families[familyName]
      } else if (newConfig.reasoning_families) {
        delete newConfig.reasoning_families[familyName]
      }
      await saveConfig(newConfig)
    }

    // Reasoning Families table
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
      <div className={styles.sectionPanel}>
        {/* Reasoning Families Table */}
        <TableHeader
          title="Reasoning Families"
          count={reasoningFamilyData.length}
          searchPlaceholder=""
          searchValue=""
          onSearchChange={() => { }}
          onAdd={handleAddReasoningFamily}
          addButtonText="Add Family"
          disabled={isReadonly}
        />

        <DataTable
          columns={reasoningFamilyColumns}
          data={reasoningFamilyData}
          keyExtractor={(row) => row.name}
          onView={(row) => handleViewReasoningFamily(row.name)}
          onEdit={(row) => handleEditReasoningFamily(row.name)}
          onDelete={(row) => handleDeleteReasoningFamily(row.name)}
          emptyMessage="No reasoning families configured"
          readonly={isReadonly}
        />

        {/* Models Table */}
        <div style={{ marginTop: '2rem' }}>
          <TableHeader
            title="Models"
            count={models.length}
            searchPlaceholder="Search models..."
            searchValue={modelsSearch}
            onSearchChange={setModelsSearch}
            onAdd={handleAddModel}
            addButtonText="Add Model"
            disabled={isReadonly}
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
            readonly={isReadonly}
          />
        </div>
      </div>
    )
  }

  // Router Configuration Section - System defaults from router-defaults.yaml
  const renderRouterConfigSection = () => (
    <div className={styles.sectionPanel}>
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
            <span className={styles.errorIcon}></span>
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
        onSave={editModalCallback || (async () => { })}
        title={editModalTitle}
        data={editModalData}
        fields={editModalFields}
        mode={editModalMode}
      />

      {/* View Modal */}
      <ViewModal
        isOpen={viewModalOpen}
        onClose={handleCloseViewModal}
        onEdit={isReadonly ? undefined : (viewModalEditCallback || undefined)}
        title={viewModalTitle}
        sections={viewModalSections}
      />
    </div>
  )
}

export default ConfigPage
