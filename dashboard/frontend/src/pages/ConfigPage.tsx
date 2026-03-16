import React, { useState, useEffect } from 'react'
import styles from './ConfigPage.module.css'
import { ConfigSection } from '../components/ConfigNav'
import EditModal, { FieldConfig } from '../components/EditModal'
import ViewModal, { ViewSection } from '../components/ViewModal'
import { DataTable, Column } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import EndpointsEditor, { Endpoint } from '../components/EndpointsEditor'
import RoutingPresetModal from '../components/RoutingPresetModal'
import { useReadonly } from '../contexts/ReadonlyContext'
import ConfigPageRouterConfigSection from './ConfigPageRouterConfigSection'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import {
  canonicalizeConfigForManagerSave,
  cloneConfigData,
  ensureProviderDefaultsConfig,
  ensureProvidersConfig,
  projectCanonicalConfigForManager,
  removeRoutingModelCard,
  upsertRoutingModelCard,
} from './configPageCanonicalization'
import {
  getRoutingPreset,
  listDecisionNames,
  listSignalNames,
  type RoutingPresetId,
} from '../presets/routingPresets'
import {
  ConfigFormat,
  detectConfigFormat,
  hasDecisions,
  hasFlatSignals,
  DecisionConditionType
} from '../types/config'
import { MCPConfigPanel } from '../components/MCPConfigPanel'
import {
  AddSignalFormState,
  BackendRefEntry,
  CanonicalGlobalConfig,
  clonePresetDecisions,
  clonePresetSignals,
  ComplexitySignal,
  collectConfiguredSignalNames,
  ConfigData,
  DecisionConfig,
  DecisionFormState,
  DomainSignal,
  EmbeddingSignal,
  FactCheckSignal,
  formatThreshold,
  JailbreakSignal,
  KeywordSignal,
  LanguageSignal,
  ModelConfigEntry, NormalizedModel, normalizeEndpoint, normalizeEndpoints,
  ModalitySignal,
  PIISignal,
  PreferenceSignal,
  ReasoningFamily,
  RoleBindingSignal,
  SignalType,
  Subject,
  TABLE_COLUMN_WIDTH,
  Tool,
  UserFeedbackSignal,
  ContextSignal,
  VLLMEndpoint,
} from './configPageSupport'

interface ConfigPageProps {
  activeSection?: ConfigSection
}

// Removed maskAddress - no longer needed after removing endpoint visibility toggle

const ConfigPage: React.FC<ConfigPageProps> = ({ activeSection = 'global-config' }) => {
  const { isReadonly } = useReadonly()
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [configFormat, setConfigFormat] = useState<ConfigFormat>('python-cli')

  // Effective global runtime config resolved from router defaults + config.yaml overrides
  const [routerDefaults, setRouterDefaults] = useState<CanonicalGlobalConfig | null>(null)

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
  const [viewModalEditCallback, setViewModalEditCallback] = useState<(() => void) | null>(null)

  // Search state
  const [decisionsSearch, setDecisionsSearch] = useState('')
  const [signalsSearch, setSignalsSearch] = useState('')
  const [modelsSearch, setModelsSearch] = useState('')
  const [presetModalOpen, setPresetModalOpen] = useState(false)
  const [selectedRoutingPresetId, setSelectedRoutingPresetId] = useState<RoutingPresetId | null>('starter-routing')
  const [presetApplyState, setPresetApplyState] = useState<'idle' | 'applying'>('idle')
  const [presetApplyError, setPresetApplyError] = useState<string | null>(null)

  // Expandable rows state for models
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set())

  useEffect(() => {
    fetchConfig()
    fetchRouterDefaults()
  }, [])

  const getRoutingModelCards = (cfg?: ConfigData) =>
    cfg?.routing?.modelCards || []

  const normalizeProviderModelEndpoints = (
    model: {
      endpoints?: Partial<Endpoint>[]
      backend_refs?: BackendRefEntry[]
    }
  ): Endpoint[] => {
    if (Array.isArray(model.backend_refs) && model.backend_refs.length > 0) {
      return model.backend_refs.map((backend, index) => {
        const baseURL = typeof backend.base_url === 'string' ? backend.base_url.trim() : ''
        const endpoint =
          typeof backend.endpoint === 'string' && backend.endpoint.trim()
            ? backend.endpoint.trim()
            : baseURL
        const protocol =
          backend.protocol ||
          (baseURL.startsWith('https://') ? 'https' : 'http')
        return normalizeEndpoint({
          name: backend.name,
          endpoint,
          protocol,
          weight: backend.weight,
        }, index)
      })
    }
    return normalizeEndpoints(model.endpoints)
  }

  const mergeProviderBackendRefs = (
    existingRefs: BackendRefEntry[] | undefined,
    endpoints: Endpoint[],
    accessKey?: string
  ): BackendRefEntry[] => {
    const existing = Array.isArray(existingRefs) ? existingRefs : []

    return endpoints.map((ep, index) => {
      const matched =
        existing.find((ref) => ref.name === ep.name) ||
        existing[index]

      const merged: BackendRefEntry = {
        ...(matched || {}),
        name: ep.name,
        protocol: ep.protocol,
        weight: ep.weight,
      }

      const matchedDisplayEndpoint =
        typeof matched?.endpoint === 'string' && matched.endpoint.trim()
          ? matched.endpoint.trim()
          : typeof matched?.base_url === 'string'
            ? matched.base_url.trim()
            : ''

      if (matched?.base_url && !matched?.endpoint) {
        if (ep.endpoint.trim() && ep.endpoint.trim() !== matchedDisplayEndpoint) {
          merged.base_url = ep.endpoint.trim()
        } else {
          merged.base_url = matched.base_url
        }
        delete merged.endpoint
      } else {
        merged.endpoint = ep.endpoint
      }

      if (accessKey?.trim()) {
        merged.api_key = accessKey.trim()
        delete merged.api_key_env
      } else if (!matched?.api_key && matched?.api_key_env) {
        delete merged.api_key
      }

      return merged
    })
  }

  // Fetch tools database when config is loaded
  useEffect(() => {
    const toolsDBPath =
      routerDefaults?.integrations?.tools?.tools_db_path ||
      config?.global?.integrations?.tools?.tools_db_path ||
      config?.tools?.tools_db_path

    if (toolsDBPath) {
      fetchToolsDB()
    }
  }, [
    config?.global?.integrations?.tools?.tools_db_path,
    config?.tools?.tools_db_path,
    routerDefaults?.integrations?.tools?.tools_db_path,
  ])

  const fetchConfig = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/router/config/all')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      const normalized = projectCanonicalConfigForManager(data)
      setConfig(normalized)
      // Detect config format
      const format = detectConfigFormat(normalized)
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
      const response = await fetch('/api/router/config/global')
      if (!response.ok) {
        console.warn('Global runtime config not available:', response.statusText)
        setRouterDefaults(null)
        return
      }
      const data = await response.json()
      setRouterDefaults(data)
    } catch (err) {
      console.warn('Failed to fetch global runtime config:', err)
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
      const canonicalConfig = canonicalizeConfigForManagerSave(updatedConfig as ConfigData)
      const response = await fetch('/api/router/config/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(canonicalConfig),
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
      case 'Context':
        cfg.signals.context = (cfg.signals.context || []).filter(s => s.name !== targetName)
        break
      case 'Complexity':
        cfg.signals.complexity = (cfg.signals.complexity || []).filter(s => s.name !== targetName)
        break
      case 'Modality':
        cfg.signals.modality = (cfg.signals.modality || []).filter(s => s.name !== targetName)
        break
      case 'Authz':
        cfg.signals.role_bindings = (cfg.signals.role_bindings || []).filter(s => s.name !== targetName)
        break
      case 'Jailbreak':
        cfg.signals.jailbreak = (cfg.signals.jailbreak || []).filter(s => s.name !== targetName)
        break
      case 'PII':
        cfg.signals.pii = (cfg.signals.pii || []).filter(s => s.name !== targetName)
        break
      default:
        break
    }
  }

  const removeDecisionByName = (cfg: ConfigData, targetName: string) => {
    cfg.decisions = (cfg.decisions || []).filter(d => d.name !== targetName)
  }

  const getSelectedPresetConflicts = () => {
    const defaultModel = getDefaultModel()
    if (!config || !defaultModel || !selectedRoutingPresetId) {
      return []
    }

    const preset = getRoutingPreset(selectedRoutingPresetId)
    if (!preset) {
      return []
    }

    const fragment = preset.build(defaultModel)
    const existingSignalNames = collectConfiguredSignalNames(config.signals)
    const existingDecisionNames = new Set((config.decisions || []).map((decision) => decision.name))
    const conflicts: string[] = []

    for (const signalName of listSignalNames(fragment.signals)) {
      if (existingSignalNames.has(signalName)) {
        conflicts.push(`Signal "${signalName}" already exists`)
      }
    }

    for (const decisionName of listDecisionNames(fragment.decisions)) {
      if (existingDecisionNames.has(decisionName)) {
        conflicts.push(`Decision "${decisionName}" already exists`)
      }
    }

    return conflicts
  }

  const handleApplyRoutingPreset = async () => {
    const defaultModel = getDefaultModel()
    if (!config || !isPythonCLI || !selectedRoutingPresetId || !defaultModel) {
      return
    }

    const conflicts = getSelectedPresetConflicts()
    if (conflicts.length > 0) {
      return
    }

    const preset = getRoutingPreset(selectedRoutingPresetId)
    if (!preset) {
      return
    }

    const fragment = preset.build(defaultModel)
    const mergedSignals = clonePresetSignals(fragment.signals as Record<string, unknown> | undefined)
    const mergedDecisions = clonePresetDecisions(fragment.decisions)
    const nextSignals = { ...(config.signals || {}) } as Record<string, Array<Record<string, unknown>>>

    const nextConfig: ConfigData = {
      ...config,
      signals: nextSignals as ConfigData['signals'],
      decisions: [...(config.decisions || [])],
    }

    if (mergedSignals) {
      for (const [key, value] of Object.entries(mergedSignals)) {
        if (!Array.isArray(value) || value.length === 0) {
          continue
        }

        const existingValues = nextSignals[key] || []
        nextSignals[key] = [
          ...existingValues,
          ...(value as Array<Record<string, unknown>>),
        ]
      }
    }

    nextConfig.decisions = [...(nextConfig.decisions || []), ...mergedDecisions]

    setPresetApplyState('applying')
    setPresetApplyError(null)

    try {
      await saveConfig(nextConfig)
      setPresetModalOpen(false)
    } catch (err) {
      setPresetApplyError(err instanceof Error ? err.message : 'Failed to apply preset')
    } finally {
      setPresetApplyState('idle')
    }
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

  // ============================================================================
  // HELPER FUNCTIONS - Normalize data access across config formats
  // ============================================================================

  // Helper: Check if using Python CLI format
  const isPythonCLI = configFormat === 'python-cli'
  const selectedPresetConflicts = getSelectedPresetConflicts()

  // Get models - from providers.models (Python CLI) or model_config (legacy)
  function getModels(): NormalizedModel[] {
    if (isPythonCLI && config?.providers?.models) {
      const cards = getRoutingModelCards(config)
      const cardByName = new Map(cards.map((card) => [card.name, card]))
      const models = config.providers.models.map((m): NormalizedModel => ({
        name: m.name,
        reasoning_family: m.reasoning_family,
        endpoints: normalizeProviderModelEndpoints(m),
        access_key: m.backend_refs?.find((ref) => ref.api_key || ref.api_key_env)?.api_key,
        param_size: cardByName.get(m.name)?.param_size,
        context_window_size: cardByName.get(m.name)?.context_window_size,
        description: cardByName.get(m.name)?.description,
        capabilities: cardByName.get(m.name)?.capabilities,
        loras: cardByName.get(m.name)?.loras,
        tags: cardByName.get(m.name)?.tags,
        quality_score: cardByName.get(m.name)?.quality_score,
        modality: cardByName.get(m.name)?.modality,
        pricing: m.pricing,
      }))

      for (const card of cards) {
        if (models.some((model) => model.name === card.name)) {
          continue
        }
        models.push({
          name: card.name,
          reasoning_family: undefined,
          endpoints: [],
          param_size: card.param_size,
          context_window_size: card.context_window_size,
          description: card.description,
          capabilities: card.capabilities,
          loras: card.loras,
          tags: card.tags,
          quality_score: card.quality_score,
          modality: card.modality,
          pricing: undefined,
        })
      }

      return models
    }
    // Legacy format - convert model_config to array
    if (config?.model_config) {
      return (Object.entries(config.model_config) as [string, ModelConfigEntry][]).map(([name, cfg]) => ({
        name,
        reasoning_family: cfg.reasoning_family,
        endpoints: cfg.preferred_endpoints?.map((ep: string) => {
          const endpoint = config.vllm_endpoints?.find((e: VLLMEndpoint) => e.name === ep)
          return endpoint ? normalizeEndpoint({
            name: ep,
            weight: endpoint.weight || 1,
            endpoint: `${endpoint.address}:${endpoint.port}`,
            protocol: 'http',
          }, 0) : null
        }).filter((e): e is NonNullable<typeof e> => e !== null) || [],
        access_key: undefined,
        pricing: cfg.pricing
      }))
    }
    return []
  }

  // Get default model
  function getDefaultModel(): string {
    if (isPythonCLI) {
      return config?.providers?.defaults?.default_model || ''
    }
    return config?.default_model || ''
  }

  // Get reasoning families
  function getReasoningFamilies(): Record<string, ReasoningFamily> {
    if (isPythonCLI) {
      return config?.providers?.defaults?.reasoning_families || {}
    }
    return config?.reasoning_families || {}
  }


  // ============================================================================
  // SECTION PANEL RENDERS - Aligned with Python CLI config structure
  // ============================================================================

  // Signals Section - Keywords, Embeddings, Domains, Preferences (config.yaml)
  const renderSignalsSection = () => {
    // Support both nested (signals.*) and flat (keyword_rules, etc.) formats.
    // After deploy, Router flattens signals.keywords → keyword_rules, etc.
    const signals = config?.signals
    const flatSignals: ConfigData['signals'] | null = !signals && hasFlatSignals(config) ? {
      keywords: config?.keyword_rules,
      embeddings: config?.embedding_rules,
      domains: (config?.categories || []).map((category) => ({
        name: category.name,
        description: category.description || '',
        mmlu_categories: category.mmlu_categories,
      })),
      fact_check: config?.fact_check_rules,
      user_feedbacks: config?.user_feedback_rules,
      preferences: config?.preference_rules,
      language: config?.language_rules,
      context: config?.context_rules,
      complexity: config?.complexity_rules,
      modality: undefined,
      role_bindings: undefined,
      jailbreak: config?.jailbreak,
      pii: config?.pii,
    } : null
    const effectiveSignals = signals || flatSignals

    type UnifiedSignalData = Partial<
      KeywordSignal &
      EmbeddingSignal &
      DomainSignal &
      PreferenceSignal &
      FactCheckSignal &
      UserFeedbackSignal &
      LanguageSignal &
      ContextSignal &
      ComplexitySignal &
      ModalitySignal &
      RoleBindingSignal &
      JailbreakSignal &
      PIISignal
    >

    interface UnifiedSignal {
      name: string
      type: SignalType
      summary: string
      rawData: UnifiedSignalData
    }

    // Flatten all signals into a unified array
    const allSignals: UnifiedSignal[] = []

    // Keywords
    effectiveSignals?.keywords?.forEach(kw => {
      allSignals.push({
        name: kw.name,
        type: 'Keywords',
        summary: `${kw.operator}, ${kw.keywords.length} keywords${kw.case_sensitive ? ', case-sensitive' : ''}`,
        rawData: kw
      })
    })

    // Embeddings
    effectiveSignals?.embeddings?.forEach(emb => {
      allSignals.push({
        name: emb.name,
        type: 'Embeddings',
        summary: `Threshold: ${Math.round(emb.threshold * 100)}%, ${emb.candidates.length} items, ${emb.aggregation_method}`,
        rawData: emb
      })
    })

    // Domains
    effectiveSignals?.domains?.forEach(domain => {
      const categoryCount = domain.mmlu_categories?.length || 0
      allSignals.push({
        name: domain.name,
        type: 'Domain',
        summary: categoryCount > 0 ? `${categoryCount} MMLU categories` : (domain.description || 'No description'),
        rawData: domain
      })
    })

    // Preferences
    effectiveSignals?.preferences?.forEach(pref => {
      const examplesCount = pref.examples?.length || 0
      const thresholdText = pref.threshold !== undefined ? ` • threshold ${formatThreshold(pref.threshold)}` : ''
      const examplesText = examplesCount > 0 ? ` • ${examplesCount} ${examplesCount === 1 ? 'example' : 'examples'}` : ''
      allSignals.push({
        name: pref.name,
        type: 'Preference',
        summary: `${pref.description || 'No description'}${examplesText}${thresholdText}`,
        rawData: pref
      })
    })

    // Fact Check
    effectiveSignals?.fact_check?.forEach(fc => {
      allSignals.push({
        name: fc.name,
        type: 'Fact Check',
        summary: fc.description || 'No description',
        rawData: fc
      })
    })

    // User Feedbacks
    effectiveSignals?.user_feedbacks?.forEach(uf => {
      allSignals.push({
        name: uf.name,
        type: 'User Feedback',
        summary: uf.description || 'No description',
        rawData: uf
      })
    })

    // Language
    effectiveSignals?.language?.forEach(lang => {
      allSignals.push({
        name: lang.name,
        type: 'Language',
        summary: 'Language detection rule',
        rawData: lang
      })
    })

    // Context
    effectiveSignals?.context?.forEach(ctx => {
      allSignals.push({
        name: ctx.name,
        type: 'Context',
        summary: `${ctx.min_tokens} to ${ctx.max_tokens} tokens`,
        rawData: ctx
      })
    })

    // Complexity
    effectiveSignals?.complexity?.forEach(comp => {
      const hardCount = comp.hard?.candidates?.length || 0
      const easyCount = comp.easy?.candidates?.length || 0
      allSignals.push({
        name: comp.name,
        type: 'Complexity',
        summary: `Threshold: ${comp.threshold}, ${hardCount} hard / ${easyCount} easy candidates`,
        rawData: comp
      })
    })

    // Modality
    effectiveSignals?.modality?.forEach(modality => {
      allSignals.push({
        name: modality.name,
        type: 'Modality',
        summary: modality.description || 'Modality signal',
        rawData: modality,
      })
    })

    // Authz role bindings
    effectiveSignals?.role_bindings?.forEach(binding => {
      const subjectCount = binding.subjects?.length || 0
      allSignals.push({
        name: binding.name,
        type: 'Authz',
        summary: `${binding.role} • ${subjectCount} ${subjectCount === 1 ? 'subject' : 'subjects'}`,
        rawData: binding,
      })
    })

    // Jailbreak
    effectiveSignals?.jailbreak?.forEach(jb => {
      const method = jb.method || 'classifier'
      allSignals.push({
        name: jb.name,
        type: 'Jailbreak',
        summary: `Method: ${method}, Threshold: ${jb.threshold}${jb.include_history ? ', includes history' : ''}`,
        rawData: jb
      })
    })

    // PII
    effectiveSignals?.pii?.forEach(p => {
      const allowed = p.pii_types_allowed?.length || 0
      allSignals.push({
        name: p.name,
        type: 'PII',
        summary: `Threshold: ${p.threshold}${allowed > 0 ? `, ${allowed} types allowed` : ', deny all'}`,
        rawData: p
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
        width: TABLE_COLUMN_WIDTH.medium,
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
            'Context': 'rgba(251, 146, 60, 0.15)',
            'Complexity': 'rgba(66, 153, 225, 0.15)',
            'Modality': 'rgba(20, 184, 166, 0.15)',
            'Authz': 'rgba(168, 85, 247, 0.15)',
            'Jailbreak': 'rgba(239, 68, 68, 0.15)',
            'PII': 'rgba(245, 158, 11, 0.15)'
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
                  {(signal.rawData.keywords || []).map((kw: string, i: number) => (
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
            { label: 'Threshold', value: `${Math.round((signal.rawData.threshold || 0) * 100)}%` },
            { label: 'Aggregation Method', value: signal.rawData.aggregation_method },
            {
              label: 'Candidates',
              value: (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {(signal.rawData.candidates || []).map((c: string, i: number) => (
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
      } else if (signal.type === 'Preference') {
        sections.push({
          title: 'Preference Configuration',
          fields: [
            { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
            { label: 'Threshold', value: signal.rawData.threshold !== undefined ? formatThreshold(signal.rawData.threshold) : 'Not set' },
            {
              label: 'Examples',
              value: signal.rawData.examples?.length ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontFamily: 'var(--font-mono)', fontSize: '0.9rem' }}>
                  {signal.rawData.examples.map((ex: string, i: number) => (
                    <div key={i} style={{ padding: '0.35rem 0.5rem', background: 'rgba(234, 179, 8, 0.1)', borderRadius: 6 }}>
                      {ex}
                    </div>
                  ))}
                </div>
              ) : 'No examples provided',
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
      } else if (signal.type === 'Modality') {
        sections.push({
          title: 'Modality Signal',
          fields: [
            { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
          ]
        })
      } else if (signal.type === 'Authz') {
        sections.push({
          title: 'Role Binding',
          fields: [
            { label: 'Role', value: signal.rawData.role || 'N/A' },
            {
              label: 'Subjects',
              value: signal.rawData.subjects?.length ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                  {signal.rawData.subjects.map((subject: Subject, i: number) => (
                    <div key={i}>{subject.kind}:{subject.name}</div>
                  ))}
                </div>
              ) : 'No subjects',
              fullWidth: true,
            },
            { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true },
          ]
        })
      } else if (signal.type === 'Jailbreak') {
        const fields = [
          { label: 'Method', value: signal.rawData.method || 'classifier', fullWidth: true },
          { label: 'Threshold', value: signal.rawData.threshold?.toString() || 'N/A', fullWidth: true },
          { label: 'Include History', value: signal.rawData.include_history ? 'Yes' : 'No', fullWidth: true },
        ]
        if (signal.rawData.method === 'contrastive') {
          fields.push(
            { label: 'Jailbreak Patterns', value: (signal.rawData.jailbreak_patterns || []).length + ' patterns', fullWidth: true },
            { label: 'Benign Patterns', value: (signal.rawData.benign_patterns || []).length + ' patterns', fullWidth: true },
          )
        }
        fields.push({ label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true })
        sections.push({ title: 'Jailbreak Signal', fields })
      } else if (signal.type === 'PII') {
        sections.push({
          title: 'PII Signal',
          fields: [
            { label: 'Threshold', value: signal.rawData.threshold?.toString() || 'N/A', fullWidth: true },
            { label: 'Allowed PII Types', value: signal.rawData.pii_types_allowed?.join(', ') || 'None (deny all)', fullWidth: true },
            { label: 'Include History', value: signal.rawData.include_history ? 'Yes' : 'No', fullWidth: true },
            { label: 'Description', value: signal.rawData.description || 'N/A', fullWidth: true }
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
        preference_examples: '',
        preference_threshold: undefined,
        min_tokens: '0',
        max_tokens: '8K',
        complexity_threshold: 0.1,
        role: '',
        subjects: '',
        hard_candidates: '',
        easy_candidates: '',
        composer_operator: 'AND',
        composer_conditions: '',
        jailbreak_threshold: 0.65,
        jailbreak_method: 'classifier',
        include_history: false,
        jailbreak_patterns: '',
        benign_patterns: '',
        pii_threshold: 0.5,
        pii_types_allowed: '',
        pii_include_history: false
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
        preference_examples: (signal.rawData.examples || []).join('\n'),
        preference_threshold: signal.rawData.threshold,
        min_tokens: signal.rawData.min_tokens || '0',
        max_tokens: signal.rawData.max_tokens || '8K',
        complexity_threshold: signal.rawData.threshold ?? 0.1,
        role: signal.type === 'Authz' ? signal.rawData.role || '' : '',
        subjects: signal.type === 'Authz'
          ? (signal.rawData.subjects || []).map((subject: Subject) => `${subject.kind}:${subject.name}`).join('\n')
          : '',
        hard_candidates: (signal.rawData.hard?.candidates || []).join('\n'),
        easy_candidates: (signal.rawData.easy?.candidates || []).join('\n'),
        composer_operator: signal.rawData.composer?.operator || 'AND',
        composer_conditions: signal.rawData.composer?.conditions?.map((c: { type: string; name: string }) => `${c.type}:${c.name}`).join('\n') || '',
        jailbreak_threshold: signal.rawData.threshold ?? 0.65,
        jailbreak_method: signal.rawData.method || 'classifier',
        include_history: !!signal.rawData.include_history,
        jailbreak_patterns: (signal.rawData.jailbreak_patterns || []).join('\n'),
        benign_patterns: (signal.rawData.benign_patterns || []).join('\n'),
        pii_threshold: signal.rawData.threshold ?? 0.5,
        pii_types_allowed: (signal.rawData.pii_types_allowed || []).join('\n'),
        pii_include_history: !!signal.rawData.include_history
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

      const preferenceFields: FieldConfig[] = [
        {
          name: 'preference_examples',
          label: 'Examples (preference only)',
          type: 'textarea',
          placeholder: 'One example per line to represent this preference',
          description: 'Few-shot hints sent to the contrastive preference classifier.',
          shouldHide: conditionallyHideFieldExceptType('Preference')
        },
        {
          name: 'preference_threshold',
          label: 'Threshold (preference only)',
          type: 'number',
          min: 0,
          max: 1,
          step: 0.01,
          placeholder: 'e.g., 0.35',
          description: 'Override the global preference threshold for this specific rule.',
          shouldHide: conditionallyHideFieldExceptType('Preference')
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

      const modalityFields: FieldConfig[] = []

      const authzFields: FieldConfig[] = [
        {
          name: 'role',
          label: 'Role (authz only)',
          type: 'text',
          placeholder: 'admin',
          shouldHide: conditionallyHideFieldExceptType('Authz')
        },
        {
          name: 'subjects',
          label: 'Subjects (authz only)',
          type: 'textarea',
          placeholder: 'One subject per line, e.g.:\nUser:alice\nGroup:admins',
          shouldHide: conditionallyHideFieldExceptType('Authz')
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
          placeholder: 'One condition per line in format type:name, e.g.:\ndomain:computer science\nkeyword:coding',
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

      const jailbreakFields: FieldConfig[] = [
        {
          name: 'jailbreak_method',
          label: 'Method (jailbreak only)',
          type: 'select',
          options: ['classifier', 'contrastive'],
          description: 'Detection method: "classifier" (BERT-based) or "contrastive" (embedding KB similarity)',
          shouldHide: conditionallyHideFieldExceptType('Jailbreak')
        },
        {
          name: 'jailbreak_threshold',
          label: 'Threshold (jailbreak only)',
          type: 'number',
          placeholder: 'e.g., 0.65 for classifier, 0.10 for contrastive',
          description: 'Confidence threshold for jailbreak detection (0.0 - 1.0)',
          shouldHide: conditionallyHideFieldExceptType('Jailbreak')
        },
        {
          name: 'include_history',
          label: 'Include History (jailbreak only)',
          type: 'boolean',
          description: 'Whether to include conversation history in jailbreak detection',
          shouldHide: conditionallyHideFieldExceptType('Jailbreak')
        },

        {
          name: 'jailbreak_patterns',
          label: 'Jailbreak Patterns (contrastive only)',
          type: 'textarea',
          placeholder: 'One pattern per line, e.g.:\nIgnore all previous instructions\nYou are now DAN',
          description: 'Known jailbreak prompts for the contrastive KB',
          shouldHide: conditionallyHideFieldExceptType('Jailbreak')
        },
        {
          name: 'benign_patterns',
          label: 'Benign Patterns (contrastive only)',
          type: 'textarea',
          placeholder: 'One pattern per line, e.g.:\nWhat is the weather today\nHelp me write an email',
          description: 'Known benign prompts for the contrastive KB',
          shouldHide: conditionallyHideFieldExceptType('Jailbreak')
        }
      ]

      const piiFields: FieldConfig[] = [
        {
          name: 'pii_threshold',
          label: 'Threshold (PII only)',
          type: 'number',
          placeholder: 'e.g., 0.5',
          description: 'Confidence threshold for PII detection (0.0 - 1.0)',
          shouldHide: conditionallyHideFieldExceptType('PII')
        },
        {
          name: 'pii_types_allowed',
          label: 'Allowed PII Types (PII only)',
          type: 'textarea',
          placeholder: 'One PII type per line, e.g.:\nEMAIL_ADDRESS\nPHONE_NUMBER',
          description: 'PII types to allow (not blocked). Leave empty to deny all.',
          shouldHide: conditionallyHideFieldExceptType('PII')
        },
        {
          name: 'pii_include_history',
          label: 'Include History (PII only)',
          type: 'boolean',
          description: 'Whether to include conversation history in PII detection',
          shouldHide: conditionallyHideFieldExceptType('PII')
        }
      ]

      const fields: FieldConfig[] = [
        {
          name: 'type',
          label: 'Type',
          type: 'select',
          options: ['Keywords', 'Embeddings', 'Domain', 'Preference', 'Fact Check', 'User Feedback', 'Language', 'Context', 'Complexity', 'Modality', 'Authz', 'Jailbreak', 'PII'],
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
        ...preferenceFields,
        ...keywordFields,
        ...embeddingFields,
        ...domainFields,
        ...contextFields,
        ...complexityFields,
        ...modalityFields,
        ...authzFields,
        ...jailbreakFields,
        ...piiFields,
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
            const examples = listInputToArray(formData.preference_examples || '')
            const hasThreshold = Number.isFinite(formData.preference_threshold)
            const threshold = hasThreshold ? Math.max(0, Math.min(1, Number(formData.preference_threshold))) : undefined

            const preferenceRule: { name: string; description: string; examples?: string[]; threshold?: number } = {
              name,
              description: formData.description || ''
            }

            if (examples.length > 0) {
              preferenceRule.examples = examples
            }

            if (threshold !== undefined && threshold > 0) {
              preferenceRule.threshold = threshold
            }

            newConfig.signals.preferences = [
              ...(newConfig.signals.preferences || []),
              preferenceRule
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
          case 'Modality': {
            newConfig.signals.modality = [
              ...(newConfig.signals.modality || []),
              {
                name,
                description: formData.description || undefined,
              }
            ]
            break
          }
          case 'Authz': {
            const role = (formData.role || '').trim()
            if (!role) {
              throw new Error('Role is required for authz signals.')
            }
            const subjects = listInputToArray(formData.subjects || '').map((entry) => {
              const [kindRaw, ...nameParts] = entry.split(':')
              const kind = (kindRaw || '').trim()
              const subjectName = nameParts.join(':').trim()
              if ((kind !== 'User' && kind !== 'Group') || !subjectName) {
                throw new Error(`Invalid authz subject "${entry}". Expected User:name or Group:name.`)
              }
              return { kind, name: subjectName } as Subject
            })
            if (subjects.length === 0) {
              throw new Error('At least one subject is required for authz signals.')
            }
            newConfig.signals.role_bindings = [
              ...(newConfig.signals.role_bindings || []),
              {
                name,
                role,
                subjects,
                description: formData.description || undefined,
              }
            ]
            break
          }
          case 'Jailbreak': {
            const jailbreak_threshold = formData.jailbreak_threshold ?? 0.65
            if (jailbreak_threshold < 0 || jailbreak_threshold > 1) {
              throw new Error('Jailbreak threshold must be between 0.0 and 1.0.')
            }
            const method = formData.jailbreak_method || 'classifier'
            const jailbreakEntry: JailbreakSignal = {
              name,
              threshold: jailbreak_threshold,
              include_history: formData.include_history || false,
              description: formData.description || undefined
            }
            if (method !== 'classifier') {
              jailbreakEntry.method = method
            }
            if (method === 'contrastive') {
              const jailbreakPatterns = (formData.jailbreak_patterns || '').trim()
              const benignPatternsStr = (formData.benign_patterns || '').trim()
              if (jailbreakPatterns) {
                jailbreakEntry.jailbreak_patterns = jailbreakPatterns.split('\n').map((p: string) => p.trim()).filter((p: string) => p.length > 0)
              }
              if (benignPatternsStr) {
                jailbreakEntry.benign_patterns = benignPatternsStr.split('\n').map((p: string) => p.trim()).filter((p: string) => p.length > 0)
              }
            }
            newConfig.signals.jailbreak = [
              ...(newConfig.signals.jailbreak || []),
              jailbreakEntry
            ]
            break
          }
          case 'PII': {
            const pii_threshold = formData.pii_threshold ?? 0.5
            if (pii_threshold < 0 || pii_threshold > 1) {
              throw new Error('PII threshold must be between 0.0 and 1.0.')
            }
            const pii_types_allowed = (formData.pii_types_allowed || '').trim()
            const allowedList = pii_types_allowed
              ? pii_types_allowed.split('\n').map(t => t.trim()).filter(t => t.length > 0)
              : undefined
            newConfig.signals.pii = [
              ...(newConfig.signals.pii || []),
              {
                name,
                threshold: pii_threshold,
                pii_types_allowed: allowedList,
                include_history: formData.pii_include_history || false,
                description: formData.description || undefined
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
      <ConfigPageManagerLayout title="Signals" description="Review the signal catalog that drives semantic routing, guardrails, and context-aware behavior.">
        <div className={styles.sectionPanel}>
          <div className={styles.sectionTableBlock}>
            <TableHeader title="Signals" count={allSignals.length} searchPlaceholder="Search signals..." searchValue={signalsSearch} onSearchChange={setSignalsSearch} onAdd={() => openSignalEditor('add')} addButtonText="Add Signal" disabled={isReadonly} variant="embedded" />
            {(isPythonCLI || hasFlatSignals(config)) ? (
              <DataTable
                columns={signalsColumns}
                data={filteredSignals}
                keyExtractor={(row) => `${row.type}-${row.name}`}
                onView={handleViewSignal}
                onEdit={handleEditSignal}
                onDelete={handleDeleteSignal}
                emptyMessage={signalsSearch ? 'No signals match your search' : 'No signals configured'}
                className={styles.managerTable}
                readonly={isReadonly}
              />
            ) : (
              <div className={styles.emptyState}>
                Signals are only available in Python CLI config format.
                Current config uses legacy format - use "Intelligent Routing" features instead.
              </div>
            )}
          </div>
        </div>
      </ConfigPageManagerLayout>
    )
  }

  // Decisions Section - Routing rules with priorities (config.yaml)
  const renderDecisionsSection = () => {
    const decisions = config?.decisions || []

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
        width: TABLE_COLUMN_WIDTH.compact,
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
        width: TABLE_COLUMN_WIDTH.medium,
        render: (row) => {
          const count = row.rules?.conditions?.length || 0
          return <span>{count} {count === 1 ? 'condition' : 'conditions'}</span>
        }
      },
      {
        key: 'models',
        header: 'Models',
        width: TABLE_COLUMN_WIDTH.medium,
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
      const conditionTypeOptions = ['keyword', 'domain', 'preference', 'user_feedback', 'embedding', 'fact_check', 'language', 'context', 'complexity', 'modality', 'authz', 'jailbreak', 'pii'] as const

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
          case 'fact_check':
            return config?.signals?.fact_check?.map((f) => f.name) || []
          default:
            break
          case 'language':
            return config?.signals?.language?.map((l) => l.name) || []
          case 'context':
            return config?.signals?.context?.map((c) => c.name) || []
          case 'complexity':
            return (config?.signals?.complexity || []).flatMap((signal) => [
              `${signal.name}:easy`,
              `${signal.name}:medium`,
              `${signal.name}:hard`,
            ])
          case 'modality':
            return config?.signals?.modality?.map((m) => m.name) || []
          case 'authz':
            return config?.signals?.role_bindings?.map((binding) => binding.name) || []
          case 'jailbreak':
            return config?.signals?.jailbreak?.map((rule) => rule.name) || []
          case 'pii':
            return config?.signals?.pii?.map((rule) => rule.name) || []
        }
        return []
      }

      const defaultForm: DecisionFormState = {
        name: '',
        description: '',
        priority: 1,
        operator: 'AND',
        conditions: [{ type: 'keyword', name: '' }],
        modelRefs: [
          {
            model: '',
            use_reasoning: false,
            reasoning_description: '',
            reasoning_effort: '',
            lora_name: '',
          },
        ],
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
          use_reasoning: !!ref.use_reasoning,
          reasoning_description: ref.reasoning_description || '',
          reasoning_effort: ref.reasoning_effort || '',
          lora_name: ref.lora_name || '',
          weight: typeof ref.weight === 'number' ? ref.weight : undefined,
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
        const modelOptions = getModels().map((model) => model.name)
        const rows = (Array.isArray(value) ? value : []).length
          ? value
          : [{ model: '', use_reasoning: false, reasoning_description: '', reasoning_effort: '', lora_name: '' }]

        const updateItem = (
          index: number,
          key: 'model' | 'use_reasoning' | 'reasoning_description' | 'reasoning_effort' | 'lora_name' | 'weight',
          val: string | boolean | number | undefined
        ) => {
          const next = rows.map((item, idx) => idx === index ? { ...item, [key]: val } : item)
          onChange(next)
        }

        const removeItem = (index: number) => {
          const next = rows.filter((_, idx) => idx !== index)
          onChange(next.length ? next : [{ model: '', use_reasoning: false, reasoning_description: '', reasoning_effort: '', lora_name: '' }])
        }

        const addItem = () =>
          onChange([
            ...rows,
            { model: '', use_reasoning: false, reasoning_description: '', reasoning_effort: '', lora_name: '' },
          ])

        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {rows.map((ref, idx) => (
              <div
                key={idx}
                style={{
                  display: 'grid',
                  gap: '0.75rem',
                  padding: '0.75rem',
                  borderRadius: 8,
                  border: '1px solid var(--color-border)'
                }}
              >
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'minmax(220px, 1fr) minmax(180px, 220px) auto auto',
                    gap: '0.5rem',
                    alignItems: 'center'
                  }}
                >
                  <select
                    value={ref?.model || ''}
                    onChange={(e) => updateItem(idx, 'model', e.target.value)}
                    style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                  >
                    <option value="">Select model</option>
                    {ref?.model && !modelOptions.includes(ref.model) ? (
                      <option value={ref.model}>{ref.model}</option>
                    ) : null}
                    {modelOptions.map((option) => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                  <select
                    value={ref?.reasoning_effort || ''}
                    onChange={(e) => updateItem(idx, 'reasoning_effort', e.target.value)}
                    style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                  >
                    <option value="">Default effort</option>
                    <option value="low">low</option>
                    <option value="medium">medium</option>
                    <option value="high">high</option>
                  </select>
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
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'minmax(220px, 1fr) minmax(140px, 180px)',
                    gap: '0.5rem',
                    alignItems: 'center'
                  }}
                >
                  <input
                    type="text"
                    value={ref?.lora_name || ''}
                    onChange={(e) => updateItem(idx, 'lora_name', e.target.value)}
                    placeholder="LoRA adapter name (optional)"
                    style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                  />
                  <input
                    type="number"
                    value={typeof ref?.weight === 'number' ? ref.weight : ''}
                    onChange={(e) => updateItem(idx, 'weight', e.target.value === '' ? undefined : Number(e.target.value))}
                    placeholder="Weight"
                    step="0.1"
                    min="0"
                    style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
                  />
                </div>
                <input
                  type="text"
                  value={ref?.reasoning_description || ''}
                  onChange={(e) => updateItem(idx, 'reasoning_description', e.target.value)}
                  placeholder="Reasoning description (optional)"
                  style={{ padding: '0.55rem 0.75rem', borderRadius: 6, border: '1px solid var(--color-border)' }}
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
          options: ['AND', 'OR', 'NOT'],
          description: 'AND: all conditions must match. OR: any condition matches. NOT: none of the conditions must match (exclusion routing).',
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
          const modelRef: DecisionConfig['modelRefs'][number] = {
            model,
            use_reasoning: !!m?.use_reasoning,
          }
          const reasoningDescription = (m?.reasoning_description || '').trim()
          if (reasoningDescription) {
            modelRef.reasoning_description = reasoningDescription
          }
          const reasoningEffort = (m?.reasoning_effort || '').trim()
          if (reasoningEffort) {
            modelRef.reasoning_effort = reasoningEffort
          }
          const loraName = (m?.lora_name || '').trim()
          if (loraName) {
            modelRef.lora_name = loraName
          }
          if (typeof m?.weight === 'number' && Number.isFinite(m.weight)) {
            modelRef.weight = m.weight
          }
          return modelRef
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
      <ConfigPageManagerLayout title="Decisions" description="Shape routing outcomes with ordered rules, plugins, and presets that map signals to concrete model behavior.">
        <div className={styles.sectionPanel}>
          <div className={styles.sectionTableBlock}>
            <TableHeader
              title="Routing Decisions"
              count={decisions.length}
              searchPlaceholder="Search decisions..."
              searchValue={decisionsSearch}
              onSearchChange={setDecisionsSearch}
              onSecondaryAction={
                isPythonCLI && getDefaultModel()
                  ? () => {
                      setPresetApplyError(null)
                      setPresetModalOpen(true)
                    }
                  : undefined
              }
              secondaryActionText="Apply preset"
              onAdd={() => openDecisionEditor('add')}
              addButtonText="Add Decision" disabled={isReadonly} variant="embedded"
            />
            {(isPythonCLI || hasDecisions(config)) ? (
              <DataTable
                columns={decisionsColumns}
                data={filteredDecisions}
                keyExtractor={(row) => row.name}
                onView={handleViewDecision}
                onEdit={handleEditDecision}
                onDelete={handleDeleteDecision}
                emptyMessage={decisionsSearch ? 'No decisions match your search' : 'No routing decisions configured'}
                className={styles.managerTable}
                readonly={isReadonly}
              />
            ) : (
              <div className={styles.emptyState}>
                Decisions are only available in Python CLI config format.
                Current config uses legacy format - see "Categories" in legacy mode.
              </div>
            )}
          </div>
        </div>
      </ConfigPageManagerLayout>
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

    // Handle view model
    const handleViewModel = (model: ModelRow) => {
      const sections: ViewSection[] = [
        {
          title: 'Basic Information',
          fields: [
            { label: 'Model Name', value: model.name },
            { label: 'Reasoning Family', value: model.reasoning_family || 'N/A' },
            { label: 'Is Default', value: model.name === getDefaultModel() ? 'Yes' : 'No' },
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

      setViewModalTitle(`Model: ${model.name}`)
      setViewModalSections(sections)
      setViewModalEditCallback(() => () => handleEditModel(model))
      setViewModalOpen(true)
    }

    // Handle add model
    const handleAddModel = () => {
      const reasoningFamiliesObj = getReasoningFamilies()
      const reasoningFamilyNames = Object.keys(reasoningFamiliesObj)

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
          // Endpoints are already validated by EndpointsEditor
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
            // Legacy format
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
          param_size: model.param_size || '',
          context_window_size: model.context_window_size || '',
          description: model.description || '',
          capabilities: (model.capabilities || []).join('\n'),
          loras: model.loras || [],
          tags: (model.tags || []).join('\n'),
          quality_score: model.quality_score ?? '',
          modality: model.modality || '',
          // Endpoints
          endpoints: normalizeEndpoints(model.endpoints),
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

          // Endpoints are already validated by EndpointsEditor
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
            type ModelType = NonNullable<ConfigData['providers']>['models'][number]
            providers.models = providers.models.map((m: ModelType) =>
              m.name === model.name ? {
                ...m,
                reasoning_family: data.reasoning_family || undefined,
                provider_model_id: m.provider_model_id || model.name,
                backend_refs: mergeProviderBackendRefs(m.backend_refs, endpoints, data.access_key),
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

    // Handle delete model
    const handleDeleteModel = (model: ModelRow) => {
      if (confirm(`Are you sure you want to delete model "${model.name}"?`)) {
        handleDeleteModelAction(model.name)
      }
    }

    const handleDeleteModelAction = async (modelName: string) => {
      if (!config) {
        return
      }
      const newConfig = cloneConfigData(config)
      if (isPythonCLI && newConfig.providers?.models) {
        const providers = ensureProvidersConfig(newConfig)
        type ModelType = NonNullable<ConfigData['providers']>['models'][number]
        providers.models = providers.models.filter((m: ModelType) => m.name !== modelName)
        removeRoutingModelCard(newConfig, modelName)
        // Update default model if deleted
        const defaults = ensureProviderDefaultsConfig(newConfig)
        if (defaults.default_model === modelName) {
          defaults.default_model = providers.models[0]?.name || ''
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
      if (!confirm(`Are you sure you want to delete reasoning family "${familyName}"?`)) {
        return
      }

      const newConfig = { ...config }
      if (isPythonCLI && newConfig.providers?.defaults?.reasoning_families) {
        const defaults = ensureProviderDefaultsConfig(newConfig)
        defaults.reasoning_families = { ...defaults.reasoning_families }
        delete defaults.reasoning_families[familyName]
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
            <TableHeader title="Models" count={models.length} searchPlaceholder="Search models..." searchValue={modelsSearch} onSearchChange={setModelsSearch} onAdd={handleAddModel} addButtonText="Add Model" disabled={isReadonly} variant="embedded" />
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

  // Global Config section - canonical global override editor backed by effective router defaults
  const renderGlobalConfigSection = () => (
          <ConfigPageRouterConfigSection
            config={config}
            toolsData={toolsData}
            toolsLoading={toolsLoading}
            toolsError={toolsError}
            isReadonly={isReadonly}
            openEditModal={openEditModal}
            saveConfig={saveConfig}
            refreshConfig={fetchConfig}
            showLegacyCategories={!isPythonCLI}
          />
  )

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'signals':
        return renderSignalsSection()
      case 'decisions':
        return renderDecisionsSection()
      case 'models':
        return renderModelsSection()
      case 'global-config':
        return renderGlobalConfigSection()
      case 'mcp':
        return <MCPConfigPanel />
      default:
        return renderGlobalConfigSection()
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

      <RoutingPresetModal
        isOpen={presetModalOpen}
        defaultModel={getDefaultModel()}
        selectedPresetId={selectedRoutingPresetId}
        conflicts={selectedPresetConflicts}
        error={presetApplyError}
        isApplying={presetApplyState === 'applying'}
        onClose={() => {
          setPresetModalOpen(false)
          setPresetApplyError(null)
        }}
        onSelectPreset={(presetId) => {
          setSelectedRoutingPresetId(presetId)
          setPresetApplyError(null)
        }}
        onApply={() => void handleApplyRoutingPreset()}
      />
    </div>
  )
}

export default ConfigPage
