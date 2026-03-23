import React, { useState, useEffect } from 'react'
import styles from './ConfigPage.module.css'
import { ConfigSection } from '../components/ConfigNav'
import EditModal, { type EditFormData, FieldConfig } from '../components/EditModal'
import ViewModal, { ViewSection } from '../components/ViewModal'
import { useReadonly } from '../contexts/ReadonlyContext'
import ConfigPageRouterConfigSection from './ConfigPageRouterConfigSection'
import ConfigPageModelsSection from './ConfigPageModelsSection'
import ConfigPageSignalsSection from './ConfigPageSignalsSection'
import ConfigPageProjectionsSection from './ConfigPageProjectionsSection'
import ConfigPageDecisionsSection from './ConfigPageDecisionsSection'
import ConfigPageMCPSection from './ConfigPageMCPSection'
import {
  canonicalizeConfigForManagerSave,
  projectCanonicalConfigForManager,
} from './configPageCanonicalization'
import {
  ConfigFormat,
  detectConfigFormat,
} from '../types/config'
import {
  CanonicalGlobalConfig,
  ConfigData,
  SignalType,
  Tool,
  getDefaultModelName,
  getNormalizedModels,
  getReasoningFamiliesMap,
} from './configPageSupport'
import type { OpenViewModal } from './configPageRouterSectionSupport'

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
  const [editModalData, setEditModalData] = useState<EditFormData | null>(null)
  const [editModalFields, setEditModalFields] = useState<FieldConfig[]>([])
  const [editModalMode, setEditModalMode] = useState<'edit' | 'add'>('edit')
  const [editModalCallback, setEditModalCallback] = useState<((data: EditFormData) => Promise<void>) | null>(null)

  // View modal state
  const [viewModalOpen, setViewModalOpen] = useState(false)
  const [viewModalTitle, setViewModalTitle] = useState('')
  const [viewModalSections, setViewModalSections] = useState<ViewSection[]>([])
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

  const saveConfig = async (updatedConfig: ConfigData) => {
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

  const openEditModal = <TForm extends object>(
    title: string,
    data: TForm,
    fields: FieldConfig<TForm>[],
    callback: (data: TForm) => Promise<void>,
    mode: 'edit' | 'add' = 'edit'
  ) => {
    setEditModalTitle(title)
    setEditModalData(data as EditFormData)
    setEditModalFields(fields as FieldConfig[])
    setEditModalMode(mode)
    setEditModalCallback(() => async (rawData: EditFormData) => callback(rawData as TForm))
    setEditModalOpen(true)
  }

  const closeEditModal = () => {
    setEditModalOpen(false)
    setEditModalData(null)
    setEditModalFields([])
    setEditModalCallback(null)
  }

  const openViewModal: OpenViewModal = (title, sections, onEdit) => {
    setViewModalTitle(title)
    setViewModalSections(sections)
    setViewModalEditCallback(() => onEdit || null)
    setViewModalOpen(true)
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
      case 'Structure':
        cfg.signals.structure = (cfg.signals.structure || []).filter(s => s.name !== targetName)
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
  const models = getNormalizedModels(config, isPythonCLI)
  const defaultModel = getDefaultModelName(config, isPythonCLI)
  const reasoningFamilies = getReasoningFamiliesMap(config, isPythonCLI)

  // ============================================================================
  // SECTION PANEL RENDERS - Aligned with Python CLI config structure
  // ============================================================================

  const renderSignalsSection = () => (
    <ConfigPageSignalsSection
      config={config}
      isPythonCLI={isPythonCLI}
      isReadonly={isReadonly}
      signalsSearch={signalsSearch}
      onSignalsSearchChange={setSignalsSearch}
      saveConfig={saveConfig}
      openEditModal={openEditModal}
      openViewModal={openViewModal}
      listInputToArray={listInputToArray}
      removeSignalByName={removeSignalByName}
    />
  )

  const renderDecisionsSection = () => (
    <ConfigPageDecisionsSection
      config={config}
      isPythonCLI={isPythonCLI}
      isReadonly={isReadonly}
      decisionsSearch={decisionsSearch}
      onDecisionsSearchChange={setDecisionsSearch}
      saveConfig={saveConfig}
      openEditModal={openEditModal}
      openViewModal={openViewModal}
      removeDecisionByName={removeDecisionByName}
      models={models}
    />
  )

  const renderProjectionsSection = () => (
    <ConfigPageProjectionsSection
      config={config}
      isReadonly={isReadonly}
      saveConfig={saveConfig}
      openEditModal={openEditModal}
      openViewModal={openViewModal}
    />
  )

  const renderModelsSection = () => (
    <ConfigPageModelsSection
      config={config}
      isPythonCLI={isPythonCLI}
      isReadonly={isReadonly}
      models={models}
      defaultModel={defaultModel}
      reasoningFamilies={reasoningFamilies}
      modelsSearch={modelsSearch}
      onModelsSearchChange={setModelsSearch}
      expandedModels={expandedModels}
      onExpandedModelsChange={setExpandedModels}
      saveConfig={saveConfig}
      openEditModal={openEditModal}
      openViewModal={openViewModal}
      listInputToArray={listInputToArray}
    />
  )

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
      case 'projections':
        return renderProjectionsSection()
      case 'models':
        return renderModelsSection()
      case 'global-config':
        return renderGlobalConfigSection()
      case 'mcp':
        return <ConfigPageMCPSection />
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
    </div>
  )
}

export default ConfigPage
