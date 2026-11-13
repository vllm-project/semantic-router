import React, { lazy, Suspense, useState, useEffect } from 'react'
import styles from './ConfigPage.module.css'
import { ConfigSection } from '../components/ConfigNav'
import EditModal, { type EditFormData, FieldConfig } from '../components/EditModal'
import ViewModal, { ViewSection } from '../components/ViewModal'
import { useReadonly } from '../contexts/ReadonlyContext'
import { useAuth } from '../contexts/AuthContext'
import { canRunEvaluation, canWriteConfig } from '../utils/accessControl'
import ConfigSaveWarnings from '../components/ConfigSaveWarnings'
import ProductLoadingState from '../components/ProductLoadingState'
import { responseConfigWarnings, type RouterConfigWarning } from '../utils/configSaveWarnings'
import {
  canonicalizeConfigForManagerSave,
  projectCanonicalConfigForManager,
} from './configPageCanonicalization'
import {
  ConfigData,
  getDefaultModelName,
  getReasoningFamiliesMap,
} from './configPageSupport'
import { getNormalizedModels } from './configPageModelNormalization'
import { responseErrorMessage } from './configPageRequestErrors'
import type { OpenViewModal } from './configPageRouterSectionSupport'
import { withRequestTimeout } from '../utils/boundedRequest'

const ConfigPageRouterConfigSection = lazy(() => import('./ConfigPageRouterConfigSection'))
const ConfigPageModelsSection = lazy(() => import('./ConfigPageModelsSection'))
const ConfigPageSignalsSection = lazy(() => import('./ConfigPageSignalsSection'))
const ConfigPageProjectionsSection = lazy(() => import('./ConfigPageProjectionsSection'))
const ConfigPageDecisionsSection = lazy(() => import('./ConfigPageDecisionsSection'))
const ConfigPageEntrypointsRecipesSection = lazy(
  () => import('./ConfigPageEntrypointsRecipesSection'),
)
const ConfigPageMCPSection = lazy(() => import('./ConfigPageMCPSection'))

interface ConfigPageProps {
  activeSection?: ConfigSection
}

// Removed maskAddress - no longer needed after removing endpoint visibility toggle

const ConfigPage: React.FC<ConfigPageProps> = ({ activeSection = 'global-config' }) => {
  const { isReadonly } = useReadonly()
  const { user } = useAuth()
  const isMCPSection = activeSection === 'mcp'
  const configReadonly = isReadonly || !canWriteConfig(user)
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [loading, setLoading] = useState(!isMCPSection)
  const [error, setError] = useState<string | null>(null)
  const configEditorReadonly = configReadonly

  // Effective global runtime config resolved from router defaults + config.yaml overrides

  // Tools database state
  const [saveWarnings, setSaveWarnings] = useState<RouterConfigWarning[]>([])

  // Removed visibleAddresses state - no longer needed

  // Edit modal state
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [editModalTitle, setEditModalTitle] = useState('')
  const [editModalData, setEditModalData] = useState<EditFormData | null>(null)
  const [editModalFields, setEditModalFields] = useState<FieldConfig[]>([])
  const [editModalMode, setEditModalMode] = useState<'edit' | 'add'>('edit')
  const [editModalCallback, setEditModalCallback] = useState<
    ((data: EditFormData) => Promise<void>) | null
  >(null)

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
    if (isMCPSection) {
      setLoading(false)
      setError(null)
      return
    }

    void fetchConfig()
  }, [isMCPSection])

  const fetchConfig = async (showLoading = true): Promise<boolean> => {
    if (showLoading) setLoading(true)
    setError(null)
    try {
      const data = await withRequestTimeout(async (signal) => {
        const response = await fetch('/api/router/config/all', { signal })
        if (!response.ok) {
          throw new Error(await responseErrorMessage(response))
        }
        return await response.json()
      })
      setConfig(projectCanonicalConfigForManager(data))
      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch config')
      setConfig(null)
      return false
    } finally {
      if (showLoading) setLoading(false)
    }
  }

  const saveConfig = async (updatedConfig: ConfigData) => {
    // Prevent save in read-only mode
    if (configReadonly) {
      throw new Error('Dashboard is in read-only mode. Configuration editing is disabled.')
    }

    try {
      setSaveWarnings([])
      const canonicalConfig = canonicalizeConfigForManagerSave(updatedConfig as ConfigData)
      const response = await fetch('/api/router/config/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(canonicalConfig),
      })

      if (!response.ok) {
        throw new Error(await responseErrorMessage(response))
      }
      setSaveWarnings(await responseConfigWarnings(response))

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
    mode: 'edit' | 'add' = 'edit',
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

  const listInputToArray = (input: string) =>
    input
      .split(/[\n,]/)
      .map((item) => item.trim())
      .filter(Boolean)

  const removeDecisionByName = (cfg: ConfigData, targetName: string) => {
    cfg.decisions = (cfg.decisions || []).filter((d) => d.name !== targetName)
  }

  const handleCloseViewModal = () => {
    setViewModalOpen(false)
    setViewModalTitle('')
    setViewModalSections([])
    setViewModalEditCallback(null)
  }

  const models = getNormalizedModels(config)
  const defaultModel = getDefaultModelName(config)
  const reasoningFamilies = getReasoningFamiliesMap()

  // ============================================================================
  // SECTION PANEL RENDERS
  // ============================================================================

  const renderSignalsSection = () => (
    <ConfigPageSignalsSection
      config={config}
      isReadonly={configEditorReadonly}
      signalsSearch={signalsSearch}
      onSignalsSearchChange={setSignalsSearch}
      saveConfig={saveConfig}
      openEditModal={openEditModal}
      openViewModal={openViewModal}
    />
  )

  const renderDecisionsSection = () => (
    <ConfigPageDecisionsSection
      config={config}
      isReadonly={configEditorReadonly}
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
      isReadonly={configEditorReadonly}
      saveConfig={saveConfig}
      openEditModal={openEditModal}
      openViewModal={openViewModal}
    />
  )

  const renderModelsSection = () => (
    <ConfigPageModelsSection
      config={config}
      isReadonly={configEditorReadonly}
      canVerifyModels={canRunEvaluation(user)}
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

  const renderEntrypointsRecipesSection = () =>
    config ? (
      <ConfigPageEntrypointsRecipesSection
        config={config}
        isReadonly={configEditorReadonly}
        models={models}
        saveConfig={saveConfig}
        openEditModal={openEditModal}
        openViewModal={openViewModal}
      />
    ) : null

  // Global Config section - canonical global override editor backed by effective router defaults
  const renderGlobalConfigSection = () => (
    <ConfigPageRouterConfigSection
      config={config}
      isReadonly={configEditorReadonly}
      openEditModal={openEditModal}
      refreshConfig={async () => {
        await fetchConfig()
      }}
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
      case 'entrypoints-recipes':
        return renderEntrypointsRecipesSection()
      case 'global-config':
        return renderGlobalConfigSection()
      case 'mcp':
        return <ConfigPageMCPSection />
      default:
        return renderGlobalConfigSection()
    }
  }

  const pageLoading = loading

  return (
    <div className={styles.container}>
      <div className={styles.content}>
        {!isMCPSection && pageLoading && <ProductLoadingState label="Loading configuration" />}

        {!isMCPSection && error && !pageLoading && (
          <div className={styles.error}>
            <span className={styles.errorIcon}></span>
            <div>
              <h3>Error Loading Config</h3>
              <p>{error}</p>
              <button type="button" className={styles.button} onClick={() => fetchConfig()}>
                Retry
              </button>
            </div>
          </div>
        )}

        {isMCPSection && (
          <div className={styles.contentArea}>
            <Suspense fallback={<ProductLoadingState label="Loading configuration" />}>
              <ConfigPageMCPSection />
            </Suspense>
          </div>
        )}

        {!isMCPSection && config && !pageLoading && !error && (
          <div className={styles.contentArea}>
            <ConfigSaveWarnings warnings={saveWarnings} />
            <Suspense fallback={<ProductLoadingState label="Loading configuration" />}>
              {renderActiveSection()}
            </Suspense>
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
        onClose={handleCloseViewModal}
        onEdit={configEditorReadonly ? undefined : viewModalEditCallback || undefined}
        title={viewModalTitle}
        sections={viewModalSections}
      />
    </div>
  )
}

export default ConfigPage
