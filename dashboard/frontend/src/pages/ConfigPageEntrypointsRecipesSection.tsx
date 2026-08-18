import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { useNavigate } from 'react-router-dom'

import type { RecipeProbeRunPlan } from '../types/recipe'
import { createProbePlaygroundInvocation } from '../types/playgroundInvocation'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageMoMOverviewPanel from './ConfigPageMoMOverviewPanel'
import ConfigPageMoMProbesPanel from './ConfigPageMoMProbesPanel'
import ConfigPageMoMRoutingPanel from './ConfigPageMoMRoutingPanel'
import styles from './ConfigPageMoMWorkspace.module.css'
import type { RecipePackageCapabilities } from './configPageMoMPackagesSupport'
import type { ConfigData, NormalizedModel } from './configPageSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { useBuiltInModelCatalog } from './useBuiltInModelCatalog'

interface ConfigPageEntrypointsRecipesSectionProps {
  config: ConfigData
  isReadonly: boolean
  models: NormalizedModel[]
  packageCapabilities: RecipePackageCapabilities
  refreshConfig: () => Promise<boolean>
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
}

type MoMView = 'overview' | 'routing' | 'probes'

const VIEWS: Array<{ id: MoMView; label: string }> = [
  { id: 'overview', label: 'Built-in Models' },
  { id: 'routing', label: 'Models & Routing' },
  { id: 'probes', label: 'Probes' },
]

export default function ConfigPageEntrypointsRecipesSection({
  config,
  isReadonly,
  models,
  packageCapabilities,
  refreshConfig,
  saveConfig,
  openEditModal,
  openViewModal,
}: ConfigPageEntrypointsRecipesSectionProps) {
  const navigate = useNavigate()
  const [activeView, setActiveView] = useState<MoMView>('overview')
  const [recipeRevision, setRecipeRevision] = useState(0)
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])
  const modelCatalog = useBuiltInModelCatalog()

  useEffect(() => {
    const refreshRecipeViews = () => setRecipeRevision((current) => current + 1)
    window.addEventListener('config-deployed', refreshRecipeViews)
    return () => window.removeEventListener('config-deployed', refreshRecipeViews)
  }, [])

  const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let nextIndex: number | null = null
    if (event.key === 'ArrowRight') nextIndex = (index + 1) % VIEWS.length
    if (event.key === 'ArrowLeft') nextIndex = (index - 1 + VIEWS.length) % VIEWS.length
    if (event.key === 'Home') nextIndex = 0
    if (event.key === 'End') nextIndex = VIEWS.length - 1
    if (nextIndex === null) return
    event.preventDefault()
    setActiveView(VIEWS[nextIndex].id)
    tabRefs.current[nextIndex]?.focus()
  }

  const launchProbe = (intent: 'run' | 'edit', plan: RecipeProbeRunPlan) => {
    navigate('/playground', {
      state: {
        playgroundInvocation: createProbePlaygroundInvocation(intent, plan),
      },
    })
  }

  const handleRecipeLifecycleChanged = async (): Promise<boolean> => {
    const configRefreshed = await refreshConfig()
    setRecipeRevision((current) => current + 1)
    window.dispatchEvent(new CustomEvent('config-deployed'))
    return configRefreshed
  }

  return (
    <ConfigPageManagerLayout
      eyebrow="Dispatch"
      title="Mixture-of-Models"
      description="Browse versioned built-in models, inspect deployed routing policies, and run verified offline probes."
      configArea="Multi-recipe dispatch"
      scope="Active Recipe"
      panelEyebrow="Unified model workspace"
      panelTitle="One model surface, many model paths"
      panelDescription="Inspect installed model metadata, manage custom request-facing IDs and routing, then exercise verified requests in Playground."
      pills={[
        { label: 'Models' },
        { label: 'Decisions' },
        { label: 'Mixture-of-Models', active: true },
      ]}
    >
      <div className={styles.tabs} role="tablist" aria-label="Mixture-of-Models views">
        {VIEWS.map((view, index) => (
          <button
            key={view.id}
            ref={(element) => {
              tabRefs.current[index] = element
            }}
            id={`mom-tab-${view.id}`}
            type="button"
            role="tab"
            aria-selected={activeView === view.id}
            aria-controls="mom-active-panel"
            tabIndex={activeView === view.id ? 0 : -1}
            className={`${styles.tab} ${activeView === view.id ? styles.activeTab : ''}`}
            onClick={() => setActiveView(view.id)}
            onKeyDown={(event) => handleTabKeyDown(event, index)}
          >
            {view.label}
          </button>
        ))}
      </div>

      <div
        id="mom-active-panel"
        className={styles.tabPanel}
        role="tabpanel"
        aria-labelledby={`mom-tab-${activeView}`}
      >
        {activeView === 'overview' ? (
          <ConfigPageMoMOverviewPanel
            catalog={modelCatalog.catalog}
            catalogLoading={modelCatalog.loading}
            catalogError={modelCatalog.error}
            onCatalogRetry={modelCatalog.retry}
            packageCapabilities={packageCapabilities}
            recipeRevision={recipeRevision}
            onRecipeLifecycleChanged={handleRecipeLifecycleChanged}
          />
        ) : null}
        {activeView === 'routing' ? (
          <ConfigPageMoMRoutingPanel
            config={config}
            isReadonly={isReadonly}
            models={models}
            saveConfig={saveConfig}
            openEditModal={openEditModal}
            openViewModal={openViewModal}
            catalog={modelCatalog.catalog}
            catalogLoading={modelCatalog.loading}
            catalogError={modelCatalog.error}
            onCatalogRetry={modelCatalog.retry}
          />
        ) : null}
        {activeView === 'probes' ? (
          <ConfigPageMoMProbesPanel onLaunch={launchProbe} recipeRevision={recipeRevision} />
        ) : null}
      </div>
    </ConfigPageManagerLayout>
  )
}
