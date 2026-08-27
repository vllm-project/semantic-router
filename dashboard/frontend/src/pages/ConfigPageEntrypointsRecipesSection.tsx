import { useRef, useState, type KeyboardEvent } from 'react'
import { useNavigate } from 'react-router-dom'

import type { RecipeProbeRunPlan } from '../types/recipe'
import { createProbePlaygroundInvocation } from '../types/playgroundInvocation'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageMoMProbesPanel from './ConfigPageMoMProbesPanel'
import ConfigPageMoMRoutingPanel from './ConfigPageMoMRoutingPanel'
import styles from './ConfigPageMoMWorkspace.module.css'
import type { ConfigData, NormalizedModel } from './configPageSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'

interface ConfigPageEntrypointsRecipesSectionProps {
  config: ConfigData
  isReadonly: boolean
  models: NormalizedModel[]
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
}

export type MixtureWorkspaceView = 'recipes' | 'models' | 'probes'

const VIEWS: Array<{ id: MixtureWorkspaceView; label: string }> = [
  { id: 'models', label: 'Models' },
  { id: 'recipes', label: 'Recipes' },
  { id: 'probes', label: 'Probes' },
]

export default function ConfigPageEntrypointsRecipesSection({
  config,
  isReadonly,
  models,
  saveConfig,
  openEditModal,
  openViewModal,
}: ConfigPageEntrypointsRecipesSectionProps) {
  const navigate = useNavigate()
  const [activeView, setActiveView] = useState<MixtureWorkspaceView>('models')
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])

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

  return (
    <ConfigPageManagerLayout
      title="Mixture-of-Models"
      description="Design a recipe. Publish one model."
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
        {activeView === 'recipes' || activeView === 'models' ? (
          <ConfigPageMoMRoutingPanel
            activeView={activeView}
            config={config}
            isReadonly={isReadonly}
            models={models}
            saveConfig={saveConfig}
            openEditModal={openEditModal}
            openViewModal={openViewModal}
          />
        ) : null}
        {activeView === 'probes' ? (
          <ConfigPageMoMProbesPanel onLaunch={launchProbe} recipeRevision={0} />
        ) : null}
      </div>
    </ConfigPageManagerLayout>
  )
}
