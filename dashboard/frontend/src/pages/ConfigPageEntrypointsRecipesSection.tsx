import { useRef, useState, type KeyboardEvent } from 'react'

import ProductIcon from '../components/ProductIcon'
import { useAuth } from '../contexts/AuthContext'
import { canManageRouting, canReadRouting } from '../utils/accessControl'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageMoMRoutingPanel from './ConfigPageMoMRoutingPanel'
import styles from './ConfigPageMoMWorkspace.module.css'

export type MixtureWorkspaceView = 'recipes' | 'models'

const VIEWS: Array<{ id: MixtureWorkspaceView; label: string }> = [
  { id: 'recipes', label: 'Recipes' },
  { id: 'models', label: 'Models' },
]

export default function ConfigPageEntrypointsRecipesSection() {
  const { user, refreshSession } = useAuth()
  const [activeView, setActiveView] = useState<MixtureWorkspaceView>('recipes')
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])
  const canRead = canReadRouting(user)
  const canManage = canManageRouting(user)
  const identityError = user?.managementIdentityError

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

  return (
    <ConfigPageManagerLayout
      title="Mixture-of-Models"
      description="Design a recipe. Publish one model."
    >
      <div className={styles.tabs} role="tablist" aria-label="Mixture-of-Models">
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
        {identityError ? (
          <section className={styles.identityError} role="alert">
            <ProductIcon name="alert" />
            <div>
              <strong>Routing access unavailable</strong>
              <p>{identityError}</p>
            </div>
            <button type="button" onClick={() => void refreshSession()}>
              <ProductIcon name="refresh" />
              Try again
            </button>
          </section>
        ) : (
          <ConfigPageMoMRoutingPanel
            activeView={activeView}
            canRead={canRead}
            canManage={canManage}
          />
        )}
      </div>
    </ConfigPageManagerLayout>
  )
}
