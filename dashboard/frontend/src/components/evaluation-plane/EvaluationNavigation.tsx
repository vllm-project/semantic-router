import { useEffect, useRef, type KeyboardEvent } from 'react'

import type { EvaluationView } from '../../pages/evaluationRoute'
import ProductIcon, { type ProductIconName } from '../ProductIcon'
import styles from './EvaluationNavigation.module.css'

const VIEWS: Array<{ id: EvaluationView; label: string; icon: ProductIconName }> = [
  { id: 'overview', label: 'Overview', icon: 'dashboard' },
  { id: 'new', label: 'New experiment', icon: 'plus' },
  { id: 'runs', label: 'Runs', icon: 'list' },
  { id: 'reports', label: 'Reports', icon: 'chart' },
  { id: 'compare', label: 'Compare', icon: 'decision' },
]

interface EvaluationNavigationProps {
  active: EvaluationView
  onChange: (view: EvaluationView) => void
}

export default function EvaluationNavigation({ active, onChange }: EvaluationNavigationProps) {
  const navigation = useRef<HTMLDivElement | null>(null)
  const buttons = useRef<Array<HTMLButtonElement | null>>([])
  const activeIndex = VIEWS.findIndex((view) => view.id === active)

  useEffect(() => {
    const activeButton = buttons.current[activeIndex]
    if (!activeButton || !navigation.current) return
    navigation.current.scrollLeft =
      activeButton.offsetLeft - (navigation.current.clientWidth - activeButton.offsetWidth) / 2
  }, [activeIndex])

  const move = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let next = index
    if (event.key === 'ArrowRight') next = (index + 1) % VIEWS.length
    else if (event.key === 'ArrowLeft') next = (index - 1 + VIEWS.length) % VIEWS.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = VIEWS.length - 1
    else return
    event.preventDefault()
    onChange(VIEWS[next].id)
    requestAnimationFrame(() => buttons.current[next]?.focus())
  }

  return (
    <div className={styles.navigationShell}>
      <div
        ref={navigation}
        className={styles.navigation}
        role="tablist"
        aria-label="Evaluation views"
      >
        {VIEWS.map((view, index) => (
          /* Tabs retain their compact, underlined navigation treatment rather than action-button styling. */
          <button
            key={view.id}
            id={`evaluation-tab-${view.id}`}
            type="button"
            role="tab"
            data-evaluation-navigation-tab="true"
            aria-selected={active === view.id}
            aria-controls="evaluation-panel"
            tabIndex={active === view.id ? 0 : -1}
            className={`${styles.navigationButton} ${active === view.id ? styles.navigationActive : ''}`}
            onClick={() => onChange(view.id)}
            onKeyDown={(event) => move(event, index)}
            ref={(node) => {
              buttons.current[index] = node
            }}
          >
            <ProductIcon name={view.icon} />
            <span>{view.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
