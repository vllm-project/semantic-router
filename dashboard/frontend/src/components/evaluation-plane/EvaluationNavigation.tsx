import ProductIcon, { type ProductIconName } from '../ProductIcon'
import styles from './EvaluationPlane.module.css'

export type EvaluationView = 'overview' | 'new' | 'runs' | 'reports' | 'compare'

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
  return (
    <div className={styles.navigation} role="tablist" aria-label="Evaluation plane views">
      {VIEWS.map((view) => (
        <button
          key={view.id}
          id={`evaluation-tab-${view.id}`}
          type="button"
          role="tab"
          aria-selected={active === view.id}
          aria-controls={`evaluation-panel-${view.id}`}
          tabIndex={active === view.id ? 0 : -1}
          className={`${styles.navigationButton} ${active === view.id ? styles.navigationActive : ''}`}
          onClick={() => onChange(view.id)}
        >
          <ProductIcon name={view.icon} />
          <span>{view.label}</span>
        </button>
      ))}
    </div>
  )
}
