import type { RoutingScope } from '../utils/routingScopes'
import styles from './RoutingScopeSelector.module.css'

interface RoutingScopeSelectorProps {
  label?: string
  onChange: (scopeId: string) => void
  scopes: RoutingScope[]
  value: string
}

export default function RoutingScopeSelector({
  label = 'Recipe',
  onChange,
  scopes,
  value,
}: RoutingScopeSelectorProps) {
  const selected = scopes.find((scope) => scope.id === value) ?? scopes[0]
  if (!selected) return null

  return (
    <div className={styles.root}>
      <div className={styles.heading}>
        <span className={styles.label}>{label}</span>
        <span className={styles.detail}>
          {selected.entrypointModelNames.length > 0
            ? selected.entrypointModelNames.join(', ')
            : selected.description || 'Draft Recipe'}
        </span>
      </div>
      <div className={styles.tabs} role="group" aria-label={label}>
        {scopes.map((scope) => (
          <button
            key={scope.id}
            type="button"
            aria-pressed={scope.id === selected.id}
            className={scope.id === selected.id ? styles.activeTab : styles.tab}
            onClick={() => onChange(scope.id)}
          >
            {scope.label}
          </button>
        ))}
      </div>
    </div>
  )
}
