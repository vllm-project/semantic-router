import { useId } from 'react'

import type { RoutingScope } from '../utils/routingScopes'
import styles from './RoutingScopeSelector.module.css'

interface RoutingScopeSelectorProps {
  label?: string
  onChange: (scopeId: string) => void
  scopes: RoutingScope[]
  value: string
}

export default function RoutingScopeSelector({
  label = 'Routing profile',
  onChange,
  scopes,
  value,
}: RoutingScopeSelectorProps) {
  const selectId = useId()
  const selected = scopes.find((scope) => scope.id === value) ?? scopes[0]
  if (!selected) return null

  return (
    <div className={styles.root}>
      <label htmlFor={selectId}>{label}</label>
      <select
        id={selectId}
        value={selected.id}
        onChange={(event) => onChange(event.target.value)}
      >
        {scopes.map((scope) => (
          <option key={scope.id} value={scope.id}>
            {scope.label}
          </option>
        ))}
      </select>
      <span className={styles.detail}>
        {selected.entrypointModelNames.length > 0
          ? selected.entrypointModelNames.join(', ')
          : selected.description || 'Top-level automatic routing'}
      </span>
    </div>
  )
}
