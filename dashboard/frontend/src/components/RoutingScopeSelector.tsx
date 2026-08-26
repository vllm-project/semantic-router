import { useId } from 'react'

import type { RoutingScope } from '../utils/routingScopes'
import ProductIcon from './ProductIcon'
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
  const selectId = useId()
  const detailId = `${selectId}-detail`
  const selected = scopes.find((scope) => scope.id === value) ?? scopes[0]
  if (!selected) return null

  const detail =
    selected.entrypointModelNames.length > 0
      ? selected.entrypointModelNames.join(', ')
      : selected.description || 'Draft Recipe'

  return (
    <div className={styles.root}>
      <div className={styles.heading}>
        <label className={styles.label} htmlFor={selectId}>
          {label}
        </label>
        <span className={styles.detail} id={detailId} title={detail}>
          {detail}
        </span>
      </div>
      <div className={styles.selectShell}>
        <select
          id={selectId}
          value={selected.id}
          onChange={(event) => onChange(event.target.value)}
          aria-describedby={detailId}
        >
          {scopes.map((scope) => (
            <option key={scope.id} value={scope.id}>
              {scope.label}
            </option>
          ))}
        </select>
        <ProductIcon name="chevron-down" aria-hidden="true" />
      </div>
    </div>
  )
}
