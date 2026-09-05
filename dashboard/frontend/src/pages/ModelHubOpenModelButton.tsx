import React from 'react'

import type { ModelHubRow } from './modelHubSupport'
import styles from './ModelHubPage.module.css'

export const OpenModelButton: React.FC<{
  row: ModelHubRow
  selected: ModelHubRow | null
  select: (id: string) => void
  className: string
  ariaLabel?: string
  children: React.ReactNode
}> = ({ row, selected, select, className, ariaLabel, children }) => (
  <button
    type="button"
    className={`${className} ${selected?.model.id === row.model.id ? styles.selected : ''}`}
    onClick={() => select(row.model.id)}
    aria-label={ariaLabel ?? `Inspect ${row.model.display_name}`}
    aria-pressed={selected?.model.id === row.model.id}
  >
    {children}
  </button>
)
