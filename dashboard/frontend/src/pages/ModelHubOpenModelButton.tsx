import React from 'react'

import type { ModelHubRow } from './modelHubSupport'

export const OpenModelButton: React.FC<{
  row: ModelHubRow
  selected?: ModelHubRow | null
  select: (id: string) => void
  className: string
  ariaLabel?: string
  style?: React.CSSProperties
  children: React.ReactNode
}> = ({ row, selected, select, className, ariaLabel, style, children }) => (
  <button
    type="button"
    className={className}
    onClick={() => select(row.model.id)}
    aria-label={ariaLabel ?? `Inspect ${row.model.display_name}`}
    aria-pressed={selected === undefined ? undefined : selected?.model.id === row.model.id}
    style={style}
  >
    {children}
  </button>
)
