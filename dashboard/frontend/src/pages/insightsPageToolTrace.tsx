import type { ReactNode } from 'react'

import { traceToolNames } from './insightsAgentIdentity'
import type { InsightsRecord } from './insightsPageTypes'

export function renderToolNamesCell(record: InsightsRecord): ReactNode {
  const toolNames = traceToolNames(record.tool_trace)
  if (toolNames.length === 0) {
    return <span>-</span>
  }

  const visibleToolNames = toolNames.slice(0, 2)
  const hiddenCount = toolNames.length - visibleToolNames.length
  const summary =
    hiddenCount > 0
      ? `${visibleToolNames.join(' · ')} · +${hiddenCount}`
      : visibleToolNames.join(' · ')

  return (
    <span className="truncate text-xs text-muted-foreground" title={toolNames.join(', ')}>
      {summary}
    </span>
  )
}
