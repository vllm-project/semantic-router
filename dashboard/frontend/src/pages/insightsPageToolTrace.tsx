import type { ReactNode } from 'react'

import type { InsightsRecord, ToolTrace } from './insightsPageTypes'
import styles from './InsightsPage.module.css'

export function renderToolNamesCell(record: InsightsRecord): ReactNode {
  const toolNames = getTraceToolNames(record.tool_trace)
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
    <span className={styles.tableSummaryText} title={toolNames.join(', ')}>
      {summary}
    </span>
  )
}

function getTraceToolNames(trace?: ToolTrace) {
  if (!trace) {
    return []
  }

  const toolNames = new Set<string>(trace.tool_names ?? [])
  for (const step of trace.steps ?? []) {
    if (step.tool_name) {
      toolNames.add(step.tool_name)
    }
  }
  return [...toolNames]
}
