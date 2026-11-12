import { useMemo } from 'react'
import { ThinkingOrb } from 'thinking-orbs'

import type { ToolCall, ToolResult } from '../tools'

import styles from './ChatComponent.module.css'
import { OpenWebCard, WebSearchCard } from './ChatComponentWebToolCards'
import { getToolDisplayName, getToolStatusLabel, getToolSummary } from './chatToolCardPresentation'

const TOOL_STATUS_CLASS_NAMES: Record<ToolCall['status'], string> = {
  pending: styles.toolStatusPending,
  running: styles.toolStatusRunning,
  completed: styles.toolStatusCompleted,
  failed: styles.toolStatusFailed,
  skipped: styles.toolStatusSkipped,
}

function buildResultPreview(toolResult?: ToolResult) {
  if (!toolResult || toolResult.error) {
    return ''
  }

  if (typeof toolResult.content === 'string') {
    return toolResult.content
  }

  try {
    return JSON.stringify(toolResult.content, null, 2)
  } catch {
    return String(toolResult.content ?? '')
  }
}

export const ToolCard = ({
  toolCall,
  toolResult,
  isExpanded,
  onToggle,
}: {
  toolCall: ToolCall
  toolResult?: ToolResult
  isExpanded: boolean
  onToggle: () => void
}) => {
  const toolName = toolCall.function.name
  const displayToolName = getToolDisplayName(toolName)
  const rawArgs = toolCall.function.arguments || ''
  const parsedArgs = useMemo(() => {
    try {
      const parsed = JSON.parse(rawArgs)
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null
      return parsed as Record<string, unknown>
    } catch {
      return null
    }
  }, [rawArgs])
  const statusLabel = getToolStatusLabel(toolCall.status)
  const summary = getToolSummary(parsedArgs)
  const resultPreview = useMemo(() => buildResultPreview(toolResult), [toolResult])

  if (toolName === 'search_web') {
    return (
      <WebSearchCard
        toolCall={toolCall}
        toolResult={toolResult}
        isExpanded={isExpanded}
        onToggle={onToggle}
      />
    )
  }

  if (toolName === 'open_web') {
    return (
      <OpenWebCard
        toolCall={toolCall}
        toolResult={toolResult}
        isExpanded={isExpanded}
        onToggle={onToggle}
      />
    )
  }

  return (
    <div className={styles.webSearchCard}>
      <button
        type="button"
        className={styles.webSearchHeader}
        onClick={onToggle}
        aria-expanded={isExpanded}
        aria-label={`${isExpanded ? 'Collapse' : 'Expand'} details for ${displayToolName}`}
      >
        <div className={styles.webSearchIcon}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
          </svg>
        </div>
        <div className={styles.webSearchInfo}>
          <div className={styles.toolCardHeadingRow}>
            <span className={styles.webSearchTitle}>{displayToolName}</span>
          </div>
          <span className={styles.webSearchQuery}>{summary}</span>
        </div>
        <div className={styles.webSearchStatus}>
          {toolCall.status === 'running' || toolCall.status === 'pending' ? (
            <span className={styles.toolActivity} aria-label={statusLabel}>
              <ThinkingOrb state="working" size={20} theme="dark" />
              {statusLabel}
            </span>
          ) : (
            <span
              className={`${styles.toolStatusPill} ${TOOL_STATUS_CLASS_NAMES[toolCall.status]}`}
            >
              {statusLabel}
            </span>
          )}
          <svg
            className={`${styles.webSearchChevron} ${isExpanded ? styles.expanded : ''}`}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </button>
      {isExpanded && toolCall.status === 'failed' && toolResult?.error && (
        <div className={styles.webSearchResults}>
          <div className={styles.sourceDetails}>
            <div className={styles.sourceItem}>
              <p className={styles.sourceItemSnippet} style={{ color: 'var(--color-error)' }}>
                {toolResult.error}
              </p>
            </div>
          </div>
        </div>
      )}
      {isExpanded && toolCall.status === 'completed' && resultPreview && (
        <div className={styles.webSearchResults}>
          <div className={styles.sourceDetails}>
            <div className={styles.sourceItem}>
              <pre className={styles.openWebContent}>{resultPreview}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
