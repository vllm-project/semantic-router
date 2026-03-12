import { useMemo } from 'react'

import type { ToolCall, ToolResult } from '../tools'
import { parseMCPToolName } from '../tools/mcp'

import styles from './ChatComponent.module.css'
import {
  buildClawRequestHighlights,
  buildClawResultHighlights,
  truncateHighlight,
} from './ChatComponentTypes'
import { OpenWebCard, WebSearchCard } from './ChatComponentWebToolCards'
import {
  getToolDisplayName,
  getToolStatusLabel,
  getToolSummary,
} from './chatToolCardPresentation'

const TOOL_STATUS_CLASS_NAMES: Record<ToolCall['status'], string> = {
  pending: styles.toolStatusPending,
  running: styles.toolStatusRunning,
  completed: styles.toolStatusCompleted,
  failed: styles.toolStatusFailed,
}

export const ToolCard = ({
  toolCall,
  toolResult,
  isExpanded,
  onToggle
}: {
  toolCall: ToolCall
  toolResult?: ToolResult
  isExpanded: boolean
  onToggle: () => void
}) => {
  const toolName = toolCall.function.name
  const parsedMCPTool = parseMCPToolName(toolName)
  const clawToolName = parsedMCPTool?.toolName || ''
  const displayToolName = getToolDisplayName(clawToolName || toolName)
  const isClawMCPToolCall = clawToolName.startsWith('claw_')
  const isClawCreateToolCall = clawToolName === 'claw_create_team' || clawToolName === 'claw_create_worker'
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
  const requestHighlights = useMemo(
    () => (isClawCreateToolCall ? buildClawRequestHighlights(clawToolName, parsedArgs, rawArgs) : []),
    [clawToolName, isClawCreateToolCall, parsedArgs, rawArgs]
  )
  const resultHighlights = useMemo(
    () => (isClawCreateToolCall ? buildClawResultHighlights(clawToolName, toolResult?.content, parsedArgs, rawArgs) : []),
    [clawToolName, isClawCreateToolCall, parsedArgs, rawArgs, toolResult?.content]
  )
  const showResultHighlights = isClawCreateToolCall && (toolCall.status === 'completed' || toolCall.status === 'failed')
  const statusLabel = getToolStatusLabel(toolCall.status)
  const summary = getToolSummary(clawToolName || toolName, parsedArgs, isClawMCPToolCall)

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
    <div className={`${styles.webSearchCard} ${isClawMCPToolCall ? styles.mcpToolCard : ''}`}>
      <div className={styles.webSearchHeader} onClick={onToggle}>
        <div className={`${styles.webSearchIcon} ${isClawMCPToolCall ? styles.mcpToolIcon : ''}`}>
          {isClawMCPToolCall ? (
            <img src="/openclaw.svg" alt="" aria-hidden="true" />
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
            </svg>
          )}
        </div>
        <div className={styles.webSearchInfo}>
          <div className={styles.toolCardHeadingRow}>
            <span className={styles.webSearchTitle}>{displayToolName}</span>
            {isClawMCPToolCall ? <span className={styles.toolCardBrand}>HireClaw</span> : null}
          </div>
          <span className={styles.webSearchQuery}>{summary}</span>
        </div>
        <div className={styles.webSearchStatus}>
          <span className={`${styles.toolStatusPill} ${TOOL_STATUS_CLASS_NAMES[toolCall.status]}`}>
            {statusLabel}
          </span>
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
      </div>
      {toolCall.status === 'running' || toolCall.status === 'pending' ? (
        <div className={styles.webSearchLoading}>
          <div className={`${styles.webSearchLoadingBar} ${isClawMCPToolCall ? styles.mcpToolLoadingBar : ''}`} />
        </div>
      ) : null}
      {isExpanded && isClawCreateToolCall ? (
        <div className={styles.clawToolHighlights}>
          {requestHighlights.length > 0 && (
            <div className={styles.clawToolHighlightSection}>
              <span className={styles.clawToolHighlightHeading}>Request</span>
              <div className={styles.clawToolHighlightRows}>
                {requestHighlights.map(item => (
                  <div key={`request-${item.label}`} className={styles.clawToolHighlightRow}>
                    <span className={styles.clawToolHighlightKey}>{item.label}</span>
                    <span className={styles.clawToolHighlightValue}>{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {showResultHighlights && (resultHighlights.length > 0 || Boolean(toolResult?.error)) && (
            <div className={styles.clawToolHighlightSection}>
              <span className={styles.clawToolHighlightHeading}>Result</span>
              <div className={styles.clawToolHighlightRows}>
                {resultHighlights.map(item => (
                  <div key={`result-${item.label}`} className={styles.clawToolHighlightRow}>
                    <span className={styles.clawToolHighlightKey}>{item.label}</span>
                    <span className={styles.clawToolHighlightValue}>{item.value}</span>
                  </div>
                ))}
                {toolResult?.error && (
                  <div className={styles.clawToolHighlightRow}>
                    <span className={styles.clawToolHighlightKey}>error</span>
                    <span className={`${styles.clawToolHighlightValue} ${styles.clawToolHighlightError}`}>
                      {truncateHighlight(toolResult.error, 180)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      ) : null}
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
    </div>
  )
}
