import { useMemo } from 'react'

import type { ToolCall, ToolResult } from '../tools'
import { parseMCPToolName } from '../tools/mcp'

import styles from './ChatComponent.module.css'
import {
  buildClawRequestHighlights,
  buildClawResultHighlights,
  type SearchResult,
  truncateHighlight,
} from './ChatComponentTypes'

const WebSearchCard = ({
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
  let query = ''
  try {
    const args = JSON.parse(toolCall.function.arguments || '{}')
    query = args.query || ''
  } catch {
    const match = toolCall.function.arguments?.match(/"query"\s*:\s*"([^"]*)/)
    query = (match && match[1]) || 'Searching...'
  }

  const results = useMemo(() => {
    if (!toolResult?.content) return undefined
    if (Array.isArray(toolResult.content)) {
      return toolResult.content as SearchResult[]
    }
    return undefined
  }, [toolResult?.content])

  return (
    <div className={styles.webSearchCard}>
      <div className={styles.webSearchHeader} onClick={onToggle}>
        <div className={styles.webSearchIcon}>
          {toolCall.status === 'running' ? (
            <svg className={styles.searchSpinner} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <path d="M21 21l-4.35-4.35" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <path d="M21 21l-4.35-4.35" />
            </svg>
          )}
        </div>
        <div className={styles.webSearchInfo}>
          <span className={styles.webSearchTitle}>
            {toolCall.status === 'running' ? 'Searching...' : 'Web Search'}
          </span>
          <span className={styles.webSearchQuery}>"{query}"</span>
        </div>
        <div className={styles.webSearchStatus}>
          {toolCall.status === 'completed' && results && (
            <span className={styles.webSearchCount}>{results.length} sources</span>
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
      </div>

      {isExpanded && toolCall.status === 'completed' && results && results.length > 0 && (
        <div className={styles.webSearchResults}>
          <div className={styles.sourcePills}>
            {results.map((result, idx) => (
              <a
                key={idx}
                href={result.url}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.sourcePill}
                title={result.snippet}
              >
                <span className={styles.sourcePillNumber}>{idx + 1}</span>
                <span className={styles.sourcePillDomain}>{(() => { try { return new URL(result.url).hostname } catch { return result.url } })()}</span>
              </a>
            ))}
          </div>
          <div className={styles.sourceDetails}>
            {results.map((result, idx) => (
              <div key={idx} className={styles.sourceItem}>
                <div className={styles.sourceItemHeader}>
                  <span className={styles.sourceItemNumber}>[{idx + 1}]</span>
                  <a
                    href={result.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={styles.sourceItemTitle}
                  >
                    {result.title}
                  </a>
                </div>
                <p className={styles.sourceItemSnippet}>{result.snippet}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {toolCall.status === 'running' && (
        <div className={styles.webSearchLoading}>
          <div className={styles.webSearchLoadingBar} />
        </div>
      )}
    </div>
  )
}

const OpenWebCard = ({
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
  let url = ''
  try {
    const args = JSON.parse(toolCall.function.arguments || '{}')
    url = args.url || ''
  } catch {
    const match = toolCall.function.arguments?.match(/"url"\s*:\s*"([^"]*)/)
    url = (match && match[1]) || 'Loading...'
  }

  const domain = useMemo(() => {
    try {
      return new URL(url).hostname
    } catch {
      return url
    }
  }, [url])

  const resultData = useMemo(() => {
    if (!toolResult?.content) return null
    if (typeof toolResult.content === 'object' && toolResult.content !== null) {
      return toolResult.content as { title?: string; content?: string; length?: number; truncated?: boolean }
    }
    return null
  }, [toolResult?.content])

  return (
    <div className={styles.webSearchCard}>
      <div className={styles.webSearchHeader} onClick={onToggle}>
        <div className={styles.webSearchIcon}>
          {toolCall.status === 'running' ? (
            <svg className={styles.searchSpinner} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          )}
        </div>
        <div className={styles.webSearchInfo}>
          <span className={styles.webSearchTitle}>
            {toolCall.status === 'running' ? 'Opening page...' : 'Web Page'}
          </span>
          <span className={styles.webSearchQuery}>{domain}</span>
        </div>
        <div className={styles.webSearchStatus}>
          {toolCall.status === 'completed' && resultData && (
            <span className={styles.webSearchCount}>
              {resultData.length ? `${Math.round(resultData.length / 1000)}k chars` : ''}
              {resultData.truncated ? ' (truncated)' : ''}
            </span>
          )}
          {toolCall.status === 'failed' && (
            <span className={styles.webSearchCount} style={{ color: 'var(--color-error)' }}>Failed</span>
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
      </div>

      {isExpanded && toolCall.status === 'completed' && resultData && (
        <div className={styles.webSearchResults}>
          <div className={styles.sourceDetails}>
            <div className={styles.sourceItem}>
              <div className={styles.sourceItemHeader}>
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={styles.sourceItemTitle}
                >
                  {resultData.title || 'Untitled'}
                </a>
              </div>
              <div className={styles.openWebContent}>
                {resultData.content?.substring(0, 500)}
                {(resultData.content?.length || 0) > 500 && '...'}
              </div>
            </div>
          </div>
        </div>
      )}

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

      {toolCall.status === 'running' && (
        <div className={styles.webSearchLoading}>
          <div className={styles.webSearchLoadingBar} />
        </div>
      )}
    </div>
  )
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
  const displayToolName = clawToolName || toolName
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
          <span className={styles.webSearchTitle}>{displayToolName}</span>
          <span className={styles.webSearchQuery}>{toolCall.status}</span>
        </div>
      </div>
      {isClawCreateToolCall && (
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
      )}
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
