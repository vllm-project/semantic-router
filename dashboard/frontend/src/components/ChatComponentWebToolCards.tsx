import { useId, useMemo, type ReactNode } from 'react'
import { ThinkingOrb } from 'thinking-orbs'

import type { ToolCall, ToolResult } from '../tools'

import styles from './ChatComponent.module.css'
import type { SearchResult } from './ChatComponentTypes'

interface WebToolCardHeaderProps {
  icon: ReactNode
  title: string
  subtitle: ReactNode
  status?: ReactNode
  isExpanded: boolean
  controlsId: string
  onToggle: () => void
}

/**
 * Shared disclosure header for the web tool cards. Both cards render the same
 * icon / label / status skeleton, so the button and its expanded state live here.
 */
function WebToolCardHeader({
  icon,
  title,
  subtitle,
  status,
  isExpanded,
  controlsId,
  onToggle,
}: WebToolCardHeaderProps) {
  return (
    <button
      type="button"
      className={styles.webSearchHeader}
      onClick={onToggle}
      aria-expanded={isExpanded}
      aria-controls={controlsId}
    >
      <div className={styles.webSearchIcon}>{icon}</div>
      <div className={styles.webSearchInfo}>
        <span className={styles.webSearchTitle}>{title}</span>
        <span className={styles.webSearchQuery}>{subtitle}</span>
      </div>
      <div className={styles.webSearchStatus}>
        {status}
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
  )
}

export function WebSearchCard({
  toolCall,
  toolResult,
  isExpanded,
  onToggle,
}: {
  toolCall: ToolCall
  toolResult?: ToolResult
  isExpanded: boolean
  onToggle: () => void
}) {
  let query = ''
  try {
    const args = JSON.parse(toolCall.function.arguments || '{}')
    query = args.query || ''
  } catch {
    const match = toolCall.function.arguments?.match(/"query"\s*:\s*"([^"]*)/)
    query = (match && match[1]) || 'Searching...'
  }

  const resultsId = useId()

  const results = useMemo(() => {
    if (!toolResult?.content) return undefined
    if (Array.isArray(toolResult.content)) {
      return toolResult.content as SearchResult[]
    }
    return undefined
  }, [toolResult?.content])

  return (
    <div className={styles.webSearchCard}>
      <WebToolCardHeader
        isExpanded={isExpanded}
        controlsId={resultsId}
        onToggle={onToggle}
        icon={
          toolCall.status === 'running' ? (
            <ThinkingOrb state="working" size={20} theme="dark" />
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <path d="M21 21l-4.35-4.35" />
            </svg>
          )
        }
        title={toolCall.status === 'running' ? 'Searching...' : 'Web Search'}
        subtitle={`"${query}"`}
        status={
          <>
            {toolCall.status === 'completed' && results && (
              <span className={styles.webSearchCount}>{results.length} sources</span>
            )}
            {toolCall.status === 'skipped' ? (
              <span className={styles.webSearchCount}>Not executed</span>
            ) : null}
          </>
        }
      />

      {isExpanded && toolCall.status === 'completed' && results && results.length > 0 && (
        <div id={resultsId} className={styles.webSearchResults}>
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
                <span className={styles.sourcePillDomain}>
                  {(() => {
                    try {
                      return new URL(result.url).hostname
                    } catch {
                      return result.url
                    }
                  })()}
                </span>
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
    </div>
  )
}

export function OpenWebCard({
  toolCall,
  toolResult,
  isExpanded,
  onToggle,
}: {
  toolCall: ToolCall
  toolResult?: ToolResult
  isExpanded: boolean
  onToggle: () => void
}) {
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

  const detailsId = useId()

  const resultData = useMemo(() => {
    if (!toolResult?.content) return null
    if (typeof toolResult.content === 'object' && toolResult.content !== null) {
      return toolResult.content as {
        title?: string
        content?: string
        length?: number
        truncated?: boolean
      }
    }
    return null
  }, [toolResult?.content])

  return (
    <div className={styles.webSearchCard}>
      <WebToolCardHeader
        isExpanded={isExpanded}
        controlsId={detailsId}
        onToggle={onToggle}
        icon={
          toolCall.status === 'running' ? (
            <ThinkingOrb state="working" size={20} theme="dark" />
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          )
        }
        title={toolCall.status === 'running' ? 'Opening page...' : 'Web Page'}
        subtitle={domain}
        status={
          <>
            {toolCall.status === 'completed' && resultData && (
              <span className={styles.webSearchCount}>
                {resultData.length ? `${Math.round(resultData.length / 1000)}k chars` : ''}
                {resultData.truncated ? ' (truncated)' : ''}
              </span>
            )}
            {toolCall.status === 'failed' && (
              <span className={styles.webSearchCount} style={{ color: 'var(--color-error)' }}>
                Failed
              </span>
            )}
            {toolCall.status === 'skipped' ? (
              <span className={styles.webSearchCount}>Not executed</span>
            ) : null}
          </>
        }
      />

      {isExpanded && toolCall.status === 'completed' && resultData && (
        <div id={detailsId} className={styles.webSearchResults}>
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
    </div>
  )
}
