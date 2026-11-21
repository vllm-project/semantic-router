import { useId, useMemo, type ReactNode } from 'react'
import { ThinkingOrb } from 'thinking-orbs'
import { ChevronDown, Globe, Search } from 'lucide-react'

import { cn } from '../ui'
import type { ToolCall, ToolResult } from '../tools'

import type { SearchResult } from './ChatComponentTypes'

const styles = {
  card: 'w-full overflow-hidden rounded-xl border border-border bg-card shadow-xs',
  webSearchHeader:
    'flex w-full items-center gap-3 px-3 py-2.5 text-left outline-none transition-colors hover:bg-accent/50 focus-visible:ring-[3px] focus-visible:ring-inset focus-visible:ring-ring/40',
  icon: 'grid size-8 shrink-0 place-items-center rounded-lg border border-border bg-surface-subtle text-muted-foreground [&_svg]:size-4',
  info: 'flex min-w-0 flex-1 flex-col',
  title: 'text-[13px] font-medium text-foreground',
  query: 'truncate text-xs text-subtle-foreground',
  status: 'flex shrink-0 items-center gap-2 text-xs text-subtle-foreground',
  results: 'flex flex-col gap-3 border-t border-border bg-surface-subtle/50 px-3 py-3',
  sourceItem: 'flex min-w-0 flex-col gap-1',
  sourceTitle:
    'truncate text-[13px] font-medium text-primary underline-offset-4 outline-none hover:underline focus-visible:underline',
  snippet: 'text-xs leading-relaxed text-muted-foreground',
  pre: 'max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface px-2.5 py-2 font-mono text-[11.5px] leading-relaxed text-foreground/85',
} as const

function hostOf(url: string): string {
  try {
    return new URL(url).hostname
  } catch {
    return url
  }
}

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
      <span className={styles.icon}>{icon}</span>
      <span className={styles.info}>
        <span className={styles.title}>{title}</span>
        <span className={styles.query}>{subtitle}</span>
      </span>
      <span className={styles.status}>
        {status}
        <ChevronDown
          className={cn('size-4 transition-transform duration-200', isExpanded && 'rotate-180')}
          aria-hidden
        />
      </span>
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
    <div className={styles.card}>
      <WebToolCardHeader
        isExpanded={isExpanded}
        controlsId={resultsId}
        onToggle={onToggle}
        icon={
          toolCall.status === 'running' ? <ThinkingOrb state="searching" size={20} /> : <Search />
        }
        title={toolCall.status === 'running' ? 'Searching...' : 'Web Search'}
        subtitle={`"${query}"`}
        status={
          <>
            {toolCall.status === 'completed' && results ? (
              <span className="tabular">{results.length} sources</span>
            ) : null}
            {toolCall.status === 'skipped' ? <span>Not executed</span> : null}
          </>
        }
      />

      {isExpanded && toolCall.status === 'completed' && results && results.length > 0 && (
        <div id={resultsId} className={styles.results}>
          <div className="flex flex-wrap gap-1.5">
            {results.map((result, idx) => (
              <a
                key={idx}
                href={result.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex h-6 max-w-48 items-center gap-1.5 rounded-full border border-border bg-surface pl-1 pr-2 text-[11.5px] text-muted-foreground outline-none transition-colors hover:border-border-strong hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40"
                title={result.snippet}
              >
                <span className="grid size-4 place-items-center rounded-full bg-muted text-[10px] font-semibold tabular">
                  {idx + 1}
                </span>
                <span className="truncate">{hostOf(result.url)}</span>
              </a>
            ))}
          </div>
          <div className="flex flex-col gap-2.5">
            {results.map((result, idx) => (
              <div key={idx} className={styles.sourceItem}>
                <div className="flex min-w-0 items-baseline gap-1.5">
                  <span className="shrink-0 font-mono text-[11px] text-subtle-foreground">
                    [{idx + 1}]
                  </span>
                  <a
                    href={result.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={styles.sourceTitle}
                  >
                    {result.title}
                  </a>
                </div>
                <p className={styles.snippet}>{result.snippet}</p>
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

  const domain = useMemo(() => hostOf(url), [url])

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
    <div className={styles.card}>
      <WebToolCardHeader
        isExpanded={isExpanded}
        controlsId={detailsId}
        onToggle={onToggle}
        icon={toolCall.status === 'running' ? <ThinkingOrb state="working" size={20} /> : <Globe />}
        title={toolCall.status === 'running' ? 'Opening page...' : 'Web Page'}
        subtitle={domain}
        status={
          <>
            {toolCall.status === 'completed' && resultData ? (
              <span className="tabular">
                {resultData.length ? `${Math.round(resultData.length / 1000)}k chars` : ''}
                {resultData.truncated ? ' (truncated)' : ''}
              </span>
            ) : null}
            {toolCall.status === 'failed' ? (
              <span className="text-destructive-foreground">Failed</span>
            ) : null}
            {toolCall.status === 'skipped' ? <span>Not executed</span> : null}
          </>
        }
      />

      {isExpanded && toolCall.status === 'completed' && resultData && (
        <div id={detailsId} className={styles.results}>
          <div className={styles.sourceItem}>
            <a href={url} target="_blank" rel="noopener noreferrer" className={styles.sourceTitle}>
              {resultData.title || 'Untitled'}
            </a>
            <div className={styles.pre}>
              {resultData.content?.substring(0, 500)}
              {(resultData.content?.length || 0) > 500 && '...'}
            </div>
          </div>
        </div>
      )}

      {isExpanded && toolCall.status === 'failed' && toolResult?.error && (
        <div className={styles.results}>
          <p className={cn(styles.snippet, 'text-destructive-foreground')}>{toolResult.error}</p>
        </div>
      )}
    </div>
  )
}
