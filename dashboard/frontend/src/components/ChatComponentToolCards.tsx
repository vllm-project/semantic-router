import { useMemo } from 'react'
import { ThinkingOrb } from 'thinking-orbs'
import { ChevronDown, Wrench } from 'lucide-react'

import { Badge, cn } from '../ui'
import type { ToolCall, ToolResult } from '../tools'

import { OpenWebCard, WebSearchCard } from './ChatComponentWebToolCards'
import { getToolDisplayName, getToolStatusLabel, getToolSummary } from './chatToolCardPresentation'

const TOOL_STATUS_TONES = {
  pending: 'neutral',
  running: 'info',
  completed: 'success',
  failed: 'danger',
  skipped: 'outline',
} as const satisfies Record<ToolCall['status'], string>

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
    <div className="w-full overflow-hidden rounded-xl border border-border bg-card shadow-xs">
      <button
        type="button"
        className="flex w-full items-center gap-3 px-3 py-2.5 text-left outline-none transition-colors hover:bg-accent/50 focus-visible:ring-[3px] focus-visible:ring-inset focus-visible:ring-ring/40"
        onClick={onToggle}
        aria-expanded={isExpanded}
        aria-label={`${isExpanded ? 'Collapse' : 'Expand'} details for ${displayToolName}`}
      >
        <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-border bg-surface-subtle text-muted-foreground">
          <Wrench className="size-4" />
        </span>
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="text-[13px] font-medium">{displayToolName}</span>
          <span className="truncate font-mono text-xs text-subtle-foreground">{summary}</span>
        </span>
        <span className="flex shrink-0 items-center gap-2">
          {toolCall.status === 'running' || toolCall.status === 'pending' ? (
            <span
              className="inline-flex items-center gap-1.5 text-xs text-muted-foreground"
              aria-label={statusLabel}
            >
              <ThinkingOrb state="working" size={20} />
              {statusLabel}
            </span>
          ) : (
            <Badge size="sm" tone={TOOL_STATUS_TONES[toolCall.status]}>
              {statusLabel}
            </Badge>
          )}
          <ChevronDown
            className={cn(
              'size-4 text-subtle-foreground transition-transform duration-200',
              isExpanded && 'rotate-180',
            )}
            aria-hidden
          />
        </span>
      </button>
      {isExpanded && toolCall.status === 'failed' && toolResult?.error && (
        <div className="border-t border-border bg-surface-subtle/50 px-3 py-3">
          <p className="text-xs leading-relaxed text-destructive-foreground">{toolResult.error}</p>
        </div>
      )}
      {isExpanded && toolCall.status === 'completed' && resultPreview && (
        <div className="border-t border-border bg-surface-subtle/50 px-3 py-3">
          <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface px-2.5 py-2 font-mono text-[11.5px] leading-relaxed text-foreground/85">
            {resultPreview}
          </pre>
        </div>
      )}
    </div>
  )
}
