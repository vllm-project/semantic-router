import { useState, useEffect } from 'react'
import { ThinkingOrb } from 'thinking-orbs'
import { ChevronDown } from 'lucide-react'

import { cn } from '../ui'
import MarkdownRenderer from './MarkdownRenderer'
import { getTranslateAttr } from '../hooks/useNoTranslate'

interface ThinkingBlockProps {
  content: string
  isStreaming?: boolean
  thinkingTime?: number // in seconds
}

const AUTO_COLLAPSE_THRESHOLD = 2000 // Auto collapse when content exceeds 2000 characters

const ThinkingBlock = ({ content, isStreaming = false, thinkingTime }: ThinkingBlockProps) => {
  const [displayTime, setDisplayTime] = useState(0)
  const [isExpanded, setIsExpanded] = useState(true)
  const [hasAutoCollapsed, setHasAutoCollapsed] = useState(false)

  // Auto-collapse when content exceeds threshold (even during streaming)
  useEffect(() => {
    const shouldAutoCollapse = content.length > AUTO_COLLAPSE_THRESHOLD

    if (shouldAutoCollapse && !hasAutoCollapsed) {
      setIsExpanded(false)
      setHasAutoCollapsed(true)
    }
  }, [content.length, hasAutoCollapsed, isExpanded, isStreaming])

  // Simulate thinking time counter when streaming
  useEffect(() => {
    if (isStreaming) {
      const interval = setInterval(() => {
        setDisplayTime((prev) => prev + 0.1)
      }, 100)
      return () => clearInterval(interval)
    } else if (thinkingTime !== undefined) {
      setDisplayTime(thinkingTime)
    }
  }, [isStreaming, thinkingTime])

  if (!content || content.trim().length === 0) {
    return null
  }

  const formatTime = (seconds: number) => {
    if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
    return `${seconds.toFixed(1)}s`
  }

  return (
    <div
      className="w-full overflow-hidden rounded-lg border border-border bg-surface-subtle/60"
      translate={getTranslateAttr(isStreaming)}
      data-testid="thinking-block"
    >
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left outline-none transition-colors hover:bg-accent/50 focus-visible:ring-[3px] focus-visible:ring-inset focus-visible:ring-ring/40"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <ThinkingOrb
          className="shrink-0"
          state="working"
          paused={!isStreaming}
          aria-label={isStreaming ? 'Thinking' : 'Reasoning finished'}
          size={20}
        />
        <span className="text-[13px] font-medium">{isStreaming ? 'Thinking' : 'Reasoning'}</span>
        {displayTime > 0 ? (
          <span className="font-mono text-xs text-subtle-foreground tabular">
            {formatTime(displayTime)}
          </span>
        ) : null}
        <ChevronDown
          className={cn(
            'ml-auto size-4 text-subtle-foreground transition-transform duration-200',
            isExpanded && 'rotate-180',
          )}
          aria-hidden
        />
      </button>
      {isExpanded && (
        <div
          className="max-h-80 overflow-y-auto border-t border-border px-3.5 py-2.5 text-[13px] leading-relaxed text-muted-foreground motion-safe:animate-fade-up"
          data-testid="thinking-content"
        >
          <MarkdownRenderer content={content} />
        </div>
      )}
    </div>
  )
}

export default ThinkingBlock
