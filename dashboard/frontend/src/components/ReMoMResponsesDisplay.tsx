import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { ChevronDown, Layers } from 'lucide-react'

import { Badge, cn } from '../ui'

interface ReMoMIntermediateResp {
  model: string
  content: string
  reasoning?: string
  compacted_content?: string
  token_count?: number
}

interface ReMoMRoundResponse {
  round: number
  breadth: number
  responses: ReMoMIntermediateResp[]
}

interface ReMoMResponsesDisplayProps {
  rounds: ReMoMRoundResponse[]
}

function Section({ label, content }: { label: string; content: string }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[11px] font-medium text-subtle-foreground">{label}</div>
      <div className="max-h-72 overflow-y-auto text-[13px] leading-relaxed text-foreground/90 [&_code]:font-mono [&_code]:text-[12px] [&_p]:my-1.5 [&_pre]:my-2 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:border [&_pre]:border-border [&_pre]:bg-surface-sunken [&_pre]:p-2.5">
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
          {content}
        </ReactMarkdown>
      </div>
    </div>
  )
}

const ReMoMResponsesDisplay = ({ rounds }: ReMoMResponsesDisplayProps) => {
  // Track expanded state for each response
  const [expandedResponses, setExpandedResponses] = useState<Set<string>>(new Set())

  const toggleResponse = (roundIndex: number, responseIndex: number) => {
    const key = `${roundIndex}-${responseIndex}`
    setExpandedResponses((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(key)) {
        newSet.delete(key)
      } else {
        newSet.add(key)
      }
      return newSet
    })
  }

  const isExpanded = (roundIndex: number, responseIndex: number) => {
    return expandedResponses.has(`${roundIndex}-${responseIndex}`)
  }

  if (!rounds || rounds.length === 0) {
    return null
  }

  return (
    <div className="w-full overflow-hidden rounded-xl border border-border bg-card shadow-xs">
      <div className="flex items-center gap-2 border-b border-border bg-surface-subtle/70 px-3.5 py-2.5 text-[13px] font-medium">
        <Layers className="size-4 text-chart-3" />
        <span>ReMoM test-time scaling</span>
        <Badge size="sm" tone="violet" className="ml-auto">
          {rounds.length} round{rounds.length > 1 ? 's' : ''}
        </Badge>
      </div>

      <div className="flex flex-col divide-y divide-border">
        {rounds.map((round, roundIndex) => (
          <div key={roundIndex} className="flex flex-col gap-2 px-3.5 py-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium">Round {round.round}</span>
              <span className="text-subtle-foreground">
                {round.breadth} parallel explore{round.breadth > 1 ? 's' : ''}
              </span>
            </div>

            <div className="flex flex-col gap-1.5">
              {round.responses.map((response, responseIndex) => {
                const expanded = isExpanded(roundIndex, responseIndex)
                return (
                  <div
                    key={responseIndex}
                    className="overflow-hidden rounded-lg border border-border bg-surface-subtle/50"
                  >
                    <button
                      type="button"
                      className="flex w-full items-center gap-2 px-3 py-2 text-left outline-none transition-colors hover:bg-accent/60 focus-visible:ring-[3px] focus-visible:ring-inset focus-visible:ring-ring/40"
                      aria-expanded={expanded}
                      onClick={() => toggleResponse(roundIndex, responseIndex)}
                    >
                      <span className="min-w-0 flex-1 truncate font-mono text-xs font-medium">
                        {response.model}
                      </span>
                      {response.token_count ? (
                        <span className="shrink-0 text-[11px] text-subtle-foreground tabular">
                          {response.token_count} tokens
                        </span>
                      ) : null}
                      <ChevronDown
                        className={cn(
                          'size-4 shrink-0 text-subtle-foreground transition-transform duration-200',
                          expanded && 'rotate-180',
                        )}
                        aria-hidden
                      />
                    </button>

                    {expanded && (
                      <div className="flex flex-col gap-3 border-t border-border bg-card px-3 py-2.5">
                        {response.compacted_content ? (
                          <Section label="Compacted content" content={response.compacted_content} />
                        ) : null}
                        {response.reasoning ? (
                          <Section label="Reasoning" content={response.reasoning} />
                        ) : null}
                        {response.content && !response.compacted_content ? (
                          <Section label="Full content" content={response.content} />
                        ) : null}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ReMoMResponsesDisplay
