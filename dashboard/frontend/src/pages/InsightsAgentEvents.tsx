import {
  ArrowLeftRight,
  ArrowUpRight,
  ChevronsDownUp,
  CircleX,
  GitBranch,
  Image,
  OctagonX,
  Repeat,
  RotateCcw,
  ShieldAlert,
  Tag,
  TriangleAlert,
  type LucideIcon,
} from 'lucide-react'

import { Badge, Tooltip, cn } from '../ui'

import { AGENT_EVENT_INFO, eventCountText, type AgentEvent } from './insightsAgentEvents'

const EVENT_ICONS: Record<AgentEvent, LucideIcon> = {
  subagent_start: GitBranch,
  compaction: ChevronsDownUp,
  housekeeping: Tag,
  tool_loop: Repeat,
  image: Image,
  block: OctagonX,
  guard: ShieldAlert,
  failure: CircleX,
  tool_failure: TriangleAlert,
  switch: ArrowLeftRight,
  escalation: ArrowUpRight,
  fallback: RotateCcw,
}

/** An event as a small labelled chip, with a count when it stands for several calls. */
export function AgentEventChip({
  event,
  count,
  compact = false,
}: {
  event: AgentEvent
  count?: number
  compact?: boolean
}) {
  const info = AGENT_EVENT_INFO[event]
  const Icon = EVENT_ICONS[event]
  const text = count === undefined ? info.label : eventCountText(event, count)
  return (
    <Tooltip content={info.description}>
      <Badge
        tone={info.tone}
        data-event={event}
        className={cn('gap-1 whitespace-nowrap', compact && 'px-1')}
        aria-label={text}
      >
        <Icon className="size-3 shrink-0" aria-hidden />
        {compact ? (count !== undefined && count > 1 ? count.toLocaleString() : null) : text}
      </Badge>
    </Tooltip>
  )
}

/** Several events in the page's order; compact chips show the icon and a count. */
export function AgentEventChips({
  events,
  compact = false,
  className,
}: {
  events: (AgentEvent | [AgentEvent, number])[]
  compact?: boolean
  className?: string
}) {
  if (events.length === 0) return null
  return (
    <span className={cn('inline-flex flex-wrap items-center gap-1', className)}>
      {events.map((entry) => {
        const [event, count] = Array.isArray(entry) ? entry : [entry, undefined]
        return <AgentEventChip key={event} event={event} count={count} compact={compact} />
      })}
    </span>
  )
}
