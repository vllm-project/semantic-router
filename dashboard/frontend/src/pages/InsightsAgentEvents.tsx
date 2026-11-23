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
import { forwardRef, type ComponentPropsWithoutRef } from 'react'

import { Badge, Tooltip, cn } from '../ui'

import {
  AGENT_EVENT_INFO,
  eventCountText,
  type AgentEvent,
  type AgentEventTone,
} from './insightsAgentEvents'

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

/** An event's icon alone, as a filter option shows it. */
export function AgentEventIcon({ event, className }: { event: AgentEvent; className?: string }) {
  const Icon = EVENT_ICONS[event]
  return <Icon className={className} aria-hidden />
}

/** The solid fill of an event's marker on the Timeline, by its tone. */
const MARKER_FILL: Record<AgentEventTone, string> = {
  info: 'bg-info',
  violet: 'bg-chart-3',
  teal: 'bg-chart-4',
  warning: 'bg-warning',
  danger: 'bg-destructive',
  neutral: 'bg-chart-8',
}

/**
 * An event as a labelled badge, with a count when it stands for several calls.
 * A routine event (a tool loop, housekeeping) reads as a quiet outline mark; a
 * compact badge shows the icon and a count only.
 */
export function AgentEventChip({
  event,
  count,
  compact = false,
  strong = false,
}: {
  event: AgentEvent
  count?: number
  compact?: boolean
  /** The event a list is filtered by: drawn as the solid marker is. */
  strong?: boolean
}) {
  const info = AGENT_EVENT_INFO[event]
  const Icon = EVENT_ICONS[event]
  const text = count === undefined ? info.label : eventCountText(event, count)
  const bare = compact || info.quiet
  return (
    <Tooltip content={info.description}>
      <Badge
        tone={info.quiet ? 'outline' : info.tone}
        size="sm"
        data-event={event}
        className={cn(
          'gap-1 whitespace-nowrap',
          bare && 'px-1',
          strong && cn(MARKER_FILL[info.tone], 'border-transparent text-white'),
        )}
        aria-label={text}
      >
        <Icon className="size-3 shrink-0" aria-hidden />
        {bare ? (count !== undefined && count > 1 ? count.toLocaleString() : null) : text}
      </Badge>
    </Tooltip>
  )
}

/**
 * Several events in the page's order. The events a list is filtered by come
 * first and solid; compact chips show the icon and a count.
 */
export function AgentEventChips({
  events,
  compact = false,
  emphasize = [],
  className,
}: {
  events: (AgentEvent | [AgentEvent, number])[]
  compact?: boolean
  emphasize?: readonly string[]
  className?: string
}) {
  if (events.length === 0) return null
  const entries = events.map((entry) =>
    Array.isArray(entry) ? entry : ([entry, undefined] as const),
  )
  const ordered = [
    ...entries.filter(([event]) => emphasize.includes(event)),
    ...entries.filter(([event]) => !emphasize.includes(event)),
  ]
  return (
    <span className={cn('inline-flex flex-wrap items-center gap-1', className)}>
      {ordered.map(([event, count]) => (
        <AgentEventChip
          key={event}
          event={event}
          count={count}
          compact={compact}
          strong={emphasize.includes(event)}
        />
      ))}
    </span>
  )
}

/**
 * An event as the Timeline draws it: a solid round mark with its icon, its
 * label beside it when there is room, and a count when it stands for several.
 */
export const AgentEventMarker = forwardRef<
  HTMLSpanElement,
  ComponentPropsWithoutRef<'span'> & {
    event: AgentEvent
    count?: number
    labelled?: boolean
    size?: 'sm' | 'default'
  }
>(function AgentEventMarker(
  { event, count, labelled = false, size = 'default', className, ...props },
  ref,
) {
  const info = AGENT_EVENT_INFO[event]
  const Icon = EVENT_ICONS[event]
  return (
    <span
      ref={ref}
      data-event={event}
      className={cn(
        'relative inline-flex shrink-0 items-center justify-center gap-1 rounded-full font-semibold text-white',
        MARKER_FILL[info.tone],
        size === 'sm' ? 'h-[18px] min-w-[18px] text-[10.5px]' : 'h-5 min-w-5 text-[11px]',
        labelled ? 'pl-1.5 pr-2' : 'px-[3px]',
        className,
      )}
      {...props}
    >
      <Icon className={size === 'sm' ? 'size-[11px]' : 'size-3'} aria-hidden />
      {labelled ? <span className="whitespace-nowrap leading-none">{info.label}</span> : null}
      {count !== undefined && count > 1 ? (
        <span className="absolute -right-2 -top-2 min-w-4 rounded-full bg-foreground px-1 text-center font-mono text-[9.5px] leading-4 text-background ring-2 ring-card">
          {count > 99 ? '99+' : count}
        </span>
      ) : null}
    </span>
  )
})
