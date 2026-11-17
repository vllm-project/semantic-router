import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronLeft, ChevronRight, ListTree, Lock, RefreshCw, ServerOff } from 'lucide-react'

import { Alert, Badge, Button, EmptyState, StatusDot, Tooltip, cn, type Tone } from '../ui'
import { formatDateTime } from '../utils/dateTime'

import { timeAgo } from './insightsDisplayFormat'
import type { InsightsUnavailableReason } from './insightsPageApi'
import { getInsightsLifecyclePresentation } from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'

export function RelativeTime({
  iso,
  now,
  className,
}: {
  iso: string
  now?: number
  className?: string
}) {
  return (
    <Tooltip content={formatDateTime(iso)}>
      <time dateTime={iso} className={cn('tabular whitespace-nowrap', className)}>
        {timeAgo(iso, now)}
      </time>
    </Tooltip>
  )
}

/** A request's outcome: its HTTP status when it has one, its lifecycle otherwise. */
export function LifecycleStatus({
  record,
  size = 'sm',
}: {
  record: Pick<InsightsRecord, 'lifecycle_state' | 'response_status'>
  size?: 'sm' | 'default'
}) {
  const lifecycle = getInsightsLifecyclePresentation(record as InsightsRecord)
  const title = lifecycle.label
  if (lifecycle.pending) {
    return (
      <span title={title} className="inline-flex items-center gap-1.5 text-xs text-foreground/85">
        <StatusDot tone="info" pulse />
        Running
      </span>
    )
  }
  const tone = lifecycle.successful ? 'success' : lifecycle.errored ? 'danger' : 'neutral'
  const text = record.response_status
    ? String(record.response_status)
    : lifecycle.state.replace('_', ' ')
  return (
    <Badge tone={tone} size={size} title={title} className="font-mono tabular">
      {text}
    </Badge>
  )
}

/** Live polling with a pulsing dot; `held` shows it paused while the pointer rests on the data. */
export function LiveToggle({
  live,
  onLiveChange,
  held = false,
}: {
  live: boolean
  onLiveChange: (live: boolean) => void
  held?: boolean
}) {
  return (
    <Tooltip
      content={
        live
          ? held
            ? 'Paused while your pointer is over the list'
            : 'Refreshing every 5 seconds · click to stop'
          : 'Refresh every 5 seconds'
      }
    >
      <button
        type="button"
        aria-pressed={live}
        onClick={() => onLiveChange(!live)}
        className={cn(
          'inline-flex h-8 items-center gap-2 rounded-md border px-2.5 text-[13px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40',
          live
            ? 'border-border bg-surface text-foreground shadow-xs hover:border-border-strong'
            : 'border-dashed border-border-strong text-muted-foreground hover:bg-accent hover:text-foreground',
        )}
      >
        <StatusDot tone={!live ? 'neutral' : held ? 'warning' : 'success'} pulse={live && !held} />
        {live && held ? 'Paused' : 'Live'}
      </button>
    </Tooltip>
  )
}

export function RefreshButton({ onClick, loading }: { onClick: () => void; loading?: boolean }) {
  return (
    <Tooltip content="Refresh">
      <Button variant="outline" size="icon-sm" aria-label="Refresh" onClick={onClick}>
        <RefreshCw className={cn(loading && 'animate-spin')} />
      </Button>
    </Tooltip>
  )
}

/** A thin bar for a 0–1 value, with an optional threshold tick. */
export function ScoreBar({
  value,
  threshold,
  color = 'var(--chart-2)',
  className,
  label,
}: {
  value: number
  threshold?: number
  color?: string
  className?: string
  label?: string
}) {
  return (
    <span
      role="img"
      aria-label={label ?? `${Math.round(value * 100)}%`}
      className={cn('relative block h-1.5 w-full rounded-full bg-muted', className)}
    >
      <span
        className="absolute inset-y-0 left-0 rounded-full"
        style={{ width: `${Math.min(100, Math.max(2, value * 100))}%`, background: color }}
      />
      {threshold !== undefined ? (
        <span
          className="absolute -top-[3px] h-3 w-px bg-foreground/35"
          style={{ left: `${threshold * 100}%` }}
        />
      ) : null}
    </span>
  )
}

/** Previous and next page under a server-paged table. */
export function Pager({
  page,
  pageSize,
  total,
  noun,
  onPage,
  note,
}: {
  page: number
  pageSize: number
  total: number
  noun: string
  onPage: (page: number) => void
  note?: ReactNode
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize))
  const first = total === 0 ? 0 : (page - 1) * pageSize + 1
  const last = Math.min(total, page * pageSize)
  return (
    <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-subtle-foreground">
      <span className="tabular">
        {total === 0
          ? `No ${noun}s`
          : `${first.toLocaleString()}–${last.toLocaleString()} of ${total.toLocaleString()} ${noun}${total === 1 ? '' : 's'}`}
        {note ? <span className="ml-2">{note}</span> : null}
      </span>
      {pages > 1 ? (
        <span className="flex items-center gap-1.5">
          <span className="tabular">
            Page {page} of {pages}
          </span>
          <Button
            variant="outline"
            size="icon-xs"
            aria-label="Previous page"
            disabled={page <= 1}
            onClick={() => onPage(page - 1)}
          >
            <ChevronLeft />
          </Button>
          <Button
            variant="outline"
            size="icon-xs"
            aria-label="Next page"
            disabled={page >= pages}
            onClick={() => onPage(page + 1)}
          >
            <ChevronRight />
          </Button>
        </span>
      ) : null}
    </div>
  )
}

const UNAVAILABLE_COPY: Record<
  InsightsUnavailableReason,
  { title: string; description: string; tone: Tone }
> = {
  replay_off: {
    title: 'Router Replay is not recording',
    description:
      'Requests appear here once a decision runs the router_replay plugin. Turn Replay on for a recipe, then send a request through the gateway.',
    tone: 'blue',
  },
  router_unreachable: {
    title: 'The router is not reachable',
    description:
      'The dashboard could not reach the router API, so recorded requests cannot load. Check the router on the Status page, then try again.',
    tone: 'slate',
  },
  forbidden: {
    title: 'Your role cannot read Router Replay',
    description:
      'Viewing recorded requests needs the replay.read permission. Ask an admin for access.',
    tone: 'slate',
  },
}

/** Why a replay view is empty, with the next step to take. */
export function InsightsUnavailable({
  reason,
  noun = 'requests',
  onRetry,
}: {
  reason: InsightsUnavailableReason
  noun?: string
  onRetry?: () => void
}) {
  const navigate = useNavigate()
  const copy = UNAVAILABLE_COPY[reason]
  const Icon =
    reason === 'router_unreachable' ? ServerOff : reason === 'forbidden' ? Lock : ListTree
  return (
    <EmptyState
      icon={Icon}
      tone={copy.tone}
      title={copy.title}
      description={copy.description.replace('recorded requests', `recorded ${noun}`)}
      actions={
        <>
          {reason === 'replay_off' ? (
            <Button size="sm" onClick={() => navigate('/routing/recipes')}>
              Open recipes
            </Button>
          ) : null}
          {reason === 'router_unreachable' ? (
            <Button size="sm" onClick={() => navigate('/observe/status')}>
              Check status
            </Button>
          ) : null}
          {onRetry ? (
            <Button size="sm" variant="outline" onClick={onRetry}>
              <RefreshCw />
              Try again
            </Button>
          ) : null}
        </>
      }
      footer={
        reason === 'replay_off' ? (
          <a
            className="underline-offset-4 hover:text-foreground hover:underline"
            href="https://vllm-sr.ai/docs/api/router#router-replay"
            target="_blank"
            rel="noopener noreferrer"
          >
            Read the replay guide
          </a>
        ) : undefined
      }
    />
  )
}

/** A failed load that is not one of the known unavailable states. */
export function InsightsLoadError({
  title = 'Could not load',
  message,
  onRetry,
}: {
  title?: string
  message: string
  onRetry: () => void
}) {
  return (
    <Alert
      tone="danger"
      title={title}
      className="mb-3"
      actions={
        <Button size="xs" variant="outline" onClick={onRetry}>
          <RefreshCw />
          Retry
        </Button>
      }
    >
      {message}
    </Alert>
  )
}
