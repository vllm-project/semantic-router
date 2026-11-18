import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronLeft, ChevronRight, ListTree, Lock, RefreshCw, ServerOff } from 'lucide-react'

import {
  Alert,
  Badge,
  Button,
  Card,
  EmptyState,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  StatusDot,
  Tooltip,
  cn,
  type Tone,
} from '../ui'
import { formatDateTime } from '../utils/dateTime'

import { requestStatusText, timeAgo } from './insightsDisplayFormat'
import { PAGE_SIZES } from './insightsListData'
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
  const status = requestStatusText(record.response_status, lifecycle)
  return (
    <Badge
      tone={tone}
      size={size}
      title={status.hiddenStatus ? `${title} · HTTP ${status.hiddenStatus}` : title}
      className={status.code ? 'font-mono tabular' : undefined}
    >
      {status.text}
    </Badge>
  )
}

/** Live polling with a pulsing dot; `held` shows it paused while the pointer rests on the data. */
export function LiveToggle({
  live,
  onLiveChange,
  held = false,
  every = 'every 5 seconds',
}: {
  live: boolean
  onLiveChange: (live: boolean) => void
  held?: boolean
  /** How often a live view refreshes, as the tooltip says it. */
  every?: string
}) {
  return (
    <Tooltip
      content={
        live
          ? held
            ? 'Paused while your pointer is over the list'
            : `Refreshing ${every} · click to stop`
          : `Refresh ${every}`
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

/**
 * The rows a paged table shows out of its total, the rows per page, and the
 * previous and next page.
 */
export function Pager({
  page,
  pageSize,
  total,
  noun,
  onPage,
  onPageSize,
  note,
  className,
}: {
  page: number
  pageSize: number
  total: number
  noun: string
  onPage: (page: number) => void
  /** Offers the page sizes when given and the total fills more than the smallest page. */
  onPageSize?: (size: number) => void
  note?: ReactNode
  className?: string
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize))
  const first = total === 0 ? 0 : Math.min(total, (page - 1) * pageSize + 1)
  const last = Math.min(total, page * pageSize)
  return (
    <div
      className={cn(
        'mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 text-xs text-subtle-foreground',
        className,
      )}
    >
      <span className="min-w-0 tabular">
        {total === 0
          ? `No ${noun}s`
          : `${first.toLocaleString()}–${last.toLocaleString()} of ${total.toLocaleString()} ${noun}${total === 1 ? '' : 's'}`}
        {note ? <span className="ml-2">{note}</span> : null}
      </span>
      <span className="flex items-center gap-4">
        {onPageSize && total > PAGE_SIZES[0] ? (
          <span className="flex items-center gap-1.5">
            Rows
            <Select value={String(pageSize)} onValueChange={(value) => onPageSize(Number(value))}>
              <SelectTrigger
                size="sm"
                aria-label="Rows per page"
                className="h-7 gap-1 px-2 text-xs"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="end">
                {PAGE_SIZES.map((size) => (
                  <SelectItem key={size} value={String(size)}>
                    {size}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </span>
        ) : null}
        {pages > 1 ? (
          <span className="flex items-center gap-1.5">
            <span className="tabular">
              Page {page.toLocaleString()} of {pages.toLocaleString()}
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
      </span>
    </div>
  )
}

/** One figure of a summary strip: its label and icon, the value, and a line under it. */
export function SummaryTile({
  label,
  icon,
  value,
  hint,
  tone,
}: {
  label: string
  icon?: ReactNode
  value: ReactNode
  hint?: ReactNode
  tone?: 'negative'
}) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5 px-4 py-3">
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground [&_svg]:size-3.5">
        {icon}
        {label}
      </span>
      <span
        className={cn(
          'truncate text-lg font-semibold leading-none tracking-[-0.02em] tabular',
          tone === 'negative' && 'text-destructive-foreground',
        )}
      >
        {value}
      </span>
      {hint ? <span className="truncate text-[11.5px] text-subtle-foreground">{hint}</span> : null}
    </div>
  )
}

/** Summary tiles in one card, divided by hairlines; className sets the grid columns. */
export function SummaryStrip({
  label,
  className,
  children,
}: {
  label: string
  className?: string
  children: ReactNode
}) {
  return (
    <Card
      role="group"
      aria-label={label}
      className={cn('grid gap-px overflow-hidden bg-border [&>*]:bg-card', className)}
    >
      {children}
    </Card>
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
