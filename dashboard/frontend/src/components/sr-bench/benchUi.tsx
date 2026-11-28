import * as React from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Info,
  OctagonAlert,
  type LucideIcon,
} from 'lucide-react'
import { Skeleton, StatusDot, cn } from '../../ui'
import { benchText, statusTone } from './benchStyles'
import { active } from './model'

/** A workspace surface: one card per view or major block. */
export const BenchPanel = React.forwardRef<HTMLElement, React.HTMLAttributes<HTMLElement>>(
  ({ className, ...props }, ref) => (
    <section
      ref={ref}
      className={cn(
        'mb-5 min-w-0 rounded-xl border border-border bg-card p-4 shadow-xs sm:p-5',
        className,
      )}
      {...props}
    />
  ),
)
BenchPanel.displayName = 'BenchPanel'

/** A nested block inside a panel, e.g. a chart, an activity list or a call section. */
export const BenchBlock = React.forwardRef<HTMLElement, React.HTMLAttributes<HTMLElement>>(
  ({ className, ...props }, ref) => (
    <section
      ref={ref}
      className={cn('min-w-0 rounded-lg border border-border bg-surface p-4', className)}
      {...props}
    />
  ),
)
BenchBlock.displayName = 'BenchBlock'

export function BenchHeader({
  title,
  titleId,
  level = 2,
  description,
  actions,
  children,
  className,
}: {
  title: React.ReactNode
  titleId?: string
  level?: 2 | 3 | 4
  description?: React.ReactNode
  actions?: React.ReactNode
  children?: React.ReactNode
  className?: string
}) {
  const Heading = `h${level}` as const
  return (
    <div
      className={cn('mb-4 flex flex-wrap items-start justify-between gap-x-6 gap-y-3', className)}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <Heading
          id={titleId}
          className={level === 2 ? benchText.h2 : level === 3 ? benchText.h3 : benchText.h4}
        >
          {title}
        </Heading>
        {description ? <p className={cn(benchText.muted, 'max-w-3xl')}>{description}</p> : null}
        {children}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  )
}

export function ActionRow({
  className,
  children,
}: {
  className?: string
  children: React.ReactNode
}) {
  return <div className={cn('flex flex-wrap items-center gap-2', className)}>{children}</div>
}

export type NoticeTone = 'info' | 'success' | 'warning' | 'danger' | 'neutral'

const noticeTone: Record<NoticeTone, string> = {
  info: 'border-info/25 bg-info-soft [&>svg]:text-info',
  success: 'border-success/25 bg-success-soft [&>svg]:text-success',
  warning: 'border-warning/30 bg-warning-soft [&>svg]:text-warning',
  danger: 'border-destructive/25 bg-destructive-soft [&>svg]:text-destructive',
  neutral: 'border-border bg-surface-subtle [&>svg]:text-muted-foreground',
}

const noticeIcon: Record<NoticeTone, LucideIcon> = {
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  danger: OctagonAlert,
  neutral: Info,
}

type NoticeElement = 'div' | 'p' | 'section' | 'aside'

/**
 * Alert-styled notice that keeps the caller's element and role, so existing
 * landmarks (alert, status, note, named regions) stay intact.
 */
export function Notice({
  tone = 'neutral',
  as = 'div',
  icon,
  className,
  children,
  ...props
}: {
  tone?: NoticeTone
  as?: NoticeElement
  icon?: LucideIcon | false
  className?: string
  children: React.ReactNode
} & Omit<React.HTMLAttributes<HTMLElement>, 'children'>) {
  const Tag = as as React.ElementType
  const Icon = icon === false ? null : (icon ?? noticeIcon[tone])
  const Body = as === 'p' ? 'span' : 'div'
  return (
    <Tag
      className={cn(
        'my-3 flex min-w-0 gap-3 rounded-lg border px-3.5 py-3 text-[13px] leading-relaxed text-muted-foreground',
        '[&>svg]:mt-[3px] [&>svg]:size-4 [&>svg]:shrink-0',
        '[&_h3]:mb-1 [&_h3]:text-[13px] [&_h3]:font-semibold [&_h3]:text-foreground [&_h4]:mb-1 [&_h4]:font-medium [&_h4]:text-foreground [&_strong]:font-medium [&_strong]:text-foreground',
        noticeTone[tone],
        className,
      )}
      {...props}
    >
      {Icon ? <Icon aria-hidden="true" /> : null}
      <Body
        className={cn(
          'min-w-0 flex-1 break-words',
          Body === 'div' && 'flex flex-col gap-1.5 [&_p]:m-0',
        )}
      >
        {children}
      </Body>
    </Tag>
  )
}

/** Native disclosure: keyboard behaviour and the closed default come from the platform. */
export function Disclosure({
  summary,
  summaryId,
  summaryClassName,
  className,
  bodyClassName,
  children,
  bare = false,
  ...props
}: {
  summary: React.ReactNode
  summaryId?: string
  summaryClassName?: string
  bodyClassName?: string
  bare?: boolean
  children: React.ReactNode
} & Omit<React.DetailsHTMLAttributes<HTMLDetailsElement>, 'children'>) {
  return (
    <details
      className={cn(
        'group/disclosure my-3 min-w-0',
        !bare && 'rounded-lg border border-border bg-surface',
        className,
      )}
      {...props}
    >
      <summary
        id={summaryId}
        className={cn(
          'flex cursor-pointer list-none items-center gap-2 rounded-lg text-[13px] font-medium text-foreground/85 outline-none transition-colors hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 [&::-webkit-details-marker]:hidden',
          bare ? 'py-1.5' : 'px-3.5 py-2.5 hover:bg-surface-subtle',
          summaryClassName,
        )}
      >
        <ChevronRight
          aria-hidden="true"
          size={14}
          className="shrink-0 text-subtle-foreground transition-transform duration-200 group-open/disclosure:rotate-90"
        />
        {summary}
      </summary>
      <div
        className={cn(
          'min-w-0 text-[13px]',
          bare ? 'pb-1 pt-2' : 'border-t border-border px-3.5 py-3',
          bodyClassName,
        )}
      >
        {children}
      </div>
    </details>
  )
}

/** Scroll container for evidence tables; header and cell styling come from benchTable. */
export const TableFrame = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'my-3 min-w-0 max-w-full overflow-auto rounded-lg border border-border bg-card',
        className,
      )}
      {...props}
    />
  ),
)
TableFrame.displayName = 'TableFrame'

/** Bordered metric strip: label, value and a short qualifier per cell. */
export function MetricGrid({
  className,
  children,
}: {
  className?: string
  children: React.ReactNode
}) {
  return (
    <div
      className={cn(
        'my-4 grid min-w-0 grid-cols-2 gap-px overflow-hidden rounded-xl border border-border bg-border md:grid-cols-4',
        className,
      )}
    >
      {children}
    </div>
  )
}

export function Metric({
  label,
  value,
  hint,
  className,
  ...props
}: {
  label: React.ReactNode
  value: React.ReactNode
  hint?: React.ReactNode
  className?: string
} & Omit<React.HTMLAttributes<HTMLDivElement>, 'children'>) {
  return (
    <div className={cn('flex min-w-0 flex-col gap-1 bg-card px-4 py-3', className)} {...props}>
      <span className="text-xs text-muted-foreground">{label}</span>
      <strong className="min-w-0 break-words text-lg font-semibold leading-tight tracking-[-0.02em] text-foreground tabular-nums">
        {value}
      </strong>
      {hint ? <small className="text-[11px] text-subtle-foreground">{hint}</small> : null}
    </div>
  )
}

/** A headline figure in a sheet summary strip; extra content (e.g. a bar) goes below the hint. */
export function SummaryStat({
  label,
  value,
  hint,
  className,
  children,
  ...props
}: {
  label: React.ReactNode
  value: React.ReactNode
  hint?: React.ReactNode
  className?: string
  children?: React.ReactNode
} & Omit<React.HTMLAttributes<HTMLDivElement>, 'children'>) {
  return (
    <div
      role={props['aria-label'] ? 'group' : undefined}
      className={cn(
        'flex min-w-0 flex-col rounded-lg border border-border bg-surface-subtle/60 px-3 py-2.5',
        className,
      )}
      {...props}
    >
      <span className="text-xs text-subtle-foreground">{label}</span>
      <strong className="mt-1 truncate text-lg font-semibold leading-tight tracking-[-0.02em] text-foreground tabular-nums">
        {value}
      </strong>
      {hint ? <span className="mt-0.5 truncate text-xs text-muted-foreground">{hint}</span> : null}
      {children}
    </div>
  )
}

/** In-flow loading placeholder: a visible label over shimmering lines, never a spinner. */
export function BenchLoading({
  label,
  lines = 2,
  className,
}: {
  label: string
  lines?: number
  className?: string
}) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn('my-3 flex min-w-0 flex-col gap-2.5 py-1', className)}
    >
      <span className="text-xs text-muted-foreground">{label}</span>
      {Array.from({ length: lines }, (_, index) => (
        <Skeleton
          key={index}
          aria-hidden="true"
          className={cn('h-3', index % 3 === 0 ? 'w-3/4' : index % 3 === 1 ? 'w-1/2' : 'w-2/3')}
        />
      ))}
    </div>
  )
}

/** Run status keeps the service's own word; the dot carries the tone. */
export function RunStatus({ status }: { status: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-1.5 py-0.5 text-xs font-medium text-foreground/85">
      <StatusDot tone={statusTone(status)} pulse={active(status)} />
      {status}
    </span>
  )
}

export function EmptyNote({
  icon: Icon,
  title,
  children,
  className,
  role,
}: {
  icon?: LucideIcon
  title?: React.ReactNode
  children?: React.ReactNode
  className?: string
  role?: string
}) {
  return (
    <div
      role={role}
      className={cn(
        'my-3 flex min-w-0 flex-col items-center gap-2 rounded-lg border border-dashed border-border-strong px-6 py-8 text-center',
        className,
      )}
    >
      {Icon ? (
        <span className="grid size-9 place-items-center rounded-lg bg-surface-subtle text-subtle-foreground ring-1 ring-inset ring-border">
          <Icon className="size-[18px]" strokeWidth={1.8} aria-hidden="true" />
        </span>
      ) : null}
      {title ? <h3 className={benchText.h3}>{title}</h3> : null}
      {children ? <div className={cn(benchText.muted, 'max-w-md')}>{children}</div> : null}
    </div>
  )
}
