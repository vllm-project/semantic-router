import type { LucideIcon } from 'lucide-react'
import {
  Braces,
  BrushCleaning,
  CornerDownRight,
  Route,
  ShieldCheck,
  Sparkle,
  User,
} from 'lucide-react'

import { Badge, cn } from '../ui'

import { REQUEST_CLASS_LABELS, subagentKindLabel, type RequestClass } from './insightsAgentIdentity'
import { API_CLIENT_LABEL, harnessInfo } from './insightsHarness'

/** A harness's logo, a neutral monogram for one this registry does not know, or the API client glyph for none. */
export function HarnessLogo({
  harness,
  size = 16,
  framed = false,
  className,
}: {
  harness?: string | null
  size?: number
  framed?: boolean
  className?: string
}) {
  const info = harnessInfo(harness)
  const glyph = info?.logo ? (
    <img
      src={info.logo}
      alt=""
      width={size}
      height={size}
      draggable={false}
      data-harness={info.id}
      className={cn('shrink-0 select-none object-contain', info.monochrome && 'dark:invert')}
      style={{ width: size, height: size }}
    />
  ) : info ? (
    <span
      data-harness={info.id}
      className="grid shrink-0 place-items-center rounded-[5px] bg-muted font-semibold uppercase text-muted-foreground"
      style={{ width: size, height: size, fontSize: Math.max(9, size * 0.5) }}
    >
      {info.label.slice(0, 1)}
    </span>
  ) : (
    <Braces
      data-harness="api"
      className="shrink-0 text-subtle-foreground"
      style={{ width: size, height: size }}
      strokeWidth={1.8}
    />
  )
  const label = info?.label ?? API_CLIENT_LABEL
  if (!framed) {
    return (
      <span
        role="img"
        aria-label={label}
        title={label}
        className={cn('inline-flex shrink-0', className)}
      >
        {glyph}
      </span>
    )
  }
  return (
    <span
      role="img"
      aria-label={label}
      title={label}
      className={cn(
        'grid shrink-0 place-items-center rounded-lg border border-border bg-surface shadow-xs',
        className,
      )}
      style={{ width: size + 14, height: size + 14 }}
    >
      {glyph}
    </span>
  )
}

/** A harness's logo and name. */
export function HarnessName({
  harness,
  size = 14,
  className,
}: {
  harness?: string | null
  size?: number
  className?: string
}) {
  const label = harnessInfo(harness)?.label ?? API_CLIENT_LABEL
  return (
    <span className={cn('inline-flex min-w-0 items-center gap-1.5', className)}>
      <HarnessLogo harness={harness} size={size} />
      <span className="truncate">{label}</span>
    </span>
  )
}

const CLASS_ICONS: Record<RequestClass, LucideIcon> = {
  main: User,
  subagent: CornerDownRight,
  housekeeping: BrushCleaning,
  review: ShieldCheck,
  router: Route,
  api: Sparkle,
}

/** What a call was to its agent: a quiet badge, the subagent's kind beside it. */
export function RequestClassBadge({
  requestClass,
  kind,
  className,
}: {
  requestClass: RequestClass
  kind?: string
  className?: string
}) {
  const Icon = CLASS_ICONS[requestClass]
  const kindLabel = requestClass === 'subagent' ? subagentKindLabel(kind) : undefined
  return (
    <Badge
      tone="outline"
      size="sm"
      data-request-class={requestClass}
      className={cn('max-w-full', className)}
    >
      <Icon />
      <span className="truncate">
        {REQUEST_CLASS_LABELS[requestClass]}
        {kindLabel ? ` · ${kindLabel}` : ''}
      </span>
    </Badge>
  )
}

/** The tools a request's conversation called, as recorded. */
export function ToolChips({
  names,
  max = 4,
  className,
}: {
  names: string[]
  max?: number
  className?: string
}) {
  if (names.length === 0) return null
  const shown = names.slice(0, max)
  const hidden = names.length - shown.length
  return (
    <span className={cn('inline-flex min-w-0 flex-wrap items-center gap-1', className)}>
      {shown.map((name) => (
        <span
          key={name}
          className="inline-flex h-5 items-center rounded-[5px] border border-border bg-surface-subtle px-1.5 font-mono text-[11px] text-foreground/80"
        >
          {name}
        </span>
      ))}
      {hidden > 0 ? (
        <span className="text-[11px] text-subtle-foreground" title={names.slice(max).join(', ')}>
          +{hidden}
        </span>
      ) : null}
    </span>
  )
}
