import type { LucideIcon } from 'lucide-react'
import { Brain, Lock, ShieldAlert, Split, Zap } from 'lucide-react'

import { Badge, cn } from '../ui'

import { decisionTone, type Tone } from './insightsDecisionTone'

const DECISION_ICONS: Record<Tone, LucideIcon> = {
  guard: ShieldAlert,
  privacy: Lock,
  deep: Brain,
  fast: Zap,
  route: Split,
}

/** The icon a decision's name reads as, alone. */
export function DecisionIcon({ decision, className }: { decision: string; className?: string }) {
  const Icon = DECISION_ICONS[decisionTone(decision)]
  return <Icon className={className} aria-hidden />
}

// A decision as a violet pill with the icon its name reads as; `label` is the
// name to show. `quiet` drops the pill for a table where every row has one.
export default function InsightsDecisionBadge({
  decision,
  label,
  quiet = false,
  className,
}: {
  decision: string
  label: string
  quiet?: boolean
  className?: string
}) {
  const tone = decisionTone(decision)
  const Icon = DECISION_ICONS[tone]
  if (quiet) {
    return (
      <span
        data-tone={tone}
        title={decision}
        className={cn(
          'inline-flex min-w-0 items-center gap-1.5 text-[13px] text-foreground/85',
          className,
        )}
      >
        <Icon className="size-3.5 shrink-0 text-chart-3" />
        <span className="truncate">{label}</span>
      </span>
    )
  }
  return (
    <Badge
      tone="violet"
      data-tone={tone}
      title={decision}
      className={cn('max-w-[180px]', className)}
    >
      <Icon />
      <span className="truncate">{label}</span>
    </Badge>
  )
}
