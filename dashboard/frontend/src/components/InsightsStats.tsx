import { cn } from '../ui'
import { isInsightsNoCurrency } from '../utils/insightsCost'

export interface InsightsStat {
  label: string
  value: string
  /** The figure its block is about, marked with the accent. */
  lead?: boolean
  tone?: 'negative' | 'muted'
  caption?: string
}

/** A disclosure under a block; its summary stays class-free. */
export const INSIGHTS_DETAILS =
  'rounded-lg border border-border bg-surface-subtle/50 px-3 py-2 [&>summary]:cursor-pointer [&>summary]:select-none [&>summary]:font-medium [&>summary]:text-foreground/85 [&>summary:hover]:text-foreground [&>p]:mt-2'

export function InsightsStatGrid({ stats }: { stats: InsightsStat[] }) {
  return (
    <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 xl:grid-cols-4">
      {stats.map((stat) => (
        <article
          key={stat.label}
          className={cn(
            'flex min-w-0 flex-col gap-1 rounded-lg border border-border bg-surface-subtle/60 px-3 py-2.5',
            stat.lead && 'border-primary/25 bg-primary-soft/60',
          )}
        >
          <span className="truncate text-xs text-subtle-foreground">{stat.label}</span>
          <strong
            className={cn(
              'truncate text-lg font-semibold tracking-[-0.02em] tabular',
              stat.tone === 'negative' && 'text-destructive-foreground',
              stat.tone === 'muted' && 'text-foreground/75',
            )}
          >
            {stat.value}
          </strong>
          {stat.caption ? (
            <span className="truncate text-[11.5px] text-muted-foreground">{stat.caption}</span>
          ) : null}
        </article>
      ))}
    </div>
  )
}

/** Marks amounts whose prices name no currency (ISO 4217 XXX): they are plain price units. */
export function InsightsUnitsBadge({ currencies }: { currencies: Array<string | undefined> }) {
  return currencies.some((currency) => isInsightsNoCurrency(currency)) ? (
    <span
      className="inline-flex h-5 items-center rounded-full border border-border bg-secondary px-2 text-[11px] font-medium text-muted-foreground"
      title="The configured prices name no currency (XXX), so amounts are in price units."
    >
      Price units
    </span>
  ) : null
}
