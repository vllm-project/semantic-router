import { useState, type ReactNode } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'

import { Skeleton, cn } from '../ui'
import { formatInsightsCount, formatInsightsShare } from '../utils/insightsCost'

export interface InsightsRankedEntry {
  /** A stable key, kept as the row's data attribute. */
  id: string
  label: string
  value: number
  /** The logo or icon at the head of the row, 18px wide. */
  mark?: ReactNode
  /** The full name, when the label shortens it. */
  title?: string
}

interface InsightsRankedBarsProps {
  title: string
  /** The header icon, colored as the summary row colors its icons. */
  icon: ReactNode
  /** The color of every bar in the card, a chart token. */
  color: string
  /** The list's accessible name. */
  label: string
  /** The name each row's data attribute takes, as data-harness. */
  rowKey?: string
  entries: InsightsRankedEntry[] | null
  /** The whole each share is of: the entries' sum unless given. */
  total?: number
  meta?: ReactNode
  /** Rows shown before the rest fold behind "Show more". */
  visible?: number
}

const ROW =
  'grid grid-cols-[18px_minmax(0,1fr)_auto_3.25rem] items-center gap-x-2.5 gap-y-1.5 py-1.5 text-[13px]'

/**
 * One card of the Breakdown: a header with its icon, then a ranked row per
 * entry (its mark, name, count and share over one slim bar), the same anatomy
 * for agents, models and decisions.
 */
export default function InsightsRankedBars({
  title,
  icon,
  color,
  label,
  rowKey,
  entries,
  total,
  meta,
  visible = 7,
}: InsightsRankedBarsProps) {
  const [all, setAll] = useState(false)
  const whole = total ?? entries?.reduce((sum, entry) => sum + entry.value, 0) ?? 0
  const peak = Math.max(0, ...(entries ?? []).map((entry) => entry.value))
  const shown = entries ? (all ? entries : entries.slice(0, visible)) : []
  const hidden = entries ? entries.length - visible : 0
  return (
    <section
      className="flex min-w-0 flex-col gap-2 rounded-xl border border-border bg-card px-4 pb-3 pt-3.5 shadow-xs"
      aria-label={title}
    >
      <header className="flex items-center justify-between gap-2">
        <h4 className="flex min-w-0 items-center gap-1.5 text-[13px] font-semibold [&_svg]:size-3.5 [&_svg]:shrink-0">
          {icon}
          {title}
        </h4>
        {meta ? <span className="shrink-0 text-xs text-subtle-foreground">{meta}</span> : null}
      </header>
      {entries === null ? (
        <div className="flex flex-col gap-3 py-1.5" aria-busy>
          {Array.from({ length: 4 }, (_, index) => (
            <Skeleton key={index} className="h-6 w-full" />
          ))}
        </div>
      ) : entries.length === 0 ? (
        <p className="py-2 text-[13px] text-muted-foreground">Nothing recorded in range.</p>
      ) : (
        <>
          <ol className="flex flex-col" aria-label={label}>
            {shown.map((entry, rank) => (
              <li
                key={entry.id}
                className={ROW}
                {...(rowKey ? { [`data-${rowKey}`]: entry.id } : {})}
              >
                <span className="grid size-[18px] place-items-center">{entry.mark}</span>
                <span className="truncate" title={entry.title ?? entry.label}>
                  {entry.label}
                </span>
                <span
                  className="text-right font-mono text-[12.5px] tabular"
                  title={`${formatInsightsCount(entry.value)} requests`}
                >
                  {formatInsightsCount(entry.value)}
                </span>
                <span className="text-right font-mono text-xs text-muted-foreground tabular">
                  {formatInsightsShare(entry.value, whole)}
                </span>
                <span
                  className="col-start-2 col-end-[-1] h-[3px] overflow-hidden rounded-full bg-muted"
                  aria-hidden
                >
                  <span
                    className={cn('block h-full rounded-full', rank > 0 && 'opacity-55')}
                    style={{
                      width: `${peak > 0 ? Math.max((entry.value / peak) * 100, 1.5) : 0}%`,
                      background: color,
                    }}
                  />
                </span>
              </li>
            ))}
          </ol>
          {hidden > 0 ? (
            <button
              type="button"
              aria-expanded={all}
              onClick={() => setAll((value) => !value)}
              className="-ml-1.5 inline-flex w-fit items-center gap-1 rounded-md px-1.5 py-1 text-[12.5px] text-muted-foreground transition-colors hover:bg-surface-subtle hover:text-foreground focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
            >
              {all ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
              {all ? 'Show fewer' : `Show ${hidden} more`}
            </button>
          ) : null}
        </>
      )}
    </section>
  )
}
