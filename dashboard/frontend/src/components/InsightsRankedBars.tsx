import type { ReactNode } from 'react'

import type { InsightsAggregateTokenEntry } from '../pages/insightsPageTypes'
import { cn } from '../ui'
import {
  formatInsightsCompactCount,
  formatInsightsCount,
  formatInsightsShare,
} from '../utils/insightsCost'

export interface InsightsRankedEntry {
  name: string
  value: number
}

interface InsightsRankedBarsProps {
  title: string
  entries: InsightsRankedEntry[]
  /** The whole each share is of: the entries' sum when the response lists every entry. */
  total?: number
  meta?: string
  /** Rows shown before the rest fold away. */
  visible?: number
  /** The bar color, a chart token. */
  color?: string
}

const barWidth = (value: number, peak: number) => `${peak > 0 ? (value / peak) * 100 : 0}%`

const ROW =
  'grid grid-cols-[minmax(0,1fr)_auto_3.5rem] items-center gap-x-3 gap-y-1 py-1.5 text-[13px]'

function Panel({
  title,
  meta,
  children,
}: {
  title: string
  meta?: ReactNode
  children: ReactNode
}) {
  return (
    <section
      className="flex min-w-0 flex-col gap-2 rounded-lg border border-border bg-surface px-3.5 py-3"
      aria-label={title}
    >
      <div className="flex items-baseline justify-between gap-2">
        <h4 className="text-[13px] font-semibold">{title}</h4>
        {meta ? <span className="text-xs text-subtle-foreground">{meta}</span> : null}
      </div>
      {children}
    </section>
  )
}

/** A distribution as ranked rows, which stay legible however many entries a history holds. */
export default function InsightsRankedBars({
  title,
  entries,
  total,
  meta,
  visible = 8,
  color = 'var(--chart-1)',
}: InsightsRankedBarsProps) {
  const ranked = [...entries].sort((a, b) => b.value - a.value || a.name.localeCompare(b.name))
  const whole = total ?? ranked.reduce((sum, entry) => sum + entry.value, 0)
  const peak = ranked[0]?.value ?? 0
  const row = (entry: InsightsRankedEntry, rank: number) => (
    <li key={entry.name} className={ROW}>
      <span className="truncate" title={entry.name}>
        {entry.name}
      </span>
      <span className="font-mono text-xs tabular">{formatInsightsCount(entry.value)}</span>
      <span className="text-right font-mono text-xs text-subtle-foreground tabular">
        {formatInsightsShare(entry.value, whole)}
      </span>
      <span className="col-span-full h-1 overflow-hidden rounded-full bg-muted" aria-hidden="true">
        <span
          className={cn('block h-full rounded-full', rank > 0 && 'opacity-70')}
          style={{ width: barWidth(entry.value, peak), background: color }}
        />
      </span>
    </li>
  )
  return (
    <Panel title={title} meta={meta}>
      {ranked.length === 0 ? (
        <p className="text-xs text-muted-foreground">Nothing recorded yet.</p>
      ) : (
        <>
          <ol className="flex flex-col">
            {ranked.slice(0, visible).map((entry, rank) => row(entry, rank))}
          </ol>
          {ranked.length > visible ? (
            <details className="group text-xs">
              <summary className="cursor-pointer select-none text-muted-foreground hover:text-foreground">
                {ranked.length - visible} more
              </summary>
              <ol className="flex flex-col" start={visible + 1}>
                {ranked.slice(visible).map((entry, rank) => row(entry, rank + visible))}
              </ol>
            </details>
          ) : null}
        </>
      )}
    </Panel>
  )
}

/** Each entry's tokens as one bar split into input and output, ranked by the total. */
export function InsightsTokenRows({
  title,
  entries,
}: {
  title: string
  entries: InsightsAggregateTokenEntry[]
}) {
  const peak = Math.max(0, ...entries.map((entry) => entry.total_tokens))
  return (
    <Panel
      title={title}
      meta={
        <span className="inline-flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5">
            <span className="size-2 rounded-[3px] bg-chart-4" />
            Input
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="size-2 rounded-[3px] bg-chart-3" />
            Output
          </span>
        </span>
      }
    >
      {entries.length === 0 ? (
        <p className="text-xs text-muted-foreground">Nothing recorded yet.</p>
      ) : (
        <ol className="flex flex-col">
          {entries.map((entry) => (
            <li key={entry.name} className={ROW}>
              <span className="truncate" title={entry.name}>
                {entry.name}
              </span>
              <span
                className="font-mono text-xs tabular"
                title={`${formatInsightsCount(entry.total_tokens)} tokens`}
              >
                {formatInsightsCompactCount(entry.total_tokens)}
              </span>
              <span className="text-right font-mono text-xs text-subtle-foreground tabular">
                {formatInsightsShare(entry.output_tokens, entry.total_tokens)} out
              </span>
              <span
                className="col-span-full flex h-1 overflow-hidden rounded-full bg-muted"
                aria-hidden="true"
              >
                <span
                  className="block h-full bg-chart-4"
                  style={{ width: barWidth(entry.input_tokens, peak) }}
                />
                <span
                  className="block h-full bg-chart-3"
                  style={{ width: barWidth(entry.output_tokens, peak) }}
                />
              </span>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  )
}
