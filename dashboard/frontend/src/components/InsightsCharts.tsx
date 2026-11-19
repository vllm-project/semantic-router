import type { ReactNode } from 'react'
import { Info } from 'lucide-react'

import type {
  InsightsAggregateCurrencySummary,
  InsightsAggregateResponse,
  InsightsAggregateSummary,
} from '../pages/insightsPageTypes'
import { describeMissingAggregateCost } from '../pages/insightsCostAccounting'
import { HarnessLogo } from '../pages/InsightsHarness'
import { API_CLIENT_LABEL, harnessInfo, sortHarnessIds } from '../pages/insightsHarness'
import { modelText, useModelLabels } from '../pages/insightsModelLabelSupport'
import { cn, Tooltip } from '../ui'
import InsightsRankedBars from './InsightsRankedBars'
import {
  formatInsightsCompactCount,
  formatInsightsCost,
  formatInsightsCount,
  formatInsightsShare,
  insightsCurrencyLabel,
} from '../utils/insightsCost'

interface InsightsChartsProps {
  aggregate: InsightsAggregateResponse
  /** Requests by harness under the page's other filters, from the records list. */
  harnessCounts?: Record<string, number> | null
}

type CostSummary = InsightsAggregateSummary | InsightsAggregateCurrencySummary

const API_CLIENT = 'api_client'

const NO_COMPARISON =
  'No request in range has both a priced model cost and a priced reference model, so there is nothing to compare.'

const total = (value: number | undefined, currency?: string) =>
  formatInsightsCost(value, currency, { total: true })

const plural = (count: number, word: string) =>
  `${formatInsightsCount(count)} ${word}${count === 1 ? '' : 's'}`

interface Metric {
  label: string
  value?: string
  /** Why the value is missing, shown on the dash. */
  why?: string
  hint?: string
  /** Colors the hint: below the reference reads as good, above as a caution. */
  tone?: 'good' | 'caution'
  title?: string
}

function leadGroup(summary: InsightsAggregateSummary): CostSummary {
  const groups: CostSummary[] = summary.by_currency?.length ? summary.by_currency : [summary]
  const spend = (group: CostSummary) => group.total_spend ?? group.actual_spend
  return groups.reduce((best, group) => (spend(group) > spend(best) ? group : best))
}

function metrics(aggregate: InsightsAggregateResponse): Metric[] {
  const summary = aggregate.summary
  const group = leadGroup(summary)
  const compared = group.cost_record_count > 0
  const missing = summary.coverage ? describeMissingAggregateCost(summary.coverage) : undefined
  const why = compared ? undefined : (missing ?? NO_COMPARISON)
  const tokens = aggregate.token_volume
  const cache = aggregate.provider_cache
  // Positive when the routed requests cost more than the reference would have.
  const difference = compared ? -group.total_saved : undefined
  return [
    {
      label: 'Requests',
      value: formatInsightsCount(aggregate.record_count),
      hint: `${formatInsightsCount(aggregate.lifecycle.completed)} completed`,
    },
    {
      label: 'Input tokens',
      value: formatInsightsCompactCount(tokens.input_tokens),
      title: `${formatInsightsCount(tokens.input_tokens)} tokens`,
      hint:
        tokens.total_tokens > 0
          ? `${formatInsightsShare(tokens.input_tokens, tokens.total_tokens)} of all`
          : undefined,
    },
    {
      label: 'Output tokens',
      value: formatInsightsCompactCount(tokens.output_tokens),
      title: `${formatInsightsCount(tokens.output_tokens)} tokens`,
      hint:
        tokens.excluded_record_count > 0
          ? `${plural(tokens.excluded_record_count, 'request')} without usage`
          : undefined,
    },
    {
      label: 'Prompt cache',
      value:
        cache && cache.priced_records > 0 ? total(cache.net_savings, cache.currency) : undefined,
      why: 'No priced request reported cached prompt tokens.',
      hint:
        cache && cache.no_cache_spend > 0
          ? `${formatInsightsShare(cache.net_savings, cache.no_cache_spend)} off cache-less cost`
          : undefined,
    },
    {
      label: 'Model cost',
      value: compared ? total(group.actual_spend, group.currency) : undefined,
      why,
      hint: compared ? insightsCurrencyLabel(group.currency) || 'Price units' : undefined,
    },
    {
      label: 'Vs reference',
      value:
        difference === undefined
          ? undefined
          : `${difference > 0 ? '+' : ''}${total(difference, group.currency)}`,
      why,
      hint:
        compared && group.baseline_spend > 0
          ? `${formatInsightsShare(Math.abs(group.total_saved), group.baseline_spend)} ${group.total_saved >= 0 ? 'below' : 'above'} reference`
          : undefined,
      tone: difference === undefined ? undefined : difference > 0 ? 'caution' : 'good',
    },
  ]
}

function MetricTile({ metric }: { metric: Metric }) {
  return (
    <div className="flex min-w-0 flex-col gap-1 px-4 py-3.5" title={metric.title}>
      <span className="text-[11.5px] font-medium uppercase tracking-[0.04em] text-subtle-foreground">
        {metric.label}
      </span>
      {metric.value ? (
        <span className="truncate text-[22px] font-semibold leading-tight tracking-[-0.02em] tabular">
          {metric.value}
        </span>
      ) : (
        <Tooltip content={metric.why}>
          <span
            className="w-fit cursor-help text-[22px] font-semibold leading-tight text-subtle-foreground"
            aria-label={`Not available: ${metric.why}`}
          >
            —
          </span>
        </Tooltip>
      )}
      <span
        className={cn(
          'truncate text-xs text-muted-foreground',
          metric.tone === 'good' && 'text-success-foreground',
          metric.tone === 'caution' && 'text-warning-foreground',
        )}
      >
        {metric.hint ?? '\u00a0'}
      </span>
    </div>
  )
}

/** The method and what was left out, held in one footnote's tooltip. */
function methodNote(aggregate: InsightsAggregateResponse): string {
  const summary = aggregate.summary
  const coverage = summary.coverage
  const parts = [
    'Each request is priced twice from its recorded tokens: at the selected model’s list price and at the reference model’s.',
    'The reference is the route’s cost_baseline_model, else the dearest model of its recipe’s pool.',
    'Router calls are costed apart; nothing here is a GPU bill, an invoice or a quality comparison.',
  ]
  if (coverage) {
    parts.push(
      `Priced: ${formatInsightsCount(coverage.priced_records)} of ${formatInsightsCount(aggregate.record_count)} requests` +
        (coverage.partly_priced_records
          ? `, ${formatInsightsCount(coverage.partly_priced_records)} partly`
          : '') +
        '.',
    )
  }
  if (summary.excluded_record_count > 0) {
    parts.push(`${plural(summary.excluded_record_count, 'request')} left out of the comparison.`)
  }
  return parts.join(' ')
}

interface HarnessRow {
  id: string
  label: string
  value: number
}

function harnessRows(counts: Record<string, number>): HarnessRow[] {
  const ids = sortHarnessIds(
    Object.keys(counts).filter((id) => id !== API_CLIENT && counts[id] > 0),
  )
  const rows = ids.map((id) => ({ id, label: harnessInfo(id)?.label ?? id, value: counts[id] }))
  if (counts[API_CLIENT] > 0)
    rows.push({ id: API_CLIENT, label: API_CLIENT_LABEL, value: counts[API_CLIENT] })
  return rows
}

/** Requests by agent harness: each bar opens with its harness's logo. */
export function InsightsHarnessBars({ counts }: { counts: Record<string, number> }) {
  const rows = harnessRows(counts)
  const sum = rows.reduce((acc, row) => acc + row.value, 0)
  const peak = Math.max(0, ...rows.map((row) => row.value))
  return (
    <Panel title="Agents" meta={plural(sum, 'request')}>
      {rows.length === 0 ? (
        <p className="py-2 text-[13px] text-muted-foreground">No requests in range.</p>
      ) : (
        <ul className="flex flex-col gap-2" aria-label="Requests by agent harness">
          {rows.map((row) => (
            <li
              key={row.id}
              className="grid grid-cols-[minmax(0,1fr)_auto_3.25rem] items-center gap-3"
              data-harness={row.id}
            >
              <div className="relative h-8 min-w-0 overflow-hidden rounded-lg bg-surface-subtle">
                <span
                  aria-hidden
                  className="absolute inset-y-0 left-0 rounded-lg bg-chart-1/20"
                  style={{ width: `${peak > 0 ? Math.max((row.value / peak) * 100, 2) : 0}%` }}
                />
                <span className="relative flex h-full min-w-0 items-center gap-2 px-2">
                  <HarnessLogo
                    harness={row.id === API_CLIENT ? undefined : row.id}
                    size={18}
                    framed
                  />
                  <span className="truncate text-[13px] font-medium">{row.label}</span>
                </span>
              </div>
              <span
                className="text-right font-mono text-[13px] tabular"
                title={`${formatInsightsCount(row.value)} requests`}
              >
                {formatInsightsCompactCount(row.value)}
              </span>
              <span className="text-right font-mono text-xs text-muted-foreground tabular">
                {formatInsightsShare(row.value, sum)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}

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
      className="flex min-w-0 flex-col gap-3 rounded-xl border border-border bg-card px-4 py-3.5 shadow-xs"
      aria-label={title}
    >
      <header className="flex items-baseline justify-between gap-2">
        <h4 className="text-[13px] font-semibold">{title}</h4>
        {meta ? <span className="text-xs text-subtle-foreground">{meta}</span> : null}
      </header>
      {children}
    </section>
  )
}

// The Breakdown: the headline figures in one grid, the estimate's method behind
// one footnote, and the requests by agent harness, model and decision.
export default function InsightsCharts({ aggregate, harnessCounts }: InsightsChartsProps) {
  const labels = useModelLabels()
  if (aggregate.record_count === 0) {
    return null
  }
  const named = <T extends { name: string }>(entries: T[]) =>
    entries.map((entry) => ({ ...entry, name: modelText(labels, entry.name) }))
  return (
    <section className="flex flex-col gap-3" aria-label="Insights summary">
      <div
        role="group"
        aria-label="Breakdown figures"
        className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-border bg-border shadow-xs sm:grid-cols-3 xl:grid-cols-6 [&>*]:bg-card"
      >
        {metrics(aggregate).map((metric) => (
          <MetricTile key={metric.label} metric={metric} />
        ))}
      </div>
      <p className="flex items-center gap-1.5 px-1 text-xs text-subtle-foreground">
        Cost-only estimate against a reference model’s list price.
        <Tooltip content={methodNote(aggregate)}>
          <button
            type="button"
            aria-label="How the estimate is made"
            className="grid size-4 place-items-center rounded-full text-subtle-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
          >
            <Info className="size-3.5" />
          </button>
        </Tooltip>
      </p>
      <div
        className={cn(
          'grid grid-cols-1 gap-3',
          harnessCounts ? 'lg:grid-cols-3' : 'lg:grid-cols-2',
        )}
      >
        {harnessCounts ? <InsightsHarnessBars counts={harnessCounts} /> : null}
        <InsightsRankedBars
          title="Models"
          entries={named(aggregate.model_selection)}
          total={aggregate.record_count}
          color="var(--chart-6)"
        />
        <InsightsRankedBars
          title="Decisions"
          entries={aggregate.decision_distribution}
          total={aggregate.record_count}
          color="var(--chart-3)"
        />
      </div>
    </section>
  )
}
