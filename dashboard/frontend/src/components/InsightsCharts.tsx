import type { ReactNode } from 'react'

import type {
  InsightsAggregateResponse,
  InsightsAggregateSummary,
  InsightsAggregateCurrencySummary,
} from '../pages/insightsPageTypes'
import { describeMissingAggregateCost } from '../pages/insightsCostAccounting'
import { modelText, useModelLabels } from '../pages/insightsModelLabelSupport'
import { cn } from '../ui'
import InsightsCostCoverage from './InsightsCostCoverage'
import InsightsProviderCacheSummary from './InsightsProviderCacheSummary'
import InsightsRankedBars, { InsightsTokenRows } from './InsightsRankedBars'
import { INSIGHTS_DETAILS, InsightsUnitsBadge } from './InsightsStats'
import {
  formatInsightsCompactCount,
  formatInsightsCost,
  formatInsightsCount,
  formatInsightsShare,
  insightsCurrencyLabel,
} from '../utils/insightsCost'

interface InsightsChartsProps {
  aggregate: InsightsAggregateResponse
}

type CostSummary = InsightsAggregateSummary | InsightsAggregateCurrencySummary

const total = (value: number | undefined, currency?: string) =>
  formatInsightsCost(value, currency, { total: true })

const NO_COMPARISON =
  'No request in range has both a priced model cost and a priced reference model, so there is nothing to compare.'

/** A figure, or a dash whose title says why the figure is missing. */
interface CostRow {
  label: string
  value?: string
  why?: string
  note?: string
  lead?: boolean
  tone?: 'muted'
}

function costRows(summary: CostSummary, missing?: string): CostRow[] {
  const compared = summary.cost_record_count > 0
  const why = compared ? undefined : (missing ?? NO_COMPARISON)
  // Positive when the routed requests cost more than the reference would have.
  const difference = compared ? -summary.total_saved : undefined
  const rows: CostRow[] = [
    {
      label: 'Model cost',
      value: compared ? total(summary.actual_spend, summary.currency) : undefined,
      why,
    },
    {
      label: 'Reference cost',
      value: compared ? total(summary.baseline_spend, summary.currency) : undefined,
      why,
      note: 'The same tokens at the reference model’s list price',
    },
    {
      label: 'Difference',
      lead: true,
      value:
        difference === undefined
          ? undefined
          : `${difference > 0 ? '+' : ''}${total(difference, summary.currency)}`,
      why,
      note:
        compared && summary.baseline_spend > 0
          ? `${formatInsightsShare(Math.abs(summary.total_saved), summary.baseline_spend)} ${summary.total_saved >= 0 ? 'below' : 'above'} the reference`
          : undefined,
    },
  ]
  if (compared && (summary.netted_router_call_spend ?? 0) !== 0) {
    rows.push({
      label: 'Router calls in the difference',
      value: total(summary.netted_router_call_spend, summary.currency),
      tone: 'muted',
      note: `Before them: ${total(-(summary.gross_saved ?? 0), summary.currency)}`,
    })
  }
  if ((summary.router_call_spend ?? 0) !== 0) {
    rows.push({
      label: 'Router calls',
      value: total(summary.router_call_spend, summary.currency),
      tone: 'muted',
    })
  }
  if (summary.total_spend !== undefined && summary.total_spend !== summary.actual_spend) {
    rows.push({
      label: 'Total cost',
      value: total(summary.total_spend, summary.currency),
      tone: 'muted',
    })
  }
  return rows
}

function CostTable({ rows, label }: { rows: CostRow[]; label: string }) {
  return (
    <table className="w-full text-[13px]" aria-label={label}>
      <tbody>
        {rows.map((row) => (
          <tr key={row.label} className="border-t border-border first:border-t-0">
            <th scope="row" className="py-1.5 pr-3 text-left font-normal text-muted-foreground">
              {row.label}
              {row.note ? (
                <span className="block text-[11.5px] text-subtle-foreground">{row.note}</span>
              ) : null}
            </th>
            <td
              className={cn(
                'py-1.5 text-right font-mono tabular',
                row.lead && 'text-[15px] font-semibold',
                row.tone === 'muted' && 'text-foreground/75',
              )}
            >
              {row.value ?? (
                <span
                  className="cursor-help text-subtle-foreground"
                  title={row.why}
                  aria-label={`Not available: ${row.why}`}
                >
                  —
                </span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

const plural = (count: number, word: string) =>
  `${formatInsightsCount(count)} ${word}${count === 1 ? '' : 's'}`

// The estimate caveat and, closed by default, how the amounts are reached.
function InsightsEstimateNotes() {
  return (
    <div className="flex flex-col gap-1.5 text-xs leading-relaxed text-muted-foreground">
      <p>
        Cost-only estimates from recorded tokens and configured list prices, against one reference
        model: not a quality comparison, not GPU bills or provider invoices.
      </p>
      <details className={INSIGHTS_DETAILS}>
        <summary>How estimates work</summary>
        <p>
          Each request is priced twice with its recorded tokens: at the selected model’s rate and at
          the reference model’s, in the same currency. The reference is the route’s router_replay
          plugin’s cost_baseline_model when it sets one, else the highest estimate in the recipe’s
          complete model pool across all decisions; a request sent to a dearer model than the
          reference shows a positive difference. Direct requests use the selected model as their
          reference when no reference model is set. Older records retain their captured reference.
        </p>
        <p>
          Decision-model, System One, embedding and Looper calls are costed apart from the selected
          model and added to the total; a call without a price is counted as unpriced, never as
          free. The difference includes the calls the Router made for the compared requests.
          Recorded router spend includes failed requests. A generation model without configured
          pricing uses the built-in catalog’s list price when its provider lists one, marked as an
          estimate. On a client’s own login a cost is the API-price equivalent, not what the plan
          charges.
        </p>
        <p>
          Infrastructure charges and invoice adjustments are excluded. Currencies are shown
          separately; no exchange-rate conversion is applied. Differences leave out requests that
          are not completed or lack token usage, model pricing, or reference data, and requests
          whose Router calls lack a price rather than netting them at zero. The total still counts
          the priced calls of a request that failed. Missing data is shown as an explicit reason in
          each record. Historical requests without captured prices remain “Price not recorded”;
          current model rates do not backfill them.
        </p>
      </details>
    </div>
  )
}

function Block({
  title,
  label,
  action,
  className,
  children,
}: {
  title: string
  label: string
  action?: ReactNode
  className?: string
  children: ReactNode
}) {
  return (
    <section
      className={cn(
        'flex min-w-0 flex-col gap-3 rounded-xl border border-border bg-card px-4 py-4 shadow-xs',
        className,
      )}
      aria-label={label}
    >
      <header className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold tracking-[-0.01em]">{title}</h3>
        {action}
      </header>
      {children}
    </section>
  )
}

function TokenFigure({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5" title={`${formatInsightsCount(value)} tokens`}>
      <span className="text-[11.5px] text-subtle-foreground">{label}</span>
      <span className="font-mono text-[15px] font-semibold tabular">
        {formatInsightsCompactCount(value)}
      </span>
    </div>
  )
}

export default function InsightsCharts({ aggregate }: InsightsChartsProps) {
  const labels = useModelLabels()
  if (aggregate.record_count === 0) {
    return null
  }
  const summary = aggregate.summary
  const costGroups = summary.by_currency?.length ? summary.by_currency : [summary]
  const missing = summary.coverage ? describeMissingAggregateCost(summary.coverage) : undefined
  const tokenVolume = aggregate.token_volume
  const named = <T extends { name: string }>(entries: T[]) =>
    entries.map((entry) => ({ ...entry, name: modelText(labels, entry.name) }))

  return (
    <section className="flex flex-col gap-4" aria-label="Insights summary">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Block
          title="Cost estimate"
          label="Cost estimates"
          action={<InsightsUnitsBadge currencies={costGroups.map((group) => group.currency)} />}
        >
          {costGroups.map((group) => (
            <div key={group.currency || 'unavailable'} className="flex flex-col gap-1">
              {costGroups.length > 1 ? (
                <h4 className="text-xs font-medium text-muted-foreground">
                  {insightsCurrencyLabel(group.currency) || 'Price units'}
                </h4>
              ) : null}
              <CostTable
                rows={costRows(group, missing)}
                label={`${insightsCurrencyLabel(group.currency) || 'Price unit'} cost estimate`}
              />
            </div>
          ))}
          <InsightsEstimateNotes />
          {summary.excluded_record_count > 0 ? (
            <p className="text-xs text-muted-foreground">
              {plural(summary.excluded_record_count, 'filtered record')} left out of the comparison;
              each row explains what was not recorded. Historical rows without captured prices are
              not repriced.
            </p>
          ) : null}
          <InsightsCostCoverage summary={summary} />
        </Block>

        <Block title="Tokens" label="Token volume">
          <div className="grid grid-cols-3 gap-3">
            <TokenFigure label="Input" value={tokenVolume.input_tokens} />
            <TokenFigure label="Output" value={tokenVolume.output_tokens} />
            <TokenFigure label="Total" value={tokenVolume.total_tokens} />
          </div>
          {tokenVolume.excluded_record_count > 0 ? (
            <p className="text-xs text-muted-foreground">
              {plural(tokenVolume.excluded_record_count, 'filtered record')} left out of the token
              totals because usage data is incomplete.
            </p>
          ) : null}
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            <InsightsTokenRows
              title="By decision"
              entries={aggregate.token_breakdown.by_decision}
            />
            <InsightsTokenRows
              title="By model"
              entries={named(aggregate.token_breakdown.by_selected_model)}
            />
          </div>
        </Block>
      </div>

      <Block
        title="Routing"
        label="Routing"
        action={
          <span className="text-xs text-subtle-foreground">
            {plural(aggregate.record_count, 'record')}
          </span>
        }
      >
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
          <InsightsRankedBars
            title="Model selection"
            entries={named(aggregate.model_selection)}
            total={aggregate.record_count}
            color="var(--chart-6)"
          />
          <InsightsRankedBars
            title="Decision distribution"
            entries={aggregate.decision_distribution}
            total={aggregate.record_count}
            color="var(--chart-3)"
          />
          <InsightsRankedBars
            title="Signal distribution"
            entries={aggregate.signal_distribution}
            meta="share of matches"
            color="var(--chart-2)"
          />
        </div>
      </Block>

      <InsightsProviderCacheSummary summary={aggregate.provider_cache} />
    </section>
  )
}
