import type {
  InsightsAggregateResponse,
  InsightsAggregateSummary,
  InsightsAggregateCurrencySummary,
} from '../pages/insightsPageTypes'
import { modelText, useModelLabels } from '../pages/insightsModelLabelSupport'
import styles from './InsightsCharts.module.css'
import InsightsCostCoverage from './InsightsCostCoverage'
import InsightsProviderCacheSummary from './InsightsProviderCacheSummary'
import InsightsRankedBars, { InsightsTokenRows } from './InsightsRankedBars'
import { InsightsStatGrid, InsightsUnitsBadge, type InsightsStat } from './InsightsStats'
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

// Router calls (a Looper's model calls and the questions the Router asks to
// route) are a separate cost, and the total adds them and the priced requests
// without a baseline, failed ones included; each card shows only where it
// differs from the model cost.
const spendStats = (summary: CostSummary): InsightsStat[] => [
  ...((summary.router_call_spend ?? 0) !== 0
    ? [
        {
          label: 'Estimated Router Calls',
          value: total(summary.router_call_spend, summary.currency),
          tone: 'muted' as const,
        },
      ]
    : []),
  ...(summary.total_spend !== undefined && summary.total_spend !== summary.actual_spend
    ? [
        {
          label: 'Estimated Total Cost',
          value: total(summary.total_spend, summary.currency),
          tone: 'muted' as const,
        },
      ]
    : []),
]

// A currency can have priced costs but no request compared with a baseline;
// its savings are then unknown, not zero.
const comparedCost = (summary: CostSummary, value: number) =>
  summary.cost_record_count > 0 ? total(value, summary.currency) : 'N/A'

// Savings are net of the compared requests' Router calls; where those cost
// anything, the gross figure and what was netted show beside it.
const netStats = (summary: CostSummary): InsightsStat[] =>
  summary.cost_record_count > 0 && (summary.netted_router_call_spend ?? 0) !== 0
    ? [
        {
          label: 'Estimated Gross Savings',
          value: total(summary.gross_saved, summary.currency),
          tone: 'muted',
        },
        {
          label: 'Router Calls Netted',
          value: total(summary.netted_router_call_spend, summary.currency),
          tone: 'muted',
        },
      ]
    : []

const summaryStats = (summary: CostSummary): InsightsStat[] => [
  {
    label: 'Estimated Savings',
    value: comparedCost(summary, summary.total_saved),
    lead: true,
    tone: summary.total_saved < 0 ? 'negative' : undefined,
  },
  {
    label: 'Estimated Saved %',
    value:
      summary.cost_record_count > 0 && summary.baseline_spend > 0
        ? formatInsightsShare(summary.total_saved, summary.baseline_spend)
        : 'N/A',
  },
  { label: 'Estimated Baseline', value: comparedCost(summary, summary.baseline_spend) },
  { label: 'Estimated Model Cost', value: comparedCost(summary, summary.actual_spend) },
  ...netStats(summary),
  ...spendStats(summary),
]

const plural = (count: number, word: string) =>
  `${formatInsightsCount(count)} ${word}${count === 1 ? '' : 's'}`

// The estimate caveat and, closed by default, how the amounts are reached.
function InsightsEstimateNotes() {
  return (
    <div className={styles.estimateContext}>
      <p className={styles.summaryHint}>
        Estimates from recorded tokens and configured rates, not GPU bills or provider invoices.
      </p>
      <details className={styles.estimateDetails}>
        <summary>How estimates work</summary>
        <p>
          New records use the highest estimate in the recipe’s complete model pool across all
          decisions, in the same currency, as baseline using the same recorded tokens, unless the
          route’s router_replay plugin sets cost_baseline_model: that model is then the baseline,
          and a request sent to a dearer model shows a negative saving. Direct requests use the
          selected model as baseline when no baseline model is set. Older records retain their
          captured baseline.
        </p>
        <p>
          Decision-model, System One, embedding and Looper calls are costed apart from the selected
          model and added to the total; a call without a price is counted as unpriced, never as
          free. Savings are net of the calls the Router made for the compared requests; the gross
          figure is before them. Recorded router spend includes failed requests. A generation model
          without configured pricing uses the built-in catalog’s list price when its provider lists
          one, marked as an estimate. On a client’s own login a cost is the API-price equivalent,
          not what the plan charges.
        </p>
        <p>
          Infrastructure charges and invoice adjustments are excluded. Currencies are shown
          separately; no exchange-rate conversion is applied. Savings leave out requests that are
          not completed or lack token usage, model pricing, or baseline data, and requests whose
          Router calls lack a price rather than netting them at zero. The total still counts the
          priced calls of a request that failed. Missing data is shown as an explicit reason in each
          record. Historical requests without captured prices remain “Price not recorded”; current
          model rates do not backfill them.
        </p>
      </details>
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
  const tokenVolume = aggregate.token_volume
  const named = <T extends { name: string }>(entries: T[]) =>
    entries.map((entry) => ({ ...entry, name: modelText(labels, entry.name) }))
  const tokenStat = (label: string, value: number): InsightsStat => ({
    label,
    value: formatInsightsCompactCount(value),
    caption: `${formatInsightsCount(value)} tokens`,
  })

  return (
    <section className={styles.container} aria-label="Insights summary">
      <section className={styles.block} aria-label="Cost estimates">
        <header className={styles.blockHeader}>
          <h3 className={styles.blockTitle}>Cost and Savings</h3>
          <InsightsUnitsBadge currencies={costGroups.map((group) => group.currency)} />
        </header>
        <InsightsEstimateNotes />
        {costGroups.map((group) => (
          <div key={group.currency || 'unavailable'} className={styles.group}>
            {costGroups.length > 1 ? (
              <h4 className={styles.chartTitle}>
                {insightsCurrencyLabel(group.currency) || 'Price unit'} estimates
              </h4>
            ) : null}
            <InsightsStatGrid stats={summaryStats(group)} />
          </div>
        ))}
        {summary.excluded_record_count > 0 ? (
          <p className={styles.summaryHint}>
            {plural(summary.excluded_record_count, 'filtered record')} excluded from savings
            estimates. Each row explains what was not recorded. Historical rows without captured
            prices are not repriced.
          </p>
        ) : null}
        <InsightsCostCoverage summary={summary} />
      </section>

      <InsightsProviderCacheSummary summary={aggregate.provider_cache} />

      <section className={styles.block} aria-label="Routing">
        <header className={styles.blockHeader}>
          <h3 className={styles.blockTitle}>Routing</h3>
          <span className={styles.panelMeta}>{plural(aggregate.record_count, 'record')}</span>
        </header>
        <div className={styles.panelGrid}>
          <InsightsRankedBars
            title="Model Selection"
            entries={named(aggregate.model_selection)}
            total={aggregate.record_count}
          />
          <InsightsRankedBars
            title="Decision Distribution"
            entries={aggregate.decision_distribution}
            total={aggregate.record_count}
          />
          <InsightsRankedBars
            title="Signal Distribution"
            entries={aggregate.signal_distribution}
            meta="share of matches"
          />
        </div>
      </section>

      <section className={styles.block} aria-label="Token volume">
        <header className={styles.blockHeader}>
          <h3 className={styles.blockTitle}>Token Volume</h3>
        </header>
        <InsightsStatGrid
          stats={[
            tokenStat('Input Tokens', tokenVolume.input_tokens),
            tokenStat('Output Tokens', tokenVolume.output_tokens),
            { ...tokenStat('Total Tokens', tokenVolume.total_tokens), lead: true },
          ]}
        />
        {tokenVolume.excluded_record_count > 0 ? (
          <p className={styles.summaryHint}>
            {plural(tokenVolume.excluded_record_count, 'filtered record')} excluded from token
            totals because usage data is incomplete.
          </p>
        ) : null}
        <div className={styles.panelGrid}>
          <InsightsTokenRows
            title="Tokens by Decision"
            entries={aggregate.token_breakdown.by_decision}
          />
          <InsightsTokenRows
            title="Tokens by Selected Model"
            entries={named(aggregate.token_breakdown.by_selected_model)}
          />
        </div>
      </section>
    </section>
  )
}
