import type {
  InsightsAggregateResponse,
  InsightsAggregateSummary,
  InsightsAggregateCurrencySummary,
} from '../pages/insightsPageTypes'
import styles from './InsightsCharts.module.css'
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

const summaryStats = (summary: CostSummary): InsightsStat[] => [
  {
    label: 'Estimated Savings',
    value: total(summary.total_saved, summary.currency),
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
  { label: 'Estimated Baseline', value: total(summary.baseline_spend, summary.currency) },
  { label: 'Estimated Model Cost', value: total(summary.actual_spend, summary.currency) },
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
          decisions, in the same currency, as baseline using the same recorded tokens. Direct
          requests use the selected model as baseline. Older records retain their captured
          baseline.
        </p>
        <p>
          Infrastructure charges and invoice adjustments are excluded. Currencies are shown
          separately; no exchange-rate conversion is applied. Requests that are not completed or
          lack token usage, model pricing, or baseline data are excluded. Missing data is shown as
          an explicit reason in each record. Historical requests without captured prices remain
          “Price not recorded”; current model rates do not backfill them.
        </p>
      </details>
    </div>
  )
}

export default function InsightsCharts({ aggregate }: InsightsChartsProps) {
  if (aggregate.record_count === 0) {
    return null
  }
  const summary = aggregate.summary
  const costGroups = summary.by_currency?.length ? summary.by_currency : [summary]
  const tokenVolume = aggregate.token_volume
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
            {plural(summary.excluded_record_count, 'filtered record')} excluded from estimates.
            Each row explains what was not recorded. Historical rows without captured prices are
            not repriced.
          </p>
        ) : null}
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
            entries={aggregate.model_selection}
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
            entries={aggregate.token_breakdown.by_selected_model}
          />
        </div>
      </section>
    </section>
  )
}
