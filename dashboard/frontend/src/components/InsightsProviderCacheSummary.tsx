import type { CSSProperties } from 'react'

import type {
  InsightsAggregateProviderCache,
  InsightsProviderCacheCurrencyTotal,
  InsightsProviderCacheSpendEntry,
} from '../pages/insightsPageTypes'
import styles from './InsightsCharts.module.css'
import { InsightsStatGrid, InsightsUnitsBadge, type InsightsStat } from './InsightsStats'
import {
  formatInsightsCost,
  formatInsightsCount,
  formatInsightsShare,
  insightsCurrencyLabel,
} from '../utils/insightsCost'
import { modelText, useModelLabels } from '../pages/insightsModelLabelSupport'

interface InsightsProviderCacheSummaryProps {
  summary?: InsightsAggregateProviderCache
}

const netTone = (value: number) => (value < 0 ? ('negative' as const) : undefined)

const providerCacheStats = (group: InsightsProviderCacheCurrencyTotal): InsightsStat[] => [
  {
    label: 'Net Cache Saving',
    value: formatInsightsCost(group.net_savings, group.currency, { total: true }),
    lead: true,
    tone: netTone(group.net_savings),
  },
  {
    label: 'Net Saved %',
    value: formatInsightsShare(group.net_savings, group.no_cache_spend),
    tone: netTone(group.net_savings),
  },
  {
    label: 'No-Cache Estimate',
    value: formatInsightsCost(group.no_cache_spend, group.currency, { total: true }),
  },
  {
    label: 'Provider Cost',
    value: formatInsightsCost(group.actual_spend, group.currency, { total: true }),
  },
]

interface SpendTableProps {
  title: string
  nameLabel: string
  entries: InsightsProviderCacheSpendEntry[]
}

// One currency names its columns and leaves the cells bare; mixed currencies
// keep theirs in each cell. A row's bar is its net saving against the largest.
function SpendTable({ title, nameLabel, entries }: SpendTableProps) {
  const currencies = new Set(entries.map((entry) => entry.currency))
  const shared = currencies.size === 1
  const unit = shared ? insightsCurrencyLabel(entries[0]?.currency) : ''
  const heading = (label: string) => (unit ? `${label} (${unit})` : label)
  const amount = (value: number, currency: string) =>
    formatInsightsCost(value, currency, { total: true, bare: shared })
  const peak = Math.max(0, ...entries.map((entry) => Math.abs(entry.net_savings)))
  const bar = (value: number) =>
    ({ '--share': peak > 0 ? Math.abs(value) / peak : 0 }) as CSSProperties
  return (
    <div className={styles.chartSection}>
      <h4 className={styles.chartTitle}>{title}</h4>
      <div className={styles.tableScroll}>
        <table className={styles.spendTable}>
          <thead>
            <tr>
              <th scope="col">{nameLabel}</th>
              <th scope="col">Requests</th>
              <th scope="col">{heading('No-cache')}</th>
              <th scope="col">{heading('Provider cost')}</th>
              <th scope="col">{heading('Net saving')}</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <tr key={`${entry.name}\u0000${entry.currency}`}>
                <th scope="row">{entry.name}</th>
                <td>{formatInsightsCount(entry.priced_records)}</td>
                <td>{amount(entry.no_cache_spend, entry.currency)}</td>
                <td>{amount(entry.actual_spend, entry.currency)}</td>
                <td
                  className={
                    entry.net_savings < 0
                      ? `${styles.spendNegative} ${styles.spendBarNegative}`
                      : styles.spendBar
                  }
                  style={bar(entry.net_savings)}
                >
                  {amount(entry.net_savings, entry.currency)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function InsightsProviderCacheSummary({
  summary,
}: InsightsProviderCacheSummaryProps) {
  const labels = useModelLabels()
  if (!summary || summary.priced_records === 0) {
    return null
  }
  const groups: InsightsProviderCacheCurrencyTotal[] = summary.by_currency?.length
    ? summary.by_currency
    : [
        {
          currency: summary.currency ?? '',
          no_cache_spend: summary.no_cache_spend,
          actual_spend: summary.actual_spend,
          net_savings: summary.net_savings,
          priced_records: summary.priced_records,
        },
      ]

  return (
    <section className={styles.block} aria-label="Provider prompt cache">
      <header className={styles.blockHeader}>
        <h3 className={styles.blockTitle}>Provider Prompt Cache</h3>
        <InsightsUnitsBadge currencies={groups.map((group) => group.currency)} />
      </header>
      <div className={styles.estimateContext}>
        <p className={styles.summaryHint}>
          Net saving over {formatInsightsCount(summary.priced_records)} settled request
          {summary.priced_records === 1 ? '' : 's'}: what the same model would have cost without
          prompt caching, minus what the provider cache actually cost.
        </p>
        <details className={styles.estimateDetails}>
          <summary>How the net saving works</summary>
          <p>
            Each request is compared with the same model, all input priced at its standard input
            rate. Cache writes are charged against the saving, so requests whose writes cost more
            than their reads saved lower the total, and the total can be negative. No request is
            clamped at zero. Costs use the prices captured when each request settled.
          </p>
        </details>
      </div>
      {groups.map((group) => (
        <div key={group.currency || 'unavailable'} className={styles.group}>
          {groups.length > 1 ? (
            <h4 className={styles.chartTitle}>
              {insightsCurrencyLabel(group.currency) || 'Price unit'} provider cache
            </h4>
          ) : null}
          <InsightsStatGrid stats={providerCacheStats(group)} />
          {group.net_savings < 0 ? (
            <p className={styles.summaryHint}>
              Cache writes cost more than cache reads saved across these requests.
            </p>
          ) : null}
        </div>
      ))}
      {summary.excluded_records > 0 ? (
        <p className={styles.summaryHint}>
          {formatInsightsCount(summary.excluded_records)} filtered record
          {summary.excluded_records === 1 ? '' : 's'} excluded: not completed, served from the
          Router cache, or without complete provider usage and prices.
        </p>
      ) : null}
      <div className={styles.tokenBreakdownRow}>
        <SpendTable
          title="Net Saving by Decision"
          nameLabel="Decision"
          entries={summary.by_decision}
        />
        <SpendTable
          title="Net Saving by Selected Model"
          nameLabel="Model"
          entries={summary.by_selected_model.map((entry) => ({
            ...entry,
            name: modelText(labels, entry.name),
          }))}
        />
      </div>
    </section>
  )
}
