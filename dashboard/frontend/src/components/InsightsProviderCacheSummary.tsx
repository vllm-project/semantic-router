import type {
  InsightsAggregateProviderCache,
  InsightsProviderCacheCurrencyTotal,
  InsightsProviderCacheSpendEntry,
} from '../pages/insightsPageTypes'
import styles from './InsightsCharts.module.css'
import { formatInsightsCost as formatCurrency } from '../utils/insightsCost'

interface InsightsProviderCacheSummaryProps {
  summary?: InsightsAggregateProviderCache
}

const netClassName = (value: number) =>
  value < 0 ? styles.summaryValueNegative : styles.summaryValuePositive

const formatNetShare = (group: InsightsProviderCacheCurrencyTotal) =>
  group.no_cache_spend > 0 ? `${((group.net_savings / group.no_cache_spend) * 100).toFixed(1)}%` : 'N/A'

const providerCacheCards = (group: InsightsProviderCacheCurrencyTotal) => [
  {
    label: 'Net Cache Saving',
    value: formatCurrency(group.net_savings, group.currency),
    accentClassName: netClassName(group.net_savings),
    cardClassName: '',
  },
  {
    label: 'Net Saved %',
    value: formatNetShare(group),
    accentClassName: netClassName(group.net_savings),
    cardClassName: styles.summaryCardHighlight,
  },
  {
    label: 'No-Cache Estimate',
    value: formatCurrency(group.no_cache_spend, group.currency),
    accentClassName: '',
    cardClassName: '',
  },
  {
    label: 'Provider Cost',
    value: formatCurrency(group.actual_spend, group.currency),
    accentClassName: styles.summaryValueNeutral,
    cardClassName: '',
  },
]

interface SpendTableProps {
  title: string
  nameLabel: string
  entries: InsightsProviderCacheSpendEntry[]
}

function SpendTable({ title, nameLabel, entries }: SpendTableProps) {
  return (
    <div className={styles.chartSection}>
      <h3 className={styles.chartTitle}>{title}</h3>
      <table className={styles.spendTable}>
        <thead>
          <tr>
            <th scope="col">{nameLabel}</th>
            <th scope="col">Requests</th>
            <th scope="col">No-cache</th>
            <th scope="col">Provider cost</th>
            <th scope="col">Net saving</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr key={`${entry.name}\u0000${entry.currency}`}>
              <th scope="row">{entry.name}</th>
              <td>{entry.priced_records.toLocaleString('en-US')}</td>
              <td>{formatCurrency(entry.no_cache_spend, entry.currency)}</td>
              <td>{formatCurrency(entry.actual_spend, entry.currency)}</td>
              <td className={entry.net_savings < 0 ? styles.spendNegative : undefined}>
                {formatCurrency(entry.net_savings, entry.currency)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function InsightsProviderCacheSummary({ summary }: InsightsProviderCacheSummaryProps) {
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
    <section className={styles.providerCacheSection} aria-label="Provider prompt cache">
      <h3 className={styles.chartTitle}>Provider Prompt Cache</h3>
      <div className={styles.estimateContext}>
        <p className={styles.summaryHint}>
          Net saving over {summary.priced_records.toLocaleString('en-US')} settled request
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
        <div key={group.currency || 'unavailable'}>
          {groups.length > 1 ? (
            <h3 className={styles.chartTitle}>{group.currency} provider cache</h3>
          ) : null}
          <div className={styles.summaryGrid}>
            {providerCacheCards(group).map((card) => (
              <article
                key={card.label}
                className={`${styles.summaryCard} ${card.cardClassName}`.trim()}
              >
                <span className={styles.summaryLabel}>{card.label}</span>
                <strong className={`${styles.summaryValue} ${card.accentClassName}`.trim()}>
                  {card.value}
                </strong>
              </article>
            ))}
          </div>
          {group.net_savings < 0 ? (
            <p className={styles.summaryHint}>
              Cache writes cost more than cache reads saved across these requests.
            </p>
          ) : null}
        </div>
      ))}
      {summary.excluded_records > 0 ? (
        <p className={styles.summaryHint}>
          {summary.excluded_records.toLocaleString('en-US')} filtered record
          {summary.excluded_records === 1 ? '' : 's'} excluded: not completed, served from the
          Router cache, or without complete provider usage and prices.
        </p>
      ) : null}
      <div className={styles.tokenBreakdownRow}>
        <SpendTable title="Net Saving by Decision" nameLabel="Decision" entries={summary.by_decision} />
        <SpendTable
          title="Net Saving by Selected Model"
          nameLabel="Model"
          entries={summary.by_selected_model}
        />
      </div>
    </section>
  )
}
