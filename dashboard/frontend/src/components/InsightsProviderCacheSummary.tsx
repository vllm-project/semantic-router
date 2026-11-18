import type {
  InsightsAggregateProviderCache,
  InsightsProviderCacheCurrencyTotal,
  InsightsProviderCacheSpendEntry,
} from '../pages/insightsPageTypes'
import {
  INSIGHTS_DETAILS,
  InsightsStatGrid,
  InsightsUnitsBadge,
  type InsightsStat,
} from './InsightsStats'
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
  return (
    <div className="flex min-w-0 flex-col gap-2 rounded-lg border border-border bg-surface px-3.5 py-3">
      <h4 className="text-[13px] font-semibold">{title}</h4>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[13px] [&_td]:py-1.5 [&_td]:pl-3 [&_td]:text-right [&_td]:font-mono [&_td]:text-xs [&_td]:tabular [&_th]:py-1.5 [&_th]:text-left [&_th]:font-normal [&_thead_th]:pl-3 [&_thead_th]:text-xs [&_thead_th]:text-subtle-foreground [&_thead_th:not(:first-child)]:text-right [&_tbody_tr]:border-t [&_tbody_tr]:border-border">
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
                  className={entry.net_savings < 0 ? 'text-destructive-foreground' : undefined}
                  style={{
                    backgroundImage: `linear-gradient(to left, ${entry.net_savings < 0 ? 'var(--destructive-soft)' : 'var(--success-soft)'} ${peak > 0 ? (Math.abs(entry.net_savings) / peak) * 100 : 0}%, transparent 0)`,
                  }}
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
    <section
      className="flex flex-col gap-3 rounded-xl border border-border bg-card px-4 py-4 shadow-xs"
      aria-label="Provider prompt cache"
    >
      <header className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold tracking-[-0.01em]">Provider prompt cache</h3>
        <InsightsUnitsBadge currencies={groups.map((group) => group.currency)} />
      </header>
      <div className="flex flex-col gap-1.5 text-xs leading-relaxed text-muted-foreground">
        <p>
          Net saving over {formatInsightsCount(summary.priced_records)} settled request
          {summary.priced_records === 1 ? '' : 's'}: what the same model would have cost without
          prompt caching, minus what the provider cache actually cost.
        </p>
        <details className={INSIGHTS_DETAILS}>
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
        <div key={group.currency || 'unavailable'} className="flex flex-col gap-2">
          {groups.length > 1 ? (
            <h4 className="text-xs font-medium text-muted-foreground">
              {insightsCurrencyLabel(group.currency) || 'Price unit'} provider cache
            </h4>
          ) : null}
          <InsightsStatGrid stats={providerCacheStats(group)} />
          {group.net_savings < 0 ? (
            <p className="text-xs text-warning-foreground">
              Cache writes cost more than cache reads saved across these requests.
            </p>
          ) : null}
        </div>
      ))}
      {summary.excluded_records > 0 ? (
        <p className="text-xs text-muted-foreground">
          {formatInsightsCount(summary.excluded_records)} filtered record
          {summary.excluded_records === 1 ? '' : 's'} excluded: not completed, served from the
          Router cache, or without complete provider usage and prices.
        </p>
      ) : null}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
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
