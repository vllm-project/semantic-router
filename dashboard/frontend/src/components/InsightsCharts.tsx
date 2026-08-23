import {
  BarChart,
  Bar,
  Legend,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

import type {
  InsightsAggregateResponse,
  InsightsAggregateTokenEntry,
} from '../pages/insightsPageTypes'
import styles from './InsightsCharts.module.css'

interface InsightsChartsProps {
  aggregate: InsightsAggregateResponse
}

const CHART_PRIMARY = '#e31b23'

const formatCurrency = (value: number, currency?: string) => {
  if (!currency) {
    return 'N/A'
  }

  try {
    const minimumFractionDigits = Math.abs(value) >= 0.01 ? 2 : 4
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      minimumFractionDigits,
      maximumFractionDigits: 4,
    }).format(value)
  } catch {
    return `${value.toFixed(4)} ${currency}`
  }
}

const formatTokenCount = (value: number) =>
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value)

const formatCompactTokenCount = (value: number) =>
  new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value)

const formatSavingsRate = (saved: number, baseline: number) => {
  if (baseline <= 0) return '0.0% saved'
  return `${((saved / baseline) * 100).toFixed(1)}% saved`
}

const formatAxisLabel = (value: string) => (value.length > 20 ? `${value.slice(0, 17)}...` : value)

interface DistributionChartProps {
  title: string
  data: Array<{ name: string; value: number }>
  featured?: boolean
}

function DistributionChart({ title, data, featured = false }: DistributionChartProps) {
  const visibleData = data.slice(0, 10)
  const height = Math.max(190, Math.min(360, visibleData.length * 34 + 52))

  return (
    <div className={`${styles.chartSection} ${featured ? styles.chartFeatured : ''}`}>
      <h3 className={styles.chartTitle}>{title}</h3>
      {visibleData.length === 0 ? (
        <div className={styles.chartEmpty}>No data in this view.</div>
      ) : (
        <ResponsiveContainer width="100%" height={height}>
          <BarChart
            data={visibleData}
            layout="vertical"
            margin={{ top: 8, right: 24, left: featured ? 44 : 20, bottom: 8 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.08)"
              horizontal={false}
            />
            <XAxis
              type="number"
              allowDecimals={false}
              tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
              tickFormatter={formatCompactTokenCount}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={featured ? 170 : 125}
              tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
              tickFormatter={formatAxisLabel}
            />
            <Tooltip
              cursor={{ fill: 'rgba(255,255,255,0.025)' }}
              formatter={(value: number | string) => [formatTokenCount(Number(value)), 'Requests']}
              contentStyle={{
                background: 'var(--color-bg-secondary)',
                border: '1px solid var(--color-border)',
                borderRadius: '8px',
                color: 'var(--color-text)',
              }}
              itemStyle={{ color: 'var(--color-text)' }}
            />
            <Bar dataKey="value" name="Requests" fill={CHART_PRIMARY} radius={[0, 6, 6, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

interface TokenBreakdownChartProps {
  title: string
  data: InsightsAggregateTokenEntry[]
}

function TokenBreakdownChart({ title, data }: TokenBreakdownChartProps) {
  return (
    <div className={styles.chartSection}>
      <h3 className={styles.chartTitle}>
        <svg
          className={styles.chartIcon}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M4 19h16" />
          <path d="M7 16V8" />
          <path d="M12 16V5" />
          <path d="M17 16v-6" />
        </svg>
        {title}
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 20, right: 20, left: 0, bottom: 70 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
          <XAxis
            dataKey="name"
            angle={-30}
            textAnchor="end"
            height={90}
            interval={0}
            tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
            tickFormatter={formatAxisLabel}
          />
          <YAxis
            tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
            tickFormatter={formatCompactTokenCount}
          />
          <Tooltip
            cursor={false}
            formatter={(value: number | string) => [formatTokenCount(Number(value)), 'Tokens']}
            labelFormatter={(label) => String(label)}
            contentStyle={{
              background: 'var(--color-bg-secondary)',
              border: '1px solid var(--color-border)',
              borderRadius: '4px',
              color: 'var(--color-text)',
            }}
            itemStyle={{ color: 'var(--color-text)' }}
          />
          <Legend verticalAlign="top" height={30} />
          <Bar dataKey="input_tokens" name="Input Tokens" fill="#8f949c" radius={[6, 6, 0, 0]} />
          <Bar dataKey="output_tokens" name="Output Tokens" fill="#a6abb3" radius={[6, 6, 0, 0]} />
          <Bar
            dataKey="total_tokens"
            name="Total Tokens"
            fill={CHART_PRIMARY}
            radius={[6, 6, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

const summaryCards = (aggregate: InsightsAggregateResponse) => [
  {
    label: 'Requests',
    value: formatCompactTokenCount(aggregate.record_count),
    accentClassName: '',
    cardClassName: '',
  },
  {
    label: 'Tokens',
    value: formatCompactTokenCount(aggregate.token_volume.total_tokens),
    accentClassName: styles.summaryValueHighlight,
    cardClassName: styles.summaryCardHighlight,
  },
  {
    label: 'Models Used',
    value: formatCompactTokenCount(aggregate.model_selection.length),
    accentClassName: '',
    cardClassName: '',
  },
  {
    label: 'Total Saved',
    value: formatCurrency(aggregate.summary.total_saved, aggregate.summary.currency),
    indicator: formatSavingsRate(aggregate.summary.total_saved, aggregate.summary.baseline_spend),
    detail:
      aggregate.summary.cost_record_count > 0
        ? `${formatCurrency(aggregate.summary.actual_spend, aggregate.summary.currency)} actual · ${formatCurrency(aggregate.summary.baseline_spend, aggregate.summary.currency)} baseline`
        : undefined,
    accentClassName: styles.summaryValuePositive,
    cardClassName: styles.summaryCardSavings,
  },
]

export default function InsightsCharts({ aggregate }: InsightsChartsProps) {
  const summary = aggregate.summary
  const modelData = aggregate.model_selection
  const decisionData = aggregate.decision_distribution
  const signalData = aggregate.signal_distribution
  const tokenVolume = aggregate.token_volume
  const tokenBreakdown = aggregate.token_breakdown

  return (
    <section className={styles.container}>
      <div className={styles.summaryGrid}>
        {summaryCards(aggregate).map((card) => (
          <article
            key={card.label}
            className={`${styles.summaryCard} ${card.cardClassName}`.trim()}
          >
            <span className={styles.summaryLabel}>{card.label}</span>
            <div className={styles.summaryValueRow}>
              <strong className={`${styles.summaryValue} ${card.accentClassName}`.trim()}>
                {card.value}
              </strong>
              {'indicator' in card && card.indicator ? (
                <span className={styles.summaryIndicator}>{card.indicator}</span>
              ) : null}
            </div>
            {'detail' in card && card.detail ? (
              <small className={styles.summaryDetail}>{card.detail}</small>
            ) : null}
          </article>
        ))}
      </div>

      {summary.excluded_record_count > 0 ? (
        <p className={styles.summaryHint}>
          {summary.excluded_record_count} filtered record
          {summary.excluded_record_count === 1 ? '' : 's'} excluded from cost totals because usage
          or pricing data is incomplete.
        </p>
      ) : null}

      {aggregate.record_count === 0 ? (
        <div className={styles.emptyOverview}>
          <strong>No request data yet</strong>
          <span>Your first routed request will bring this overview to life.</span>
        </div>
      ) : (
        <>
          <div className={styles.chartsRow}>
            <DistributionChart title="Model Selection" data={modelData} featured />
            <DistributionChart title="Decision Mix" data={decisionData} />
            <DistributionChart title="Signal Mix" data={signalData} />
          </div>

          {tokenVolume.excluded_record_count > 0 ? (
            <p className={styles.summaryHint}>
              {tokenVolume.excluded_record_count} filtered record
              {tokenVolume.excluded_record_count === 1 ? '' : 's'} excluded from token totals
              because usage data is incomplete.
            </p>
          ) : null}

          <div className={styles.tokenBreakdownRow}>
            <TokenBreakdownChart title="Tokens by Decision" data={tokenBreakdown.by_decision} />
            <TokenBreakdownChart
              title="Tokens by Selected Model"
              data={tokenBreakdown.by_selected_model}
            />
          </div>
        </>
      )}
    </section>
  )
}
