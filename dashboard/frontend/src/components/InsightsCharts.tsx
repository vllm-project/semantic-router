import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
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

const CHART_COLORS = [
  '#e31b23',
  '#f4f4f5',
  '#c9cbd0',
  '#a1a1aa',
  '#858990',
  '#6f7379',
  '#55585e',
  '#393b40',
]

const generateBarColors = (count: number): string[] => {
  return Array.from({ length: count }, (_, index) => CHART_COLORS[index % CHART_COLORS.length])
}

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

const formatSavingsRate = (saved: number, baseline: number, pricedRequests: number) => {
  if (pricedRequests <= 0 || baseline <= 0) return 'No priced requests'
  return `${((saved / baseline) * 100).toFixed(1)}% saved`
}

const formatAxisLabel = (value: string) => (value.length > 20 ? `${value.slice(0, 17)}...` : value)

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
          <Bar dataKey="total_tokens" name="Total Tokens" fill="#e31b23" radius={[6, 6, 0, 0]} />
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
    indicator: formatSavingsRate(
      aggregate.summary.total_saved,
      aggregate.summary.baseline_spend,
      aggregate.summary.cost_record_count,
    ),
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
  const tokenValues = [
    { name: 'Input Tokens', value: tokenVolume.input_tokens, fill: '#8f949c' },
    { name: 'Output Tokens', value: tokenVolume.output_tokens, fill: '#a6abb3' },
    { name: 'Total Tokens', value: tokenVolume.total_tokens, fill: '#e31b23' },
  ]
  const barColors = generateBarColors(modelData.length)

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
            <div className={styles.chartSection}>
              <h3 className={styles.chartTitle}>
                <svg
                  className={styles.chartIcon}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <rect x="3" y="3" width="7" height="18" />
                  <rect x="14" y="8" width="7" height="13" />
                </svg>
                Model Selection
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={modelData} margin={{ top: 20, right: 0, left: 0, bottom: 60 }}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.1)"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="name"
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                    tickFormatter={formatAxisLabel}
                  />
                  <YAxis tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }} />
                  <Tooltip
                    cursor={false}
                    contentStyle={{
                      background: 'var(--color-bg-secondary)',
                      border: '1px solid var(--color-border)',
                      borderRadius: '4px',
                      color: 'var(--color-text)',
                    }}
                    itemStyle={{ color: 'var(--color-text)' }}
                  />
                  <Bar dataKey="value" name="Count">
                    {modelData.map((_entry, index) => (
                      <Cell key={`model-${index}`} fill={barColors[index]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className={styles.chartSection}>
              <h3 className={styles.chartTitle}>
                <svg
                  className={styles.chartIcon}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 2 L12 12 L20 12" />
                </svg>
                Decision Distribution
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={decisionData} cx="50%" cy="45%" outerRadius={76} dataKey="value">
                    {decisionData.map((_entry, index) => (
                      <Cell
                        key={`decision-${index}`}
                        fill={CHART_COLORS[index % CHART_COLORS.length]}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: 'var(--color-bg-secondary)',
                      border: '1px solid var(--color-border)',
                      borderRadius: '4px',
                      color: 'var(--color-text)',
                    }}
                    itemStyle={{ color: 'var(--color-text)' }}
                  />
                  <Legend
                    verticalAlign="bottom"
                    height={44}
                    formatter={(value) => formatAxisLabel(String(value))}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className={styles.chartSection}>
              <h3 className={styles.chartTitle}>
                <svg
                  className={styles.chartIcon}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 2 L12 12 L20 12" />
                </svg>
                Signal Distribution
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={signalData} cx="50%" cy="45%" outerRadius={76} dataKey="value">
                    {signalData.map((_entry, index) => (
                      <Cell
                        key={`signal-${index}`}
                        fill={CHART_COLORS[index % CHART_COLORS.length]}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: 'var(--color-bg-secondary)',
                      border: '1px solid var(--color-border)',
                      borderRadius: '4px',
                      color: 'var(--color-text)',
                    }}
                    itemStyle={{ color: 'var(--color-text)' }}
                  />
                  <Legend
                    verticalAlign="bottom"
                    height={44}
                    formatter={(value) => formatAxisLabel(String(value))}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

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
              Token Volume
            </h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={tokenValues} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(255,255,255,0.1)"
                  vertical={false}
                />
                <XAxis
                  dataKey="name"
                  tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                />
                <YAxis
                  tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                  tickFormatter={formatCompactTokenCount}
                />
                <Tooltip
                  cursor={false}
                  formatter={(value: number | string) => [
                    formatTokenCount(Number(value)),
                    'Tokens',
                  ]}
                  contentStyle={{
                    background: 'var(--color-bg-secondary)',
                    border: '1px solid var(--color-border)',
                    borderRadius: '4px',
                    color: 'var(--color-text)',
                  }}
                  itemStyle={{ color: 'var(--color-text)' }}
                />
                <Bar dataKey="value" name="Tokens" radius={[8, 8, 0, 0]}>
                  {tokenValues.map((entry) => (
                    <Cell key={entry.name} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            {tokenVolume.excluded_record_count > 0 ? (
              <p className={styles.summaryHint}>
                {tokenVolume.excluded_record_count} filtered record
                {tokenVolume.excluded_record_count === 1 ? '' : 's'} excluded from token totals
                because usage data is incomplete.
              </p>
            ) : null}
          </div>

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
