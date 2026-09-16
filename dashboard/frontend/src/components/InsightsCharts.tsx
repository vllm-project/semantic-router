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
  InsightsAggregateSummary,
  InsightsAggregateCurrencySummary,
  InsightsAggregateTokenEntry,
} from '../pages/insightsPageTypes'
import styles from './InsightsCharts.module.css'
import { formatInsightsCost as formatCurrency } from '../utils/insightsCost'

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

interface PieLabelProps {
  cx: number
  cy: number
  midAngle: number
  outerRadius: number
  percent: number
  name: string
}

const renderCustomLabel = ({ cx, cy, midAngle, outerRadius, percent, name }: PieLabelProps) => {
  const radian = Math.PI / 180
  const radius = outerRadius + 25
  const x = cx + radius * Math.cos(-midAngle * radian)
  const y = cy + radius * Math.sin(-midAngle * radian)

  return (
    <text
      x={x}
      y={y}
      fill="#e4e4e7"
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      style={{ fontSize: '11px', fontWeight: 500 }}
    >
      {`${name}: ${(percent * 100).toFixed(0)}%`}
    </text>
  )
}

const generateBarColors = (count: number): string[] => {
  return Array.from({ length: count }, (_, index) => CHART_COLORS[index % CHART_COLORS.length])
}

const formatTokenCount = (value: number) =>
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value)

const formatCompactTokenCount = (value: number) =>
  new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value)

const formatPercent = (value?: number) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  return `${(value * 100).toFixed(1)}%`
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

const summaryCards = (summary: InsightsAggregateSummary | InsightsAggregateCurrencySummary) => [
  {
    label: 'Estimated Savings',
    value: formatCurrency(summary.total_saved, summary.currency),
    accentClassName: styles.summaryValuePositive,
    cardClassName: '',
  },
  {
    label: 'Estimated Saved %',
    value:
      summary.cost_record_count > 0 && summary.baseline_spend > 0
        ? formatPercent(summary.total_saved / summary.baseline_spend)
        : 'N/A',
    accentClassName: styles.summaryValueHighlight,
    cardClassName: styles.summaryCardHighlight,
  },
  {
    label: 'Estimated Baseline',
    value: formatCurrency(summary.baseline_spend, summary.currency),
    accentClassName: '',
    cardClassName: '',
  },
  {
    label: 'Estimated Model Cost',
    value: formatCurrency(summary.actual_spend, summary.currency),
    accentClassName: styles.summaryValueNeutral,
    cardClassName: '',
  },
]

export default function InsightsCharts({ aggregate }: InsightsChartsProps) {
  const summary = aggregate.summary
  const costGroups = summary.by_currency?.length ? summary.by_currency : [summary]
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

  if (aggregate.record_count === 0) {
    return null
  }

  return (
    <section className={styles.container}>
      <div className={styles.estimateContext}>
        <p className={styles.summaryHint}>Estimates from recorded tokens and configured rates.</p>
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
      {costGroups.map((group) => (
        <div key={group.currency || 'unavailable'}>
          {costGroups.length > 1 ? (
            <h3 className={styles.chartTitle}>{group.currency} estimates</h3>
          ) : null}
          <div className={styles.summaryGrid}>
            {summaryCards(group).map((card) => (
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
        </div>
      ))}
      {summary.excluded_record_count > 0 ? (
        <p className={styles.summaryHint}>
          {summary.excluded_record_count} filtered record
          {summary.excluded_record_count === 1 ? '' : 's'} excluded from estimates. Each row
          explains what was not recorded. Historical rows without captured prices are not repriced.
        </p>
      ) : null}

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
              <Pie
                data={decisionData}
                cx="50%"
                cy="50%"
                labelLine={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1 }}
                label={renderCustomLabel}
                outerRadius={70}
                dataKey="value"
              >
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
              <Pie
                data={signalData}
                cx="50%"
                cy="50%"
                labelLine={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1 }}
                label={renderCustomLabel}
                outerRadius={70}
                dataKey="value"
              >
                {signalData.map((_entry, index) => (
                  <Cell key={`signal-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
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
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }} />
            <YAxis
              tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
              tickFormatter={formatCompactTokenCount}
            />
            <Tooltip
              cursor={false}
              formatter={(value: number | string) => [formatTokenCount(Number(value)), 'Tokens']}
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
            {tokenVolume.excluded_record_count === 1 ? '' : 's'} excluded from token totals because
            usage data is incomplete.
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
    </section>
  )
}
