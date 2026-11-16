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
import InsightsCostCoverage from './InsightsCostCoverage'
import InsightsProviderCacheSummary from './InsightsProviderCacheSummary'
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

// Router calls (a Looper's model calls and the questions the Router asks to
// route) are a separate cost, and the total adds them and the priced requests
// without a baseline, failed ones included; each card shows only where it
// differs from the model cost.
const spendCards = (summary: InsightsAggregateSummary | InsightsAggregateCurrencySummary) => [
  ...((summary.router_call_spend ?? 0) !== 0
    ? [
        {
          label: 'Estimated Router Calls',
          value: formatCurrency(summary.router_call_spend, summary.currency),
          accentClassName: styles.summaryValueNeutral,
          cardClassName: '',
        },
      ]
    : []),
  ...(summary.total_spend !== undefined && summary.total_spend !== summary.actual_spend
    ? [
        {
          label: 'Estimated Total Cost',
          value: formatCurrency(summary.total_spend, summary.currency),
          accentClassName: styles.summaryValueNeutral,
          cardClassName: '',
        },
      ]
    : []),
]

// A currency can have priced costs but no request compared with a baseline;
// its savings are then unknown, not zero.
const comparedCost = (
  summary: InsightsAggregateSummary | InsightsAggregateCurrencySummary,
  value: number,
) => (summary.cost_record_count > 0 ? formatCurrency(value, summary.currency) : 'N/A')

// Savings are net of the compared requests' Router calls; where those cost
// anything, the gross figure and what was netted show beside it.
const netCards = (summary: InsightsAggregateSummary | InsightsAggregateCurrencySummary) =>
  summary.cost_record_count > 0 && (summary.netted_router_call_spend ?? 0) !== 0
    ? [
        {
          label: 'Estimated Gross Savings',
          value: formatCurrency(summary.gross_saved, summary.currency),
          accentClassName: styles.summaryValueNeutral,
          cardClassName: '',
        },
        {
          label: 'Router Calls Netted',
          value: formatCurrency(summary.netted_router_call_spend, summary.currency),
          accentClassName: styles.summaryValueNeutral,
          cardClassName: '',
        },
      ]
    : []

const summaryCards = (summary: InsightsAggregateSummary | InsightsAggregateCurrencySummary) => [
  {
    label: 'Estimated Savings',
    value: comparedCost(summary, summary.total_saved),
    accentClassName:
      summary.total_saved < 0 ? styles.summaryValueNegative : styles.summaryValuePositive,
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
    value: comparedCost(summary, summary.baseline_spend),
    accentClassName: '',
    cardClassName: '',
  },
  {
    label: 'Estimated Model Cost',
    value: comparedCost(summary, summary.actual_spend),
    accentClassName: styles.summaryValueNeutral,
    cardClassName: '',
  },
  ...netCards(summary),
  ...spendCards(summary),
]

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
      <InsightsEstimateNotes />
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
          {summary.excluded_record_count === 1 ? '' : 's'} excluded from savings estimates. Each row
          explains what was not recorded. Historical rows without captured prices are not repriced.
        </p>
      ) : null}
      <InsightsCostCoverage summary={summary} />

      <InsightsProviderCacheSummary summary={aggregate.provider_cache} />

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
