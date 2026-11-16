import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type {
  InsightsReviewAggregate,
  InsightsReviewCost,
  InsightsReviewTimeBucket,
} from '../pages/insightsReviewTypes'
import { formatSeconds } from '../utils/dateTime'
import { formatInsightsCost } from '../utils/insightsCost'
import styles from './InsightsCharts.module.css'

interface InsightsReviewChartsProps {
  reviews: InsightsReviewAggregate
}

// Verdicts in the dashboard's red and grays; failures stay apart from denials.
const RESULT_SERIES = [
  { key: 'allowed', name: 'Allowed', fill: '#d4d4d8' },
  { key: 'denied', name: 'Denied', fill: '#e31b23' },
  { key: 'failed', name: 'Failed', fill: '#8f949c' },
  { key: 'invalid', name: 'Invalid answer', fill: '#6f7379' },
  { key: 'cancelled', name: 'Cancelled', fill: '#393b40' },
] as const

const TOOLTIP_STYLE = {
  background: 'var(--color-bg-secondary)',
  border: '1px solid var(--color-border)',
  borderRadius: '4px',
  color: 'var(--color-text)',
}
const AXIS_TICK = { fill: 'var(--color-text-secondary)', fontSize: 11 }

function formatReviewCost(cost: InsightsReviewCost[] | null | undefined) {
  if (!cost?.length) return 'Not priced'
  return cost.map((entry) => formatInsightsCost(entry.amount, entry.currency)).join(' + ')
}

const plural = (count: number, noun: string) => `${count} ${noun}${count === 1 ? '' : 's'}`

const formatLatency = (value?: number) => (typeof value === 'number' ? formatSeconds(value) : 'N/A')

function bucketLabel(start: string, bucket?: string) {
  const date = new Date(start)
  if (Number.isNaN(date.getTime())) return start
  if (bucket === 'day') return date.toLocaleDateString()
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function summaryTiles(reviews: InsightsReviewAggregate) {
  const verdicts = reviews.allowed + reviews.denied
  const requests = reviews.reviews + reviews.tool_calls + reviews.pending
  return [
    {
      label: 'Reviews',
      value: String(reviews.reviews),
      hint: `${plural(reviews.tool_calls, 'tool check')} · ${reviews.pending} in progress`,
    },
    {
      label: 'Approval rate',
      value:
        typeof reviews.approval_rate === 'number'
          ? `${(reviews.approval_rate * 100).toFixed(1)}%`
          : 'N/A',
      hint: `${reviews.allowed} allowed of ${verdicts} verdicts`,
    },
    { label: 'Denied', value: String(reviews.denied), hint: 'Valid deny verdicts' },
    {
      label: 'Failed',
      value: String(reviews.failed),
      hint: `${reviews.invalid} invalid · ${reviews.cancelled} cancelled`,
    },
    {
      label: 'Latency p50',
      value: formatLatency(reviews.latency_p50_ms),
      hint: `p95 ${formatLatency(reviews.latency_p95_ms)} · reviews with a verdict`,
    },
    {
      label: 'Cost',
      value: formatReviewCost(reviews.cost),
      hint: `${requests - reviews.unpriced_requests} of ${requests} requests priced`,
    },
  ]
}

function ReviewsOverTimeChart({
  data,
  bucket,
}: {
  data: InsightsReviewTimeBucket[]
  bucket?: string
}) {
  const rows = data.map((entry) => ({ ...entry, label: bucketLabel(entry.start, bucket) }))
  return (
    <div className={styles.chartSection}>
      <h3 className={styles.chartTitle}>Reviews over time{bucket ? ` (per ${bucket})` : ''}</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
          <XAxis dataKey="label" tick={AXIS_TICK} minTickGap={16} />
          <YAxis tick={AXIS_TICK} allowDecimals={false} />
          <Tooltip
            cursor={false}
            contentStyle={TOOLTIP_STYLE}
            itemStyle={{ color: 'var(--color-text)' }}
          />
          <Legend verticalAlign="top" height={30} />
          {RESULT_SERIES.map((series) => (
            <Bar
              key={series.key}
              dataKey={series.key}
              name={series.name}
              stackId="result"
              fill={series.fill}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function ReviewModelChart({ reviews }: { reviews: InsightsReviewAggregate }) {
  const rows = reviews.by_model.map((entry) => ({
    name: entry.model,
    requests: entry.requests,
    cost: formatReviewCost(entry.cost),
  }))
  return (
    <div className={styles.chartSection}>
      <h3 className={styles.chartTitle}>Requests by review model</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
          <XAxis dataKey="name" tick={AXIS_TICK} interval={0} />
          <YAxis tick={AXIS_TICK} allowDecimals={false} />
          <Tooltip
            cursor={false}
            contentStyle={TOOLTIP_STYLE}
            itemStyle={{ color: 'var(--color-text)' }}
            formatter={(value: number | string, _name, item) => [
              `${value} (${(item?.payload as { cost?: string } | undefined)?.cost ?? 'Not priced'})`,
              'Requests',
            ]}
          />
          <Bar dataKey="requests" name="Requests" fill="#a6abb3" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function InsightsReviewCharts({ reviews }: InsightsReviewChartsProps) {
  return (
    <section className={styles.container} aria-label="Review summary">
      <div className={`${styles.summaryGrid} ${styles.reviewSummaryGrid}`}>
        {summaryTiles(reviews).map((tile) => (
          <article key={tile.label} className={styles.summaryCard}>
            <span className={styles.summaryLabel}>{tile.label}</span>
            <strong className={styles.summaryValue}>{tile.value}</strong>
            <span className={styles.summaryHint}>{tile.hint}</span>
          </article>
        ))}
      </div>
      <p className={styles.summaryHint}>
        Approval rate counts allow and deny verdicts only. A failed, invalid or cancelled review is
        not a deny. Costs are the recorded estimates; requests without a price stay unpriced.
      </p>
      <div className={styles.tokenBreakdownRow}>
        <ReviewsOverTimeChart data={reviews.over_time} bucket={reviews.bucket} />
        <ReviewModelChart reviews={reviews} />
      </div>
    </section>
  )
}
