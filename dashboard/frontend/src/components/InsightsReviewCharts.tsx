import type { ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type {
  InsightsReviewAggregate,
  InsightsReviewCost,
  InsightsReviewTimeBucket,
} from '../pages/insightsReviewTypes'
import { modelText, useModelLabels } from '../pages/insightsModelLabelSupport'
import { ChartLegend, ChartTooltip, chartAxis, chartGrid, useChartReady } from '../ui'
import { formatSeconds } from '../utils/dateTime'
import { formatInsightsCost } from '../utils/insightsCost'

interface InsightsReviewChartsProps {
  reviews: InsightsReviewAggregate
}

// Verdicts in the status colors; failures stay apart from denials.
const RESULT_SERIES = [
  { key: 'allowed', name: 'Allowed', fill: 'var(--success)' },
  { key: 'denied', name: 'Denied', fill: 'var(--destructive)' },
  { key: 'failed', name: 'Failed', fill: 'var(--warning)' },
  { key: 'invalid', name: 'Invalid answer', fill: 'var(--chart-8)' },
  { key: 'cancelled', name: 'Cancelled', fill: 'var(--border-strong)' },
] as const

function formatReviewCost(cost: InsightsReviewCost[] | null | undefined) {
  if (!cost?.length) return 'Not priced'
  return cost.map((entry) => formatInsightsCost(entry.amount, entry.currency)).join(' + ')
}

const plural = (count: number, noun: string) => `${count} ${noun}${count === 1 ? '' : 's'}`

const formatLatency = (value?: number) => (typeof value === 'number' ? formatSeconds(value) : '—')

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
          : '—',
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

function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-3 rounded-xl border border-border bg-card px-4 py-3.5 shadow-xs">
      <h3 className="text-[13px] font-semibold">{title}</h3>
      {children}
    </div>
  )
}

function ReviewsOverTimeChart({
  data,
  bucket,
}: {
  data: InsightsReviewTimeBucket[]
  bucket?: string
}) {
  const ready = useChartReady()
  const rows = data.map((entry) => ({ ...entry, label: bucketLabel(entry.start, bucket) }))
  return (
    <ChartCard title={`Reviews over time${bucket ? ` (per ${bucket})` : ''}`}>
      <ChartLegend
        items={RESULT_SERIES.map((series) => ({ label: series.name, color: series.fill }))}
      />
      <div className="h-[220px]">
        {ready ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
              <CartesianGrid {...chartGrid} />
              <XAxis dataKey="label" {...chartAxis} minTickGap={16} />
              <YAxis {...chartAxis} allowDecimals={false} />
              <Tooltip cursor={{ fill: 'var(--accent)' }} content={<ChartTooltip />} />
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
        ) : null}
      </div>
    </ChartCard>
  )
}

function ReviewModelChart({ reviews }: { reviews: InsightsReviewAggregate }) {
  const ready = useChartReady()
  const labels = useModelLabels()
  const rows = reviews.by_model.map((entry) => ({
    name: modelText(labels, entry.model),
    requests: entry.requests,
    cost: formatReviewCost(entry.cost),
  }))
  return (
    <ChartCard title="Requests by review model">
      <div className="h-[244px]">
        {ready ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
              <CartesianGrid {...chartGrid} />
              <XAxis dataKey="name" {...chartAxis} interval={0} />
              <YAxis {...chartAxis} allowDecimals={false} />
              <Tooltip
                cursor={{ fill: 'var(--accent)' }}
                content={
                  <ChartTooltip
                    labelFormat={(label) =>
                      `${label} · ${rows.find((row) => row.name === label)?.cost ?? 'Not priced'}`
                    }
                  />
                }
              />
              <Bar dataKey="requests" name="Requests" fill="var(--chart-1)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : null}
      </div>
    </ChartCard>
  )
}

export default function InsightsReviewCharts({ reviews }: InsightsReviewChartsProps) {
  return (
    <section className="mb-4 flex flex-col gap-3" aria-label="Review summary">
      <div className="grid grid-cols-2 overflow-hidden rounded-xl border border-border bg-card shadow-xs sm:grid-cols-3 xl:grid-cols-6 [&>article]:border-border xl:[&>article:not(:first-child)]:border-l">
        {summaryTiles(reviews).map((tile) => (
          <article key={tile.label} className="flex min-w-0 flex-col gap-1.5 px-4 py-3">
            <span className="text-xs text-muted-foreground">{tile.label}</span>
            <strong className="truncate text-lg font-semibold leading-none tracking-[-0.02em] tabular">
              {tile.value}
            </strong>
            <span className="truncate text-[11.5px] text-subtle-foreground" title={tile.hint}>
              {tile.hint}
            </span>
          </article>
        ))}
      </div>
      <p className="text-xs text-subtle-foreground">
        Approval rate counts allow and deny verdicts only. A failed, invalid or cancelled review is
        not a deny. Costs are the recorded estimates; requests without a price stay unpriced.
      </p>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <ReviewsOverTimeChart data={reviews.over_time} bucket={reviews.bucket} />
        <ReviewModelChart reviews={reviews} />
      </div>
    </section>
  )
}
