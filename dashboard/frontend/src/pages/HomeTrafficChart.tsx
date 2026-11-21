import { useId, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { RefreshCw, ServerOff } from 'lucide-react'

import {
  Button,
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  ChartLegend,
  ChartTooltip,
  IconTile,
  Segmented,
  Skeleton,
  chartAxis,
  chartGrid,
  useChartReady,
  type Tone,
} from '../ui'
import {
  formatInsightsCompactCount,
  formatInsightsCost,
  formatInsightsCount,
} from '../utils/insightsCost'
import {
  HOME_RANGE_LABEL,
  needsTrafficWindow,
  timeTicks,
  TRAFFIC_UNAVAILABLE,
  type HomeRange,
  type TrafficView,
} from './homeSupport'
import type { HomeTraffic } from './homeTraffic'
import { formatLatency } from './insightsDisplayFormat'
import type { InsightsUnavailableReason } from './insightsPageApi'

type Metric = 'requests' | 'latency' | 'cost'

const METRICS: { value: Metric; label: string }[] = [
  { value: 'requests', label: 'Requests' },
  { value: 'latency', label: 'Latency' },
  { value: 'cost', label: 'Cost' },
]

const HOUR = 3_600_000
const DAY = 24 * HOUR
const timeFormat = new Intl.DateTimeFormat(undefined, {
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
})
const dayFormat = new Intl.DateTimeFormat(undefined, { weekday: 'short', day: 'numeric' })

function formatSpan(start: number, intervalMs: number): string {
  const end = start + intervalMs
  const times = `${timeFormat.format(start)}–${timeFormat.format(end)}`
  return intervalMs >= HOUR ? `${dayFormat.format(start)}, ${times}` : times
}

function TrafficNotice({
  reason,
  error,
  onRetry,
}: {
  reason?: InsightsUnavailableReason
  error?: string | null
  onRetry: () => void
}) {
  const navigate = useNavigate()
  const copy = reason
    ? TRAFFIC_UNAVAILABLE[reason]
    : {
        title: 'Traffic could not load',
        description: error ?? 'The replay API did not answer.',
        tone: 'slate' as Tone,
        icon: ServerOff,
      }
  return (
    <div className="flex h-[292px] flex-col items-center justify-center gap-3 px-6 text-center">
      <IconTile icon={copy.icon} tone={copy.tone} size="lg" />
      <div className="flex max-w-md flex-col gap-1">
        <span className="text-[15px] font-semibold tracking-[-0.01em]">{copy.title}</span>
        <span className="text-[13px] text-muted-foreground">{copy.description}</span>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2">
        {reason === 'replay_off' ? (
          <Button size="sm" onClick={() => navigate('/routing/recipes')}>
            Open recipes
          </Button>
        ) : null}
        <Button size="sm" variant="outline" onClick={onRetry}>
          <RefreshCw />
          Try again
        </Button>
      </div>
    </div>
  )
}

function TrafficPlot({ view, metric }: { view: TrafficView; metric: Metric }) {
  const id = useId().replace(/:/g, '')
  const ready = useChartReady()
  const money = (value: number) => formatInsightsCost(value, view.currency, { total: true })
  const format = (value: number | string) => {
    const number = Number(value)
    if (metric === 'latency') return formatLatency(number)
    if (metric === 'cost') return view.currency ? money(number) : '—'
    return formatInsightsCount(number)
  }
  if (!ready) return <Skeleton className="h-full w-full rounded-lg" />
  const axis = timeTicks(
    view.points[0]?.start ?? 0,
    view.points[view.points.length - 1]?.start ?? 0,
  )
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={view.points} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id={`traffic-primary-${id}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.28} />
            <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0} />
          </linearGradient>
          <linearGradient id={`traffic-gold-${id}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="var(--chart-2)" stopOpacity={0.3} />
            <stop offset="100%" stopColor="var(--chart-2)" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid {...chartGrid} />
        <XAxis
          dataKey="start"
          type="number"
          scale="time"
          domain={['dataMin', 'dataMax']}
          ticks={axis.ticks}
          {...chartAxis}
          tickFormatter={(value: number) =>
            axis.stepMs >= DAY ? dayFormat.format(value) : timeFormat.format(value)
          }
        />
        <YAxis
          {...chartAxis}
          width={56}
          allowDecimals={metric !== 'requests'}
          tickFormatter={(value: number) =>
            metric === 'requests' ? formatInsightsCompactCount(value) : format(value)
          }
        />
        <Tooltip
          cursor={{ stroke: 'var(--border-strong)', strokeDasharray: '3 3' }}
          content={
            <ChartTooltip
              format={format}
              labelFormat={(label) => formatSpan(Number(label), view.intervalMs)}
            />
          }
        />
        {metric === 'requests' ? (
          <>
            <Area
              type="monotone"
              dataKey="requests"
              name="Requests"
              stroke="var(--chart-1)"
              strokeWidth={2}
              fill={`url(#traffic-primary-${id})`}
              animationDuration={900}
            />
            <Area
              type="monotone"
              dataKey="cached"
              name="Served from cache"
              stroke="var(--chart-2)"
              strokeWidth={1.6}
              fill={`url(#traffic-gold-${id})`}
              animationDuration={1100}
            />
          </>
        ) : metric === 'latency' ? (
          <>
            <Area
              type="monotone"
              dataKey="p95"
              name="p95"
              stroke="var(--chart-3)"
              strokeWidth={2}
              fill="transparent"
              connectNulls
              animationDuration={900}
            />
            <Area
              type="monotone"
              dataKey="p50"
              name="p50"
              stroke="var(--chart-1)"
              strokeWidth={2}
              fill={`url(#traffic-primary-${id})`}
              connectNulls
              animationDuration={1100}
            />
          </>
        ) : (
          <>
            {view.hasBaseline ? (
              <Area
                type="monotone"
                dataKey="baseline"
                name="Without routing"
                stroke="var(--chart-8)"
                strokeDasharray="4 4"
                strokeWidth={1.5}
                fill="transparent"
                animationDuration={900}
              />
            ) : null}
            <Area
              type="monotone"
              dataKey="cost"
              name="Spend"
              stroke="var(--chart-2)"
              strokeWidth={2}
              fill={`url(#traffic-gold-${id})`}
              animationDuration={1100}
            />
          </>
        )}
      </AreaChart>
    </ResponsiveContainer>
  )
}

function legendFor(view: TrafficView, metric: Metric) {
  const sum = (pick: (point: TrafficView['points'][number]) => number) =>
    view.points.reduce((total, point) => total + pick(point), 0)
  if (metric === 'requests') {
    const requests = sum((point) => point.requests)
    const cached = sum((point) => point.cached)
    return [
      { label: 'Requests', color: 'var(--chart-1)', value: formatInsightsCount(requests) },
      {
        label: 'Served from cache',
        color: 'var(--chart-2)',
        value: requests > 0 ? `${Math.round((cached / requests) * 100)}%` : '—',
      },
    ]
  }
  if (metric === 'latency') {
    return [
      { label: 'p50', color: 'var(--chart-1)', value: formatLatency(view.latency.p50) },
      { label: 'p95', color: 'var(--chart-3)', value: formatLatency(view.latency.p95) },
    ]
  }
  const money = (value: number) =>
    view.currency ? formatInsightsCost(value, view.currency, { total: true }) : '—'
  return [
    { label: 'Spend', color: 'var(--chart-2)', value: money(sum((point) => point.cost)) },
    ...(view.hasBaseline
      ? [
          {
            label: 'Without routing',
            color: 'var(--chart-8)',
            value: money(sum((point) => point.baseline ?? point.cost)),
          },
        ]
      : []),
  ]
}

/** Requests, latency and spend over the range, from Router Replay. */
export function HomeTrafficChart({
  traffic,
  view,
  range,
}: {
  traffic: HomeTraffic
  view: TrafficView | null
  range: HomeRange
}) {
  const [metric, setMetric] = useState<Metric>('requests')
  const windowFailed = !view && needsTrafficWindow(traffic.aggregate) && traffic.windowError
  const unavailable = Boolean(
    traffic.unavailable || (traffic.error && !traffic.aggregate) || windowFailed,
  )
  // Without a series the chart waits for the newest records, which load beside the aggregate.
  const loading =
    !unavailable && !view && (!traffic.aggregate || needsTrafficWindow(traffic.aggregate))
  const empty = !view || view.points.every((point) => point.requests === 0)
  const rangeLabel = HOME_RANGE_LABEL[range]
  const description = unavailable
    ? 'Requests, latency and spend over time'
    : traffic.error
      ? 'Showing the last loaded traffic: the latest refresh failed'
      : view && !view.coversRange
        ? `Last ${formatInsightsCount(view.sampled ?? 0)} requests`
        : `${rangeLabel.charAt(0).toUpperCase()}${rangeLabel.slice(1)} across every public model`

  return (
    <Card className="min-w-0 xl:col-span-2" data-testid="home-traffic">
      <CardHeader>
        <CardTitle>Traffic</CardTitle>
        <CardDescription>{description}</CardDescription>
        {unavailable ? null : (
          <CardAction>
            <Segmented
              size="sm"
              aria-label="Traffic metric"
              value={metric}
              onValueChange={setMetric}
              options={METRICS}
            />
          </CardAction>
        )}
      </CardHeader>
      <CardContent className="pt-2">
        {unavailable ? (
          <TrafficNotice
            reason={traffic.unavailable}
            error={windowFailed ? traffic.windowError : traffic.error}
            onRetry={traffic.refresh}
          />
        ) : (
          <>
            {view ? (
              <ChartLegend className="mb-2" items={legendFor(view, metric)} />
            ) : loading ? (
              <Skeleton className="mb-2 h-4 w-48" />
            ) : (
              <div aria-hidden className="mb-2 h-4" />
            )}
            <div className="relative h-[260px]">
              {loading ? (
                <Skeleton className="h-full w-full rounded-lg" />
              ) : view ? (
                <TrafficPlot view={view} metric={metric} />
              ) : null}
              {!loading && empty ? (
                <div className="pointer-events-none absolute inset-0 grid place-items-center">
                  <span className="rounded-md bg-card/90 px-3 py-1.5 text-[13px] text-muted-foreground">
                    No requests in the {rangeLabel}
                  </span>
                </div>
              ) : null}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
