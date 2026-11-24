import { useId, useMemo, type ReactNode } from 'react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { LineChart as LineChartIcon, OctagonAlert, type LucideIcon } from 'lucide-react'

import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
  ChartLegend,
  ChartTooltip,
  Skeleton,
  chartAxis,
  chartGrid,
  cn,
  useChartReady,
} from '../ui'
import { formatMoment, formatTimeTick } from './metricsFormat'
import { chartRows, hasSamples, timeTicks } from './metricsSeries'

export interface ChartSeries {
  label: string
  color: string
  values: ReadonlyArray<number | null>
  format: (value: number) => string
  /** Areas of one chart stack when the chart is stacked; lines never do. */
  shape?: 'line' | 'area'
  axis?: 'left' | 'right'
  dashed?: boolean
}

type Domain = [number, number | 'auto']

/** A sample with no neighbours draws no line segment, so it gets a dot instead. */
function loneSampleDot(values: ReadonlyArray<number | null>, color: string) {
  return (props: { key?: string; cx?: number; cy?: number; index?: number }) => {
    const index = props.index ?? -1
    const lone = values[index] != null && values[index - 1] == null && values[index + 1] == null
    return lone && props.cx !== undefined && props.cy !== undefined ? (
      <circle key={props.key} cx={props.cx} cy={props.cy} r={2.5} fill={color} />
    ) : (
      <g key={props.key} />
    )
  }
}

export function MetricsTimeChart({
  times,
  series,
  axisFormat,
  rightAxisFormat,
  domain = [0, 'auto'],
  rightDomain = [0, 'auto'],
  stacked = false,
}: {
  times: readonly number[]
  series: readonly ChartSeries[]
  axisFormat: (value: number) => string
  rightAxisFormat?: (value: number) => string
  domain?: Domain
  rightDomain?: Domain
  stacked?: boolean
}) {
  const id = useId().replace(/:/g, '')
  const ready = useChartReady()
  const keyed = useMemo(
    () => series.map((entry, index) => ({ ...entry, key: `s${index}` })),
    [series],
  )
  const rows = useMemo(() => chartRows(times, keyed), [times, keyed])
  const start = times[0] ?? 0
  const end = times[times.length - 1] ?? 0
  const ticks = useMemo(() => timeTicks(start, end), [start, end])
  const right = keyed.some((entry) => entry.axis === 'right')
  if (!ready) return <Skeleton className="size-full rounded-lg" />
  return (
    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
      <ComposedChart
        data={rows}
        margin={{ top: 8, right: right ? 0 : 12, left: -4, bottom: 0 }}
        accessibilityLayer
      >
        <defs>
          {keyed.map((entry) =>
            entry.shape === 'area' ? (
              <linearGradient key={entry.key} id={`${id}-${entry.key}`} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor={entry.color} stopOpacity={0.24} />
                <stop offset="100%" stopColor={entry.color} stopOpacity={0.02} />
              </linearGradient>
            ) : null,
          )}
        </defs>
        <CartesianGrid {...chartGrid} />
        <XAxis
          dataKey="time"
          type="number"
          scale="time"
          domain={[start, end]}
          ticks={ticks}
          tickFormatter={(value: number) => formatTimeTick(value, end - start)}
          tickMargin={10}
          {...chartAxis}
        />
        <YAxis
          yAxisId="left"
          width={58}
          domain={domain}
          tickFormatter={axisFormat}
          {...chartAxis}
        />
        {right ? (
          <YAxis
            yAxisId="right"
            orientation="right"
            width={46}
            domain={rightDomain}
            tickFormatter={rightAxisFormat}
            {...chartAxis}
          />
        ) : null}
        <Tooltip
          filterNull={false}
          cursor={{ stroke: 'var(--border-strong)', strokeDasharray: '3 3' }}
          content={
            <ChartTooltip
              labelFormat={(label) => formatMoment(Number(label))}
              format={(value, name) => {
                const entry = keyed.find((item) => item.label === name)
                return entry ? entry.format(Number(value)) : String(value)
              }}
            />
          }
        />
        {keyed.map((entry) =>
          entry.shape === 'area' ? (
            <Area
              key={entry.key}
              yAxisId={entry.axis ?? 'left'}
              type="linear"
              dataKey={entry.key}
              name={entry.label}
              stroke={entry.color}
              strokeWidth={1.75}
              fill={`url(#${id}-${entry.key})`}
              stackId={stacked ? 'stack' : undefined}
              dot={loneSampleDot(entry.values, entry.color)}
              activeDot={{ r: 3.5, strokeWidth: 2 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          ) : (
            <Line
              key={entry.key}
              yAxisId={entry.axis ?? 'left'}
              type="linear"
              dataKey={entry.key}
              name={entry.label}
              stroke={entry.color}
              strokeWidth={1.75}
              strokeDasharray={entry.dashed ? '4 4' : undefined}
              dot={loneSampleDot(entry.values, entry.color)}
              activeDot={{ r: 3.5, strokeWidth: 2 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          ),
        )}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function ChartNotice({
  icon: Icon,
  title,
  children,
  tone = 'neutral',
}: {
  icon: LucideIcon
  title: string
  children?: ReactNode
  tone?: 'neutral' | 'danger'
}) {
  return (
    <div className="flex size-full flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed border-border px-6 text-center">
      <Icon
        className={cn('size-5', tone === 'danger' ? 'text-destructive' : 'text-subtle-foreground')}
        aria-hidden="true"
      />
      <span className="text-[13px] font-medium">{title}</span>
      {children ? (
        <span className="line-clamp-3 max-w-md break-words text-xs text-muted-foreground">
          {children}
        </span>
      ) : null}
    </div>
  )
}

/**
 * A chart card whose plot shows, in order: a skeleton while loading, the
 * query's error, an empty notice without samples, or the chart.
 */
export function MetricsChartCard({
  title,
  description,
  toolbar,
  legend,
  loading = false,
  error,
  empty = false,
  emptyHint = 'Prometheus has no Router samples for this range yet.',
  className,
  plotClassName = 'h-[220px]',
  footer,
  children,
}: {
  title: string
  description: string
  toolbar?: ReactNode
  legend?: { label: string; color: string; value?: string }[]
  loading?: boolean
  error?: string | null
  empty?: boolean
  emptyHint?: string
  className?: string
  plotClassName?: string
  footer?: ReactNode
  children: ReactNode
}) {
  const legendItems = legend?.length && !loading && !error && !empty ? legend : null
  return (
    <Card className={cn('min-w-0 overflow-hidden', className)}>
      <section aria-label={title} className="flex h-full flex-col">
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription className="text-xs">{description}</CardDescription>
        </CardHeader>
        {toolbar || legendItems ? (
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-5 pt-3">
            {toolbar}
            {legendItems ? <ChartLegend items={legendItems} /> : null}
          </div>
        ) : null}
        <div className={cn('mt-auto px-3 pb-3 pt-2', plotClassName)}>
          {loading ? (
            <Skeleton className="size-full rounded-lg" />
          ) : error ? (
            <ChartNotice icon={OctagonAlert} title="This query failed" tone="danger">
              {error}
            </ChartNotice>
          ) : empty ? (
            <ChartNotice icon={LineChartIcon} title="No samples in this range">
              {emptyHint}
            </ChartNotice>
          ) : (
            children
          )}
        </div>
        {footer}
      </section>
    </Card>
  )
}

/** A summary figure over the range, with its series as a sparkline. */
export function MetricsKpi({
  label,
  color,
  value,
  hint,
  help,
  spark,
}: {
  label: string
  color: string
  value: string
  hint: string
  help: string
  spark?: ReadonlyArray<number | null>
}) {
  const id = useId().replace(/:/g, '')
  const ready = useChartReady()
  const rows = useMemo(() => (spark ?? []).map((v, index) => ({ index, v })), [spark])
  const known = value !== '—'
  return (
    <div className="flex h-full min-w-0 flex-col overflow-hidden rounded-xl border border-border bg-card shadow-xs">
      <dt
        className="flex items-center gap-2 px-4 pt-3.5 text-[13px] font-medium text-muted-foreground"
        title={help}
      >
        <span className="size-2 shrink-0 rounded-[3px]" style={{ background: color }} />
        {label}
      </dt>
      <dd
        className={cn(
          'px-4 pt-1.5 tabular',
          known
            ? 'text-[24px] font-semibold leading-7 tracking-[-0.03em]'
            : 'text-[24px] font-medium leading-7 text-subtle-foreground',
        )}
      >
        {value}
      </dd>
      <dd className="truncate px-4 pt-1 text-xs text-muted-foreground" title={hint}>
        {hint}
      </dd>
      <dd className="mt-auto h-10 pt-2" aria-hidden="true">
        {ready && spark && hasSamples(spark) ? (
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <AreaChart data={rows} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id={`${id}-spark`} x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.22} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <YAxis hide domain={[0, 'auto']} />
              <Area
                type="linear"
                dataKey="v"
                stroke={color}
                strokeWidth={1.5}
                fill={`url(#${id}-spark)`}
                connectNulls={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : null}
      </dd>
    </div>
  )
}
