import { useId } from 'react'
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
import { LineChart as LineChartIcon } from 'lucide-react'
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
  ChartLegend,
  ChartTooltip,
  chartAxis,
  chartGrid,
  cn,
} from '../ui'
import {
  DECISION_MODEL_METRICS,
  formatDecisionModelMetric,
  formatDecisionModelRateAxis,
  type DecisionModelChartPoint,
  type DecisionModelMetricKey,
} from './decisionModelMetrics'

const clockTime = (time: number) =>
  new Date(time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
const metricByKey = (key: DecisionModelMetricKey) =>
  DECISION_MODEL_METRICS.find((metric) => metric.key === key)!

export function DecisionModelMetricCards({ points }: { points: DecisionModelChartPoint[] }) {
  const gradientId = useId().replace(/:/g, '')
  const current = points[points.length - 1]
  return (
    <dl
      className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6"
      aria-label="Model statistics"
    >
      {DECISION_MODEL_METRICS.map((metric) => {
        const value = current?.[metric.key]
        const hasHistory = points.some((point) => point[metric.key] != null)
        return (
          <div
            key={metric.key}
            className="flex flex-col overflow-hidden rounded-xl border border-border bg-card shadow-xs"
          >
            <div className="flex flex-col gap-2 px-4 pb-1 pt-3.5">
              <dt
                className="flex items-center gap-2 text-[13px] font-medium text-muted-foreground"
                title={metric.help}
              >
                <span
                  className="size-2 shrink-0 rounded-[3px]"
                  style={{ background: metric.color }}
                />
                {metric.label}
              </dt>
              <dd
                className={cn(
                  'tabular',
                  value == null
                    ? 'text-[15px] font-medium leading-6 text-subtle-foreground'
                    : 'text-[24px] font-semibold leading-6 tracking-[-0.03em]',
                )}
              >
                {formatDecisionModelMetric(value, metric.unit)}
              </dd>
            </div>
            <div className="h-10" aria-hidden="true">
              {hasHistory ? (
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <AreaChart data={points} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
                    <defs>
                      <linearGradient
                        id={`${gradientId}-${metric.key}`}
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop offset="0%" stopColor={metric.color} stopOpacity={0.22} />
                        <stop offset="100%" stopColor={metric.color} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <YAxis hide domain={['auto', 'auto']} />
                    <Area
                      type="linear"
                      dataKey={metric.key}
                      stroke={metric.color}
                      strokeWidth={1.5}
                      fill={`url(#${gradientId}-${metric.key})`}
                      connectNulls={false}
                      isAnimationActive={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <span className="block px-4 pt-1 text-[11px] text-subtle-foreground">
                  No samples in this window
                </span>
              )}
            </div>
          </div>
        )
      })}
    </dl>
  )
}

const chartSections = [
  {
    id: 'traffic',
    title: 'Traffic & reliability',
    description: 'Runtime calls per second and the share of unsuccessful calls.',
    unit: 'calls / sec',
    keys: ['calls', 'errors'],
    labels: ['Calls / sec', 'Unsuccessful %'],
  },
  {
    id: 'latency',
    title: 'Latency breakdown',
    description: 'End-to-end runtime latency compared with model forward time.',
    unit: 'milliseconds',
    keys: ['latency', 'p95', 'forward'],
    labels: ['Mean call', 'P95 call', 'Model forward'],
  },
  {
    id: 'cache',
    title: 'Result cache efficiency',
    description: 'The share of model result lookups served from cache.',
    unit: 'percent',
    keys: ['cache'],
    labels: ['Cache hit rate'],
  },
] satisfies Array<{
  id: string
  title: string
  description: string
  unit: string
  keys: DecisionModelMetricKey[]
  labels: string[]
}>

export default function DecisionModelMonitoringCharts({
  points,
  loading,
}: {
  points: DecisionModelChartPoint[]
  loading: boolean
}) {
  const gradientId = useId().replace(/:/g, '')
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      {chartSections.map((section) => {
        const hasSamples = points.some((point) => section.keys.some((key) => point[key] != null))
        const percentageOnly = section.id === 'cache'
        const latency = section.id === 'latency'
        return (
          <Card key={section.id} className={cn(section.id === 'traffic' && 'xl:col-span-2')}>
            <section aria-label={section.title} className="flex flex-col">
              <CardHeader>
                <CardTitle>{section.title}</CardTitle>
                <CardDescription className="text-xs">{section.description}</CardDescription>
              </CardHeader>
              <div className="flex items-center justify-between gap-3 px-5 pt-3">
                <ChartLegend
                  items={section.keys.map((key, index) => ({
                    label: section.labels[index],
                    color: metricByKey(key).color,
                  }))}
                />
                <span className="text-[11px] text-subtle-foreground">{section.unit}</span>
              </div>
              <div className="h-[220px] px-3 pb-3 pt-2">
                {!hasSamples ? (
                  <div className="flex h-full flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed border-border text-center">
                    <LineChartIcon className="size-5 text-subtle-foreground" />
                    <span className="text-[13px] font-medium">
                      {loading ? 'Loading observations…' : 'No samples in this window'}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      Gaps are unknown, not zero. Run a decision-model test or choose a wider time
                      range.
                    </span>
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                    <ComposedChart
                      data={points}
                      margin={{
                        top: 8,
                        right: section.id === 'traffic' ? 4 : 12,
                        left: -6,
                        bottom: 0,
                      }}
                      accessibilityLayer
                    >
                      <defs>
                        <linearGradient
                          id={`${gradientId}-${section.id}`}
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="0%"
                            stopColor={metricByKey(section.keys[0]).color}
                            stopOpacity={0.22}
                          />
                          <stop
                            offset="100%"
                            stopColor={metricByKey(section.keys[0]).color}
                            stopOpacity={0.01}
                          />
                        </linearGradient>
                      </defs>
                      <CartesianGrid {...chartGrid} />
                      <XAxis
                        dataKey="time"
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={clockTime}
                        minTickGap={45}
                        tickMargin={10}
                        {...chartAxis}
                      />
                      <YAxis
                        yAxisId="main"
                        width={56}
                        domain={percentageOnly ? [0, 100] : [0, 'auto']}
                        tickFormatter={(value: number) =>
                          latency
                            ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(
                                value * 1_000,
                              )
                            : percentageOnly
                              ? `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value)}%`
                              : formatDecisionModelRateAxis(value)
                        }
                        {...chartAxis}
                      />
                      {section.id === 'traffic' ? (
                        <YAxis
                          yAxisId="errors"
                          orientation="right"
                          domain={[0, 100]}
                          width={42}
                          tickFormatter={(value: number) => `${value}%`}
                          {...chartAxis}
                        />
                      ) : null}
                      <Tooltip
                        filterNull={false}
                        cursor={{ stroke: 'var(--border-strong)', strokeDasharray: '3 3' }}
                        content={
                          <ChartTooltip
                            labelFormat={(label) => new Date(Number(label)).toLocaleTimeString()}
                            format={(value, name) => {
                              const metric = DECISION_MODEL_METRICS.find(
                                (item) => item.label === name,
                              )
                              return metric
                                ? formatDecisionModelMetric(Number(value), metric.unit)
                                : String(value)
                            }}
                          />
                        }
                      />
                      {section.keys.map((key, index) =>
                        index === 0 && !latency ? (
                          <Area
                            key={key}
                            yAxisId="main"
                            type="linear"
                            dataKey={key}
                            name={metricByKey(key).label}
                            stroke={metricByKey(key).color}
                            fill={`url(#${gradientId}-${section.id})`}
                            strokeWidth={2}
                            connectNulls={false}
                            isAnimationActive={false}
                          />
                        ) : (
                          <Line
                            key={key}
                            yAxisId={key === 'errors' ? 'errors' : 'main'}
                            type="linear"
                            dataKey={key}
                            name={metricByKey(key).label}
                            stroke={metricByKey(key).color}
                            strokeWidth={2}
                            strokeDasharray={
                              key === 'forward' || key === 'errors' ? '4 4' : undefined
                            }
                            dot={false}
                            activeDot={{ r: 4, strokeWidth: 2 }}
                            connectNulls={false}
                            isAnimationActive={false}
                          />
                        ),
                      )}
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </div>
            </section>
          </Card>
        )
      })}
    </div>
  )
}
