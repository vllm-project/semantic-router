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
import {
  DECISION_MODEL_METRICS,
  formatDecisionModelMetric,
  formatDecisionModelRateAxis,
  type DecisionModelChartPoint,
  type DecisionModelMetricKey,
} from './decisionModelMetrics'
import styles from './DecisionModelPage.module.css'

const clockTime = (time: number) =>
  new Date(time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
const axisStyle = { fill: 'var(--color-text-secondary)', fontSize: 10 }
const metricByKey = (key: DecisionModelMetricKey) =>
  DECISION_MODEL_METRICS.find((metric) => metric.key === key)!

export function DecisionModelMetricCards({ points }: { points: DecisionModelChartPoint[] }) {
  const gradientId = useId().replace(/:/g, '')
  const current = points[points.length - 1]
  return (
    <dl className={styles.statistics} aria-label="Model statistics">
      {DECISION_MODEL_METRICS.map((metric) => {
        const hasHistory = points.some((point) => point[metric.key] != null)
        return (
          <div key={metric.key}>
            <dt title={metric.help}>
              <span className={styles.legendDot} style={{ background: metric.color }} />
              {metric.label}
            </dt>
            <dd className={current?.[metric.key] == null ? styles.metricUnknown : undefined}>
              {formatDecisionModelMetric(current?.[metric.key], metric.unit)}
            </dd>
            <div className={styles.sparkline} aria-hidden="true">
              {hasHistory ? (
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <AreaChart data={points} margin={{ top: 3, right: 0, bottom: 3, left: 0 }}>
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
                <span className={styles.sparklineEmpty}>No samples in this window</span>
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
    keys: ['calls', 'errors'],
    labels: ['Calls / sec', 'Unsuccessful %'],
  },
  {
    id: 'latency',
    title: 'Latency breakdown',
    description: 'End-to-end runtime latency compared with model forward time.',
    keys: ['latency', 'p95', 'forward'],
    labels: ['Mean call', 'P95 call', 'Model forward'],
  },
  {
    id: 'cache',
    title: 'Result cache efficiency',
    description: 'The share of model result lookups served from cache.',
    keys: ['cache'],
    labels: ['Cache hit rate'],
  },
] satisfies Array<{
  id: string
  title: string
  description: string
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
    <div className={styles.monitoringCharts}>
      {chartSections.map((section) => {
        const hasSamples = points.some((point) => section.keys.some((key) => point[key] != null))
        const percentageOnly = section.id === 'cache'
        const latency = section.id === 'latency'
        return (
          <section
            className={`${styles.chartCard} ${percentageOnly ? styles.cacheChart : ''}`}
            key={section.id}
            aria-label={section.title}
          >
            <div className={styles.chartHeader}>
              <div>
                <h4>{section.title}</h4>
                <p>{section.description}</p>
              </div>
              <span className={styles.chartUnit}>
                {latency ? 'milliseconds' : percentageOnly ? 'percent' : 'calls / sec'}
              </span>
            </div>
            <div className={styles.chartLegend}>
              {section.keys.map((key, index) => (
                <span key={key}>
                  <i style={{ background: metricByKey(key).color }} />
                  {section.labels[index]}
                </span>
              ))}
            </div>
            <div className={styles.chartCanvas}>
              {!hasSamples ? (
                <div className={styles.chartEmpty}>
                  <svg width="40" height="28" viewBox="0 0 40 28" fill="none" aria-hidden="true">
                    <path
                      d="M1 1v26h38M5 20l8-6 7 3 7-11 9 4"
                      stroke="currentColor"
                      strokeWidth="1.5"
                    />
                  </svg>
                  <strong>{loading ? 'Loading observations…' : 'No samples in this window'}</strong>
                  <span>Run a decision-model test or choose a wider time range.</span>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <ComposedChart
                    data={points}
                    margin={{
                      top: 8,
                      right: section.id === 'traffic' ? 4 : 14,
                      left: -10,
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
                          stopOpacity={0.2}
                        />
                        <stop
                          offset="100%"
                          stopColor={metricByKey(section.keys[0]).color}
                          stopOpacity={0.01}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      vertical={false}
                      stroke="var(--color-border)"
                      strokeDasharray="3 5"
                    />
                    <XAxis
                      dataKey="time"
                      type="number"
                      domain={['dataMin', 'dataMax']}
                      tickFormatter={clockTime}
                      tick={axisStyle}
                      axisLine={false}
                      tickLine={false}
                      minTickGap={45}
                      tickMargin={12}
                    />
                    <YAxis
                      yAxisId="main"
                      tick={axisStyle}
                      axisLine={false}
                      tickLine={false}
                      width={62}
                      domain={percentageOnly ? [0, 100] : [0, 'auto']}
                      tickFormatter={(value: number) =>
                        latency
                          ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(
                              value * 1_000,
                            )
                          : percentageOnly
                            ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(
                                value,
                              )
                            : formatDecisionModelRateAxis(value)
                      }
                    />
                    {section.id === 'traffic' && (
                      <YAxis
                        yAxisId="errors"
                        orientation="right"
                        domain={[0, 100]}
                        tick={axisStyle}
                        axisLine={false}
                        tickLine={false}
                        width={42}
                        tickFormatter={(value: number) => `${value}%`}
                      />
                    )}
                    <Tooltip
                      labelFormatter={(label) => new Date(Number(label)).toLocaleTimeString()}
                      formatter={(value: number | string, name: string) => {
                        const metric = metricByKey(name as DecisionModelMetricKey)
                        return [formatDecisionModelMetric(Number(value), metric.unit), metric.label]
                      }}
                      contentStyle={{
                        background: 'var(--color-bg-secondary)',
                        border: '1px solid var(--color-border)',
                        borderRadius: 10,
                        fontSize: 12,
                        boxShadow: '0 8px 28px rgb(0 0 0 / 15%)',
                      }}
                      labelStyle={{ color: 'var(--color-text-secondary)', marginBottom: 6 }}
                    />
                    {section.keys.map((key, index) =>
                      index === 0 && !latency ? (
                        <Area
                          key={key}
                          yAxisId="main"
                          type="linear"
                          dataKey={key}
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
        )
      })}
    </div>
  )
}
