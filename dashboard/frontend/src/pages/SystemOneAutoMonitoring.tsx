import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useAuth } from '../contexts/AuthContext'
import {
  Alert,
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
  ChartLegend,
  ChartTooltip,
  Segmented,
  chartAxis,
  chartGrid,
  cn,
} from '../ui'
import { canAccessDashboardPath } from '../utils/accessControl'
import { withRequestTimeout } from '../utils/boundedRequest'
import {
  DECISION_MODEL_TIME_WINDOWS,
  formatDecisionModelMetric,
  type DecisionModelTimeWindow,
} from './decisionModelMetrics'
import {
  AUTO_METRICS,
  loadSystemOneAutoMetrics,
  type AutoMetricSnapshot,
} from './systemOneAutoMetrics'

const empty: AutoMetricSnapshot = { values: {}, stages: [], unavailable: [] }
const outcomes = [
  { key: 'accepted', label: 'Accepted by gate', color: 'var(--chart-6)' },
  { key: 'rejected', label: 'Did not pass gate', color: 'var(--chart-2)' },
  { key: 'failed', label: 'Errors or invalid results', color: 'var(--chart-5)' },
] as const

export default function SystemOneAutoMonitoring({ refreshedAt }: { refreshedAt: Date | null }) {
  const { user } = useAuth()
  const canReadMetrics = canAccessDashboardPath(user, '/observe/metrics')
  const [window, setWindow] = useState<DecisionModelTimeWindow>(3600)
  const [snapshot, setSnapshot] = useState(empty)
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    setSnapshot(empty)
  }, [window])
  useEffect(() => {
    if (!canReadMetrics || document.visibilityState === 'hidden') return
    const controller = new AbortController()
    setLoading(true)
    void withRequestTimeout(
      (signal) => loadSystemOneAutoMetrics(window, signal),
      controller.signal,
      8000,
    )
      .then((next) => {
        if (!controller.signal.aborted) setSnapshot(next)
      })
      .catch(() => {
        if (!controller.signal.aborted) setSnapshot({ ...empty, unavailable: ['Auto metrics'] })
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [window, refreshedAt, canReadMetrics])
  return (
    <section className="flex flex-col gap-4" aria-labelledby="auto-monitoring-title">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <h2 id="auto-monitoring-title" className="text-[15px] font-semibold tracking-[-0.012em]">
            Automatic routing
          </h2>
          <p className="text-[13px] text-muted-foreground">
            All native auto routes on this instance. Acceptance measures the configured gate, not
            answer accuracy.
          </p>
        </div>
        <Segmented
          aria-label="Stage observation window"
          value={String(window)}
          onValueChange={(value) => setWindow(Number(value) as DecisionModelTimeWindow)}
          options={DECISION_MODEL_TIME_WINDOWS.map((item) => ({
            value: String(item.seconds),
            label: item.label,
          }))}
        />
      </div>
      {!canReadMetrics ? (
        <Alert tone="neutral">Your role does not include monitoring access.</Alert>
      ) : (
        <>
          <dl
            className="grid grid-cols-2 gap-3 lg:grid-cols-4"
            aria-label="Auto routing statistics"
          >
            {AUTO_METRICS.map((metric) => {
              const value = snapshot.values[metric.key]
              return (
                <div
                  key={metric.key}
                  className="rounded-xl border border-border bg-card px-4 py-3.5 shadow-xs"
                >
                  <dt className="text-[13px] font-medium text-muted-foreground">{metric.label}</dt>
                  <dd
                    className={cn(
                      'mt-2 tabular',
                      value == null
                        ? 'text-[15px] font-medium leading-6 text-subtle-foreground'
                        : 'text-[22px] font-semibold leading-6 tracking-[-0.03em]',
                    )}
                  >
                    {formatDecisionModelMetric(value, metric.unit)}
                  </dd>
                </div>
              )
            })}
          </dl>
          {snapshot.unavailable.length > 0 ? (
            <Alert tone="warning" title="Some observations are unavailable">
              Unavailable observations: {snapshot.unavailable.join(', ')}
            </Alert>
          ) : null}
          <Card>
            <CardHeader>
              <CardTitle>Stage activity</CardTitle>
              <CardDescription className="text-xs">
                Estimated stage attempts in the selected window, grouped by algorithm and model. A
                request can visit several stages. Statistics use a rolling 5-minute window;
                execution time starts after recipe signals.
              </CardDescription>
            </CardHeader>
            <div className="px-5 pt-3">
              <ChartLegend
                items={outcomes.map((outcome) => ({ label: outcome.label, color: outcome.color }))}
              />
            </div>
            <div
              className="px-3 pb-3 pt-2"
              style={{ height: Math.max(220, snapshot.stages.length * 52) }}
            >
              {snapshot.stages.length ? (
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <BarChart data={snapshot.stages} layout="vertical" accessibilityLayer>
                    <CartesianGrid {...chartGrid} horizontal={false} vertical />
                    <XAxis type="number" {...chartAxis} />
                    <YAxis type="category" dataKey="label" width={180} {...chartAxis} />
                    <Tooltip
                      cursor={{ fill: 'var(--muted)', opacity: 0.5 }}
                      content={<ChartTooltip />}
                    />
                    {outcomes.map((outcome) => (
                      <Bar
                        key={outcome.key}
                        dataKey={outcome.key}
                        name={outcome.label}
                        stackId="attempts"
                        fill={outcome.color}
                        maxBarSize={28}
                        isAnimationActive={false}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-border text-[13px] text-muted-foreground">
                  {loading
                    ? 'Loading auto observations…'
                    : 'No native auto stage activity reported in this window.'}
                </div>
              )}
            </div>
          </Card>
        </>
      )}
    </section>
  )
}
