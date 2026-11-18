import { useEffect, useState } from 'react'
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useAuth } from '../contexts/AuthContext'
import {
  Alert,
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
  ChartLegend,
  ChartTooltip,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
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
  loadDecisionTaskMetrics,
  TASK_METRICS,
  type TaskMetricSnapshot,
} from './decisionTaskMetrics'
import { useDecisionTasks } from './useDecisionTasks'

const empty: TaskMetricSnapshot = { points: [], results: [], unavailable: [] }
const clockTime = (value: number) =>
  new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

function Picker({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: Array<{ value: string; label: string; description?: string }>
}) {
  return (
    <Select value={value || undefined} onValueChange={onChange} disabled={!options.length}>
      <SelectTrigger size="sm" className="max-w-64 gap-2" aria-label={label}>
        <span className="text-xs text-subtle-foreground">{label}</span>
        <SelectValue placeholder="None">
          {options.find((option) => option.value === value)?.label}
        </SelectValue>
      </SelectTrigger>
      <SelectContent align="end">
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
            {option.description ? (
              <span className="ml-2 font-mono text-[11px] text-subtle-foreground">
                {option.description}
              </span>
            ) : null}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export default function DecisionTaskMonitoring({ refreshedAt }: { refreshedAt: Date | null }) {
  const { user } = useAuth()
  const catalog = useDecisionTasks()
  const [deploymentChoice, setDeployment] = useState('')
  const [taskChoice, setTask] = useState('')
  const [timeWindow, setTimeWindow] = useState<DecisionModelTimeWindow>(3600)
  const [snapshot, setSnapshot] = useState(empty)
  const [loading, setLoading] = useState(false)
  const deployments = catalog.data?.deployments ?? []
  const deployment =
    deployments.find((item) => item.deployment === deploymentChoice) ??
    deployments.find((item) => item.deployment === catalog.data?.default_deployment) ??
    deployments[0]
  const tasks = [
    ...(deployment?.native_question_types.length
      ? [{ id: 'systemone', title: 'Recipe questions' }]
      : []),
    ...(catalog.data?.tasks.filter((task) =>
      deployment?.tasks.some(
        (capability) => capability.task_id === task.id && capability.supported,
      ),
    ) ?? []),
  ]
  const task = tasks.find((item) => item.id === taskChoice) ?? tasks[0]
  const canReadMetrics = canAccessDashboardPath(user, '/observe/metrics')
  const name = deployment?.deployment ?? ''
  const taskId = task?.id ?? ''
  useEffect(() => {
    setSnapshot(empty)
  }, [name, taskId, timeWindow])
  useEffect(() => {
    if (!name || !taskId || !canReadMetrics || document.visibilityState === 'hidden') return
    const controller = new AbortController()
    setLoading(true)
    void withRequestTimeout(
      (signal) => loadDecisionTaskMetrics(name, taskId, timeWindow, signal),
      controller.signal,
      8000,
    )
      .then((next) => {
        if (!controller.signal.aborted) setSnapshot(next)
      })
      .catch(() => {
        if (!controller.signal.aborted) setSnapshot({ ...empty, unavailable: ['Task metrics'] })
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [name, taskId, timeWindow, refreshedAt, canReadMetrics])
  const current = snapshot.points[snapshot.points.length - 1]
  const hasSamples = snapshot.points.some((point) => point.calls != null)
  return (
    <section className="flex flex-col gap-4" aria-labelledby="task-monitoring-title">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <h2 id="task-monitoring-title" className="text-[15px] font-semibold tracking-[-0.012em]">
            Task observations
          </h2>
          <p className="text-[13px] text-muted-foreground">
            Model readiness and task outcomes are operational evidence. Evaluation quality requires
            a labeled dataset.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Picker
            label="Deployment"
            value={name}
            onChange={setDeployment}
            options={deployments.map((item) => ({
              value: item.deployment,
              label: item.model || item.deployment,
              description:
                item.model && item.model !== item.deployment ? item.deployment : undefined,
            }))}
          />
          <Picker
            label="Task"
            value={taskId}
            onChange={setTask}
            options={tasks.map((item) => ({ value: item.id, label: item.title }))}
          />
          <Picker
            label="Time range"
            value={String(timeWindow)}
            onChange={(value) => setTimeWindow(Number(value) as DecisionModelTimeWindow)}
            options={DECISION_MODEL_TIME_WINDOWS.map((item) => ({
              value: String(item.seconds),
              label: item.label,
            }))}
          />
        </div>
      </div>
      {catalog.error ? <Alert tone="warning">{catalog.error}</Alert> : null}
      {!canReadMetrics ? (
        <Alert tone="neutral">Your role does not include monitoring access.</Alert>
      ) : (
        <>
          <dl className="grid grid-cols-2 gap-3 lg:grid-cols-5" aria-label="Task statistics">
            {TASK_METRICS.map((metric) => {
              const value = current?.[metric.key]
              return (
                <div
                  key={metric.key}
                  className="rounded-xl border border-border bg-card px-4 py-3.5 shadow-xs"
                >
                  <dt className="flex items-center gap-2 text-[13px] font-medium text-muted-foreground">
                    <span className="size-2 rounded-[3px]" style={{ background: metric.color }} />
                    {metric.label}
                  </dt>
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
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Task traffic & outcomes</CardTitle>
                <CardDescription className="text-xs">
                  Errors and unknown answers are measured separately.
                </CardDescription>
              </CardHeader>
              <div className="px-5 pt-3">
                <ChartLegend
                  items={TASK_METRICS.slice(0, 3).map((metric) => ({
                    label: metric.label,
                    color: metric.color,
                  }))}
                />
              </div>
              <div className="h-[220px] px-3 pb-3 pt-2">
                {hasSamples ? (
                  <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                    <ComposedChart data={snapshot.points} accessibilityLayer>
                      <CartesianGrid {...chartGrid} />
                      <XAxis
                        dataKey="time"
                        tickFormatter={clockTime}
                        minTickGap={45}
                        {...chartAxis}
                      />
                      <YAxis yAxisId="calls" width={48} {...chartAxis} />
                      <YAxis
                        yAxisId="percent"
                        orientation="right"
                        domain={[0, 100]}
                        width={40}
                        {...chartAxis}
                      />
                      <Tooltip
                        filterNull={false}
                        content={
                          <ChartTooltip
                            labelFormat={(value) => new Date(Number(value)).toLocaleTimeString()}
                          />
                        }
                      />
                      <Area
                        yAxisId="calls"
                        dataKey="calls"
                        name="Task calls / sec"
                        stroke="var(--chart-1)"
                        fill="var(--chart-1)"
                        fillOpacity={0.14}
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                      <Line
                        yAxisId="percent"
                        dataKey="errors"
                        name="Errors"
                        stroke="var(--chart-5)"
                        dot={false}
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                      <Line
                        yAxisId="percent"
                        dataKey="unknown"
                        name="Unknown results"
                        stroke="var(--chart-2)"
                        dot={false}
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-border text-[13px] text-muted-foreground">
                    {loading
                      ? 'Loading task observations…'
                      : 'No task calls observed in this window.'}
                  </div>
                )}
              </div>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Result distribution</CardTitle>
                <CardDescription className="text-xs">
                  Estimated answer counts in this window. Unknown answers and errors are excluded.
                </CardDescription>
              </CardHeader>
              <div className="h-[244px] px-3 pb-3 pt-2">
                {snapshot.results.length ? (
                  <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                    <BarChart data={snapshot.results} layout="vertical" accessibilityLayer>
                      <CartesianGrid {...chartGrid} horizontal={false} vertical />
                      <XAxis type="number" {...chartAxis} />
                      <YAxis type="category" dataKey="result" width={115} {...chartAxis} />
                      <Tooltip
                        cursor={{ fill: 'var(--muted)', opacity: 0.5 }}
                        content={<ChartTooltip />}
                      />
                      <Bar
                        dataKey="count"
                        name="Answers"
                        fill="var(--chart-4)"
                        radius={[0, 5, 5, 0]}
                        isAnimationActive={false}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-border text-[13px] text-muted-foreground">
                    {loading
                      ? 'Loading result distribution…'
                      : 'No successful results observed in this window.'}
                  </div>
                )}
              </div>
            </Card>
          </div>
          <p className="text-xs text-subtle-foreground">
            Rates use a 5-minute window. An unknown result means the task could not produce a
            supported, complete answer. These statistics do not measure prediction accuracy.
          </p>
        </>
      )}
    </section>
  )
}
