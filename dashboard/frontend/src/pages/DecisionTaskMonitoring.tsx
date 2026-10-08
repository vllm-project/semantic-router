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
import SystemOneSelect from './SystemOneSelect'
import styles from './DecisionModelPage.module.css'

const empty: TaskMetricSnapshot = { points: [], results: [], unavailable: [] }
const colors = {
  tick: { fill: 'var(--color-text-secondary)', fontSize: 11 },
  tooltip: {
    background: 'var(--color-bg-secondary)',
    border: '1px solid var(--color-border)',
    borderRadius: 10,
  },
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
    deployments.find((item) => item.deployment === deploymentChoice) ?? deployments[0]
  const tasks = [
    ...(deployment?.native_question_types.length
      ? [{ id: 'decision', title: 'Recipe questions' }]
      : []),
    ...(catalog.data?.tasks.filter((task) =>
      deployment?.tasks.some(
        (capability) => capability.task_id === task.id && capability.supported,
      ),
    ) ?? []),
  ]
  const task = tasks.find((item) => item.id === taskChoice) ?? tasks[0]
  const canReadMetrics = canAccessDashboardPath(user, '/monitoring')
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
    <section className={styles.panel} aria-labelledby="task-monitoring-title">
      <div className={styles.monitoringHeader}>
        <div>
          <h2 id="task-monitoring-title">Task observations</h2>
          <p className={styles.muted}>
            Model readiness and task outcomes are operational evidence. Evaluation quality requires
            a labeled dataset.
          </p>
        </div>
      </div>
      <div className={styles.taskFilters}>
        <SystemOneSelect
          label="Deployment"
          value={name}
          onChange={setDeployment}
          options={deployments.map((item) => ({
            value: item.deployment,
            label: item.model || item.deployment,
            description: item.deployment,
          }))}
        />
        <SystemOneSelect
          label="Task"
          value={taskId}
          onChange={setTask}
          options={tasks.map((item) => ({ value: item.id, label: item.title }))}
        />
        <SystemOneSelect
          label="Time range"
          value={String(timeWindow)}
          onChange={(value) => setTimeWindow(Number(value) as DecisionModelTimeWindow)}
          options={DECISION_MODEL_TIME_WINDOWS.map((item) => ({
            value: String(item.seconds),
            label: item.label,
          }))}
        />
      </div>
      {catalog.error && <p className={styles.notice}>{catalog.error}</p>}
      {!canReadMetrics ? (
        <p className={styles.muted}>Your role does not include monitoring access.</p>
      ) : (
        <>
          <dl className={styles.statistics} aria-label="Task statistics">
            {TASK_METRICS.map((metric) => (
              <div key={metric.key}>
                <dt>{metric.label}</dt>
                <dd>{formatDecisionModelMetric(current?.[metric.key], metric.unit)}</dd>
              </div>
            ))}
          </dl>
          {snapshot.unavailable.length > 0 && (
            <p className={styles.notice}>
              Unavailable observations: {snapshot.unavailable.join(', ')}
            </p>
          )}
          <div className={styles.monitoringCharts}>
            <section className={styles.chartCard}>
              <div className={styles.chartHeader}>
                <div>
                  <h4>Task traffic & outcomes</h4>
                  <p>Errors and unknown answers are measured separately.</p>
                </div>
              </div>
              <div className={styles.chartLegend}>
                {TASK_METRICS.slice(0, 3).map((metric) => (
                  <span key={metric.key}>
                    <i style={{ background: metric.color }} />
                    {metric.label}
                  </span>
                ))}
              </div>
              <div className={styles.chartCanvas}>
                {hasSamples ? (
                  <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                    <ComposedChart data={snapshot.points} accessibilityLayer>
                      <CartesianGrid
                        vertical={false}
                        stroke="var(--color-border)"
                        strokeDasharray="3 5"
                      />
                      <XAxis
                        dataKey="time"
                        tick={colors.tick}
                        tickFormatter={(value: number) =>
                          new Date(value).toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit',
                          })
                        }
                        minTickGap={45}
                      />
                      <YAxis yAxisId="calls" tick={colors.tick} width={48} />
                      <YAxis
                        yAxisId="percent"
                        orientation="right"
                        domain={[0, 100]}
                        tick={colors.tick}
                        width={40}
                      />
                      <Tooltip
                        contentStyle={colors.tooltip}
                        labelFormatter={(value) => new Date(Number(value)).toLocaleTimeString()}
                      />
                      <Area
                        yAxisId="calls"
                        dataKey="calls"
                        stroke="#7297f5"
                        fill="#7297f530"
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                      <Line
                        yAxisId="percent"
                        dataKey="errors"
                        stroke="#ee8b9b"
                        dot={false}
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                      <Line
                        yAxisId="percent"
                        dataKey="unknown"
                        stroke="#dca965"
                        dot={false}
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : (
                  <div className={styles.chartEmpty}>
                    {loading
                      ? 'Loading task observations…'
                      : 'No task calls observed in this window.'}
                  </div>
                )}
              </div>
            </section>
            <section className={styles.chartCard}>
              <div className={styles.chartHeader}>
                <div>
                  <h4>Result distribution</h4>
                  <p>
                    Estimated answer counts in this window. Unknown answers and errors are excluded.
                  </p>
                </div>
              </div>
              <div className={styles.chartCanvas}>
                {snapshot.results.length ? (
                  <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                    <BarChart data={snapshot.results} layout="vertical" accessibilityLayer>
                      <CartesianGrid
                        horizontal={false}
                        stroke="var(--color-border)"
                        strokeDasharray="3 5"
                      />
                      <XAxis type="number" tick={colors.tick} />
                      <YAxis type="category" dataKey="result" tick={colors.tick} width={115} />
                      <Tooltip contentStyle={colors.tooltip} />
                      <Bar
                        dataKey="count"
                        fill="#49bba5"
                        radius={[0, 5, 5, 0]}
                        isAnimationActive={false}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className={styles.chartEmpty}>
                    {loading
                      ? 'Loading result distribution…'
                      : 'No successful results observed in this window.'}
                  </div>
                )}
              </div>
            </section>
          </div>
          <p className={styles.metricsFootnote}>
            Rates use a 5-minute window. An unknown result means the task could not produce a
            supported, complete answer. These statistics do not measure prediction accuracy.
          </p>
        </>
      )}
    </section>
  )
}
