import {
  decisionModelMetricRange,
  readDecisionModelMetricMatrix,
  type DecisionModelTimeWindow,
} from './decisionModelMetrics'

export const TASK_METRICS = [
  { key: 'calls', label: 'Task calls / sec', unit: 'rate', color: '#7297f5' },
  { key: 'errors', label: 'Errors', unit: 'percent', color: '#ee8b9b' },
  { key: 'unknown', label: 'Unknown results', unit: 'percent', color: '#dca965' },
  { key: 'latency', label: 'Mean latency', unit: 'seconds', color: '#49bba5' },
  { key: 'p95', label: 'P95 latency', unit: 'seconds', color: '#b19ae8' },
] as const
export type TaskMetricKey = (typeof TASK_METRICS)[number]['key']
export type TaskMetricPoint = { time: number } & Partial<Record<TaskMetricKey, number | null>>
export interface TaskMetricSnapshot {
  points: TaskMetricPoint[]
  results: Array<{ result: string; count: number }>
  unavailable: string[]
}
export function decisionTaskQueries(deployment: string, task: string, seconds: number) {
  const selector = `deployment=${JSON.stringify(deployment)},task=${JSON.stringify(task)}`
  const rate = (metric: string, extra = '') =>
    `sum by (deployment) (rate(${metric}{${selector}${extra}}[5m]))`
  const calls = rate('vsr_systemone_task_calls_total')
  const fraction = (outcome: string) =>
    `((${rate('vsr_systemone_task_calls_total', `,outcome="${outcome}"`)} or (${calls} * 0)) / ${calls}) * 100`
  return {
    calls,
    errors: fraction('error'),
    unknown: fraction('unknown'),
    latency: `${rate('vsr_systemone_task_duration_seconds_sum')} / ${rate('vsr_systemone_task_duration_seconds_count')}`,
    p95: `histogram_quantile(0.95, sum by (deployment, le) (rate(vsr_systemone_task_duration_seconds_bucket{${selector}}[5m])))`,
    results: `sum by (result) (increase(vsr_systemone_task_results_total{${selector}}[${seconds}s]))`,
  }
}
export function readTaskResultDistribution(payload: unknown): TaskMetricSnapshot['results'] {
  const response = payload as {
    status?: string
    data?: {
      resultType?: string
      result?: Array<{ metric?: { result?: string }; value?: unknown[] }>
    }
  } | null
  if (
    response?.status !== 'success' ||
    response.data?.resultType !== 'vector' ||
    !Array.isArray(response.data.result)
  )
    throw new Error('Task result distribution is unavailable.')
  return response.data.result
    .flatMap(({ metric, value }) => {
      const count = value && typeof value[1] === 'string' ? Number(value[1]) : NaN
      return metric?.result && Number.isFinite(count) && count > 0
        ? [{ result: metric.result, count }]
        : []
    })
    .sort((left, right) => right.count - left.count)
}
export async function loadDecisionTaskMetrics(
  deployment: string,
  task: string,
  window: DecisionModelTimeWindow,
  signal: AbortSignal,
): Promise<TaskMetricSnapshot> {
  const queries = decisionTaskQueries(deployment, task, window)
  const range = decisionModelMetricRange(window)
  const names = [...TASK_METRICS.map(({ key }) => key), 'results'] as const
  const results = await Promise.allSettled(
    names.map(async (key) => {
      const params = new URLSearchParams({ query: queries[key] })
      if (key !== 'results') {
        params.set('start', String(range.start))
        params.set('end', String(range.end))
        params.set('step', String(range.step))
      }
      const response = await fetch(
        `/embedded/prometheus/api/v1/${key === 'results' ? 'query' : 'query_range'}?${params}`,
        { signal },
      )
      if (!response.ok) throw new Error(`${key} is unavailable`)
      return response.json() as Promise<unknown>
    }),
  )
  const snapshot: TaskMetricSnapshot = { points: [], results: [], unavailable: [] }
  const points = new Map<number, TaskMetricPoint>()
  results.forEach((result, index) => {
    const key = names[index]
    try {
      if (result.status === 'rejected') throw result.reason
      if (key === 'results') snapshot.results = readTaskResultDistribution(result.value)
      else
        for (const sample of readDecisionModelMetricMatrix(result.value, [deployment], range)[
          deployment
        ] ?? []) {
          const point = points.get(sample.time) ?? { time: sample.time }
          point[key] = sample.value
          points.set(sample.time, point)
        }
    } catch {
      snapshot.unavailable.push(key)
    }
  })
  snapshot.points = [...points.values()].sort((a, b) => a.time - b.time)
  return snapshot
}
