export const AUTO_METRICS = [
  { key: 'calls', label: 'Auto requests / sec', unit: 'rate' },
  { key: 'unresolved', label: 'Unresolved or canceled', unit: 'percent' },
  { key: 'latency', label: 'Mean execution', unit: 'seconds' },
  { key: 'p95', label: 'P95 execution', unit: 'seconds' },
] as const
export type AutoMetricKey = (typeof AUTO_METRICS)[number]['key']
export interface AutoStageObservation {
  label: string
  algorithm: string
  stage: string
  model: string
  accepted: number
  rejected: number
  failed: number
}
export interface AutoMetricSnapshot {
  values: Partial<Record<AutoMetricKey, number | null>>
  stages: AutoStageObservation[]
  unavailable: string[]
}
export function systemOneAutoQueries(seconds: number) {
  const calls = 'sum(rate(sr_systemone_auto_requests_total[5m]))'
  return {
    calls,
    unresolved: `((sum(rate(sr_systemone_auto_requests_total{outcome!="resolved"}[5m])) or (${calls} * 0)) / ${calls}) * 100`,
    latency:
      'sum(rate(sr_systemone_auto_duration_seconds_sum[5m])) / sum(rate(sr_systemone_auto_duration_seconds_count[5m]))',
    p95: 'histogram_quantile(0.95, sum by (le) (rate(sr_systemone_auto_duration_seconds_bucket[5m])))',
    stages: `sum by (algorithm, stage, model, outcome) (increase(sr_systemone_stage_total[${seconds}s]))`,
  }
}
interface VectorItem {
  metric?: Record<string, string>
  value?: unknown[]
}
function vector(payload: unknown): VectorItem[] {
  const body = payload as {
    status?: string
    data?: { resultType?: string; result?: VectorItem[] }
  } | null
  if (
    body?.status !== 'success' ||
    body.data?.resultType !== 'vector' ||
    !Array.isArray(body.data.result)
  )
    throw new Error('Observations unavailable')
  return body.data.result
}
function finiteSample(item?: VectorItem): number | null {
  const value = item?.value?.[1]
  if (typeof value !== 'string' || !value.trim()) return null
  const number = Number(value)
  return Number.isFinite(number) && number >= 0 ? number : null
}
export function readAutoValue(payload: unknown): number | null {
  const samples = vector(payload)
  if (samples.length > 1) throw new Error('Expected one aggregate')
  return finiteSample(samples[0])
}
export function readAutoStages(payload: unknown): AutoStageObservation[] {
  const stages = new Map<string, AutoStageObservation>()
  for (const item of vector(payload)) {
    const count = finiteSample(item)
    const { algorithm, stage, model, outcome } = item.metric ?? {}
    if (count == null || count === 0 || !algorithm || !stage || !model || !outcome) continue
    const key = JSON.stringify([algorithm, stage, model])
    const row = stages.get(key) ?? {
      label: `${algorithm} / ${stage} · ${model}`,
      algorithm,
      stage,
      model,
      accepted: 0,
      rejected: 0,
      failed: 0,
    }
    if (outcome === 'accepted') row.accepted += count
    else if (outcome === 'rejected') row.rejected += count
    else row.failed += count
    stages.set(key, row)
  }
  return [...stages.values()].sort((a, b) => a.label.localeCompare(b.label))
}
export async function loadSystemOneAutoMetrics(
  seconds: number,
  signal: AbortSignal,
): Promise<AutoMetricSnapshot> {
  const queries = systemOneAutoQueries(seconds)
  const names = [...AUTO_METRICS.map(({ key }) => key), 'stages'] as const
  const results = await Promise.allSettled(
    names.map(async (key) => {
      const query = new URLSearchParams({ query: queries[key] })
      const response = await fetch(`/embedded/prometheus/api/v1/query?${query}`, { signal })
      if (!response.ok) throw new Error('Metrics unavailable')
      return response.json() as Promise<unknown>
    }),
  )
  const snapshot: AutoMetricSnapshot = { values: {}, stages: [], unavailable: [] }
  results.forEach((result, index) => {
    const key = names[index]
    try {
      if (result.status === 'rejected') throw result.reason
      if (key === 'stages') snapshot.stages = readAutoStages(result.value)
      else snapshot.values[key] = readAutoValue(result.value)
    } catch {
      snapshot.unavailable.push(key)
    }
  })
  return snapshot
}
