import type { BenchmarkMetric } from './modelHubCatalogTypes'

interface ModelHubEvaluationSelectionSource {
  status: string
  model: string
  benchmark: string
  benchmark_profile: string
  metrics?: Record<string, number | null>
}

export const MODEL_HUB_MIN_BENCHMARK_MODELS = 10

const selectionKey = (benchmark: string, profile: string, metric: string): string =>
  `${benchmark}\u0000${profile}\u0000${metric}`

export function modelHubPublicEvaluations<T extends ModelHubEvaluationSelectionSource>(
  evaluations: T[],
  minimumModels = MODEL_HUB_MIN_BENCHMARK_MODELS,
): T[] {
  const eligible = new Set(modelHubBenchmarkSelectionCounts(evaluations, minimumModels).keys())
  return evaluations.filter(
    evaluation =>
      evaluation.status === 'available'
      && Object.keys(evaluation.metrics ?? {}).some(metric =>
        eligible.has(selectionKey(evaluation.benchmark, evaluation.benchmark_profile, metric)),
      ),
  )
}

export function modelHubBenchmarkSelectionCounts(
  evaluations: ModelHubEvaluationSelectionSource[],
  minimumModels = 1,
): Map<string, number> {
  const modelsBySelection = new Map<string, Set<string>>()
  evaluations.forEach((evaluation) => {
    if (evaluation.status !== 'available') return
    Object.entries(evaluation.metrics ?? {}).forEach(([metric, value]) => {
      if (typeof value !== 'number') return
      const key = selectionKey(evaluation.benchmark, evaluation.benchmark_profile, metric)
      const models = modelsBySelection.get(key) ?? new Set<string>()
      models.add(evaluation.model)
      modelsBySelection.set(key, models)
    })
  })
  return new Map(
    Array.from(modelsBySelection.entries())
      .filter(([, models]) => models.size >= minimumModels)
      .map(([key, models]) => [key, models.size]),
  )
}

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value))

export function modelHubBenchmarkNormalizedValue(
  value: number,
  metric: BenchmarkMetric,
): number {
  const normalization = metric.normalization
  if (!normalization) {
    return metric.unit === 'proportion' || metric.unit === 'fraction' ? clamp01(value) : value
  }
  if (normalization.type === 'identity') return clamp01(value)
  if (normalization.type === 'one_minus') return clamp01(1 - value)
  if (
    normalization.type === 'linear_clamp'
    && normalization.min !== undefined
    && normalization.max !== undefined
    && normalization.min < normalization.max
  ) {
    return clamp01((value - normalization.min) / (normalization.max - normalization.min))
  }
  if (normalization.type === 'piecewise_linear' && normalization.points?.length) {
    const points = normalization.points
    if (value <= points[0].input) return clamp01(points[0].output)
    for (let index = 1; index < points.length; index += 1) {
      const right = points[index]
      if (value > right.input) continue
      const left = points[index - 1]
      const ratio = (value - left.input) / (right.input - left.input)
      return clamp01(left.output + ratio * (right.output - left.output))
    }
    return clamp01(points[points.length - 1]?.output ?? value)
  }
  if (
    normalization.type === 'logistic'
    && normalization.k !== undefined
    && normalization.x0 !== undefined
  ) {
    return clamp01(1 / (1 + Math.exp(-normalization.k * (value - normalization.x0))))
  }
  if (normalization.type === 'lookup') {
    const mapped = normalization.values?.[String(value)]
    if (mapped !== undefined) return clamp01(mapped)
  }
  return clamp01(value)
}

export function modelHubBenchmarkRawValueLabel(
  value: number,
  metric: BenchmarkMetric,
): string | undefined {
  if (!metric.normalization) return undefined
  const raw = Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2)
  return `${raw} ${metric.unit}`
}
