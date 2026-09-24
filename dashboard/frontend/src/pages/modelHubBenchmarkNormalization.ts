import type { CatalogBenchmarkMetric } from '../types/modelCatalog'

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value))

export function modelHubBenchmarkNormalizedValue(
  value: number,
  metric: CatalogBenchmarkMetric,
): number {
  const normalization = metric.normalization
  if (!normalization) {
    return metric.unit === 'proportion' || metric.unit === 'fraction' ? clamp01(value) : value
  }
  if (normalization.type === 'identity') return clamp01(value)
  if (normalization.type === 'one_minus') return clamp01(1 - value)
  if (
    normalization.type === 'linear_clamp' &&
    normalization.min !== undefined &&
    normalization.max !== undefined &&
    normalization.min < normalization.max
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
    normalization.type === 'logistic' &&
    normalization.k !== undefined &&
    normalization.x0 !== undefined
  ) {
    return clamp01(1 / (1 + Math.exp(-normalization.k * (value - normalization.x0))))
  }
  if (normalization.type === 'lookup') {
    const mapped = normalization.values?.[String(value)]
    if (mapped !== undefined) return clamp01(mapped)
  }
  return clamp01(value)
}

export function modelHubBenchmarkValueLabel(value: number, metric: CatalogBenchmarkMetric): string {
  if (metric.normalization || metric.unit === 'proportion' || metric.unit === 'fraction') {
    return `${(modelHubBenchmarkNormalizedValue(value, metric) * 100).toFixed(1)}%`
  }
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2)
}

export function modelHubBenchmarkRawValueLabel(
  value: number,
  metric: CatalogBenchmarkMetric,
): string | undefined {
  if (!metric.normalization) return undefined
  const raw = Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2)
  return `${raw} ${metric.unit}`
}
