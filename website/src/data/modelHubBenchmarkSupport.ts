import type { BenchmarkMetric } from './modelHubCatalogTypes'

export interface ModelHubBenchmarkSelection {
  benchmark: string
  profile: string
  metric: string
}

export interface ModelHubChartColor {
  lightness: number
  chroma: number
  hue: number
}

interface ModelHubEvaluationSelectionSource {
  status: string
  model: string
  benchmark: string
  benchmark_profile: string
  metrics?: Record<string, number | null>
}

export const MODEL_HUB_MIN_BENCHMARK_MODELS = 10

const selectionKey = (
  benchmark: string,
  profile: string,
  metric: string,
): string => `${benchmark}\u0000${profile}\u0000${metric}`

export function modelHubPublicEvaluations<T extends ModelHubEvaluationSelectionSource>(
  evaluations: T[],
  minimumModels = MODEL_HUB_MIN_BENCHMARK_MODELS,
): T[] {
  const eligible = new Set(
    modelHubBenchmarkSelectionCounts(evaluations, minimumModels).keys(),
  )
  return evaluations.filter(
    evaluation =>
      evaluation.status === 'available'
      && Object.keys(evaluation.metrics ?? {}).some(metric =>
        eligible.has(
          selectionKey(
            evaluation.benchmark,
            evaluation.benchmark_profile,
            metric,
          ),
        )),
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
      const key = selectionKey(
        evaluation.benchmark,
        evaluation.benchmark_profile,
        metric,
      )
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

export function preferredModelHubBenchmarkSelection(
  counts: Map<string, number>,
  benchmarkID?: string,
  profileID?: string,
): ModelHubBenchmarkSelection | undefined {
  const prefix = [benchmarkID, profileID]
    .filter(value => value !== undefined)
    .join('\u0000')
  const scopedPrefix = prefix ? `${prefix}\u0000` : ''
  const key = Array.from(counts.entries())
    .filter(([candidate]) => candidate.startsWith(scopedPrefix))
    .sort(
      (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
    )[0]?.[0]
  if (!key) return undefined
  const [benchmark, profile, metric] = key.split('\u0000')
  return { benchmark, profile, metric }
}

export function availableModelHubBenchmarkProfiles(
  counts: Map<string, number>,
  benchmarkID: string,
): Set<string> {
  const profiles = new Set<string>()
  const prefix = `${benchmarkID}\u0000`
  counts.forEach((_count, key) => {
    if (!key.startsWith(prefix)) return
    profiles.add(key.split('\u0000')[1])
  })
  return profiles
}

export function availableModelHubBenchmarkMetrics(
  counts: Map<string, number>,
  benchmarkID: string,
  profileID: string,
): Set<string> {
  const metrics = new Set<string>()
  const prefix = `${benchmarkID}\u0000${profileID}\u0000`
  counts.forEach((_count, key) => {
    if (!key.startsWith(prefix)) return
    metrics.add(key.split('\u0000')[2])
  })
  return metrics
}

const stableModelHubHash = (value: string): number => {
  let hash = 2166136261
  for (const character of value) {
    hash ^= character.charCodeAt(0)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

const modelHubPalette = (): ModelHubChartColor[] =>
  [0.52, 0.62, 0.72].flatMap(lightness =>
    [0.12, 0.18].flatMap(chroma =>
      Array.from({ length: 24 }, (_, index) => ({
        lightness,
        chroma,
        hue: index * 15,
      })),
    ),
  )

export const modelHubChartColorDistance = (
  left: ModelHubChartColor,
  right: ModelHubChartColor,
): number => {
  const radians = (degrees: number) => (degrees * Math.PI) / 180
  const leftA = left.chroma * Math.cos(radians(left.hue))
  const leftB = left.chroma * Math.sin(radians(left.hue))
  const rightA = right.chroma * Math.cos(radians(right.hue))
  const rightB = right.chroma * Math.sin(radians(right.hue))
  return Math.hypot(
    left.lightness - right.lightness,
    leftA - rightA,
    leftB - rightB,
  )
}

const fallbackModelHubColor = (id: string): ModelHubChartColor => {
  const hash = stableModelHubHash(id)
  return {
    lightness: 0.52 + ((hash >>> 24) % 20) / 100,
    chroma: 0.12 + ((hash >>> 16) % 7) / 100,
    hue: (hash / 0x1_0000_0000) * 360,
  }
}

export function modelHubChartColors(
  modelIDs: string[],
  catalogModelIDs: string[] = modelIDs,
): Map<string, ModelHubChartColor> {
  const requested = new Set(modelIDs)
  const universe = Array.from(new Set([...catalogModelIDs, ...modelIDs])).sort(
    (left, right) => left.localeCompare(right),
  )
  const candidates = modelHubPalette()
  const assigned: ModelHubChartColor[] = []
  const allocation = new Map<string, ModelHubChartColor>()

  universe.forEach((id) => {
    if (!candidates.length) {
      allocation.set(id, fallbackModelHubColor(id))
      return
    }
    let bestIndex = 0
    let bestDistance = -1
    candidates.forEach((candidate, index) => {
      const distance = assigned.length
        ? Math.min(
            ...assigned.map(color => modelHubChartColorDistance(candidate, color)),
          )
        : Number.POSITIVE_INFINITY
      if (distance > bestDistance) {
        bestDistance = distance
        bestIndex = index
      }
    })
    const [color] = candidates.splice(bestIndex, 1)
    allocation.set(id, color)
    assigned.push(color)
  })

  return new Map(Array.from(allocation).filter(([id]) => requested.has(id)))
}

export const modelHubChartColorToken = (color: ModelHubChartColor): string =>
  `oklch(${(color.lightness * 100).toFixed(0)}% ${color.chroma.toFixed(2)} ${color.hue})`

export function modelHubBenchmarkDomain(
  values: number[],
  metric: BenchmarkMetric,
): [number, number] {
  const finiteValues = values.filter(Number.isFinite)
  if (!finiteValues.length) return metric.range
  const minimum = Math.min(...finiteValues)
  const maximum = Math.max(...finiteValues)

  if (metric.direction === 'higher_is_better' && maximum > 0) {
    return [Math.min(0, minimum), maximum]
  }
  if (minimum === maximum) {
    return metric.direction === 'lower_is_better'
      ? [minimum, minimum + Math.max(Math.abs(minimum), 1)]
      : [Math.min(0, minimum), maximum || 1]
  }
  return [minimum, maximum]
}

export function modelHubBenchmarkBarHeight(
  value: number,
  minimum: number,
  maximum: number,
  direction: 'higher_is_better' | 'lower_is_better',
): number {
  const span = maximum - minimum || 1
  const position = (value - minimum) / span
  const performance = direction === 'lower_is_better' ? 1 - position : position
  return Math.max(4, Math.min(100, performance * 100))
}
