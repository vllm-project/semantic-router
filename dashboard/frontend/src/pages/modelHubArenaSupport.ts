import type {
  BuiltInModelCatalog,
  BuiltInModelMetadata,
  CatalogEvaluation,
  CatalogIndex,
  CatalogIndexComponent,
  CatalogIndexResult,
  CatalogMetricNormalization,
} from '../types/modelCatalog'

export type ModelHubArenaScope = 'all' | 'open' | 'virtual'
export type ModelHubArenaLayer = 'overall' | 'capabilities' | 'benchmarks'

export interface ModelHubArenaRoute {
  scope: ModelHubArenaScope
  layer: ModelHubArenaLayer
  capability: string
  benchmark: string
}

export interface ModelHubArenaRow {
  model: BuiltInModelMetadata
  score: number
  reasoningEffort: string
  evidence: string
  rank: number
}

export interface ModelHubArenaSurface {
  id: string
  displayName: string
  description: string
  kind: 'overall' | 'capability' | 'benchmark'
  rows: ModelHubArenaRow[]
}

export interface ModelHubArenaData {
  index: CatalogIndex
  overall: ModelHubArenaSurface
  capabilities: ModelHubArenaSurface[]
  benchmarks: ModelHubArenaSurface[]
}

const oneOf = <T extends string>(value: string | null, allowed: readonly T[], fallback: T): T =>
  value && allowed.includes(value as T) ? (value as T) : fallback

export function parseModelHubArenaRoute(parameters: URLSearchParams): ModelHubArenaRoute {
  return {
    scope: oneOf(parameters.get('arena'), ['all', 'open', 'virtual'] as const, 'all'),
    layer: oneOf(
      parameters.get('arena_layer'),
      ['overall', 'capabilities', 'benchmarks'] as const,
      'overall',
    ),
    capability: parameters.get('arena_capability') ?? '',
    benchmark: parameters.get('arena_benchmark') ?? '',
  }
}

export function serializeModelHubArenaRoute(
  route: ModelHubArenaRoute,
  current: URLSearchParams,
): URLSearchParams {
  const parameters = new URLSearchParams(current)
  const set = (key: string, value: string, fallback: string): void => {
    if (value === fallback) parameters.delete(key)
    else parameters.set(key, value)
  }
  set('arena', route.scope, 'all')
  set('arena_layer', route.layer, 'overall')
  set('arena_capability', route.capability, '')
  set('arena_benchmark', route.benchmark, '')
  return parameters
}

const matchesScope = (model: BuiltInModelMetadata, scope: ModelHubArenaScope): boolean => {
  if (scope === 'open') return model.distribution.type === 'open_weights'
  if (scope === 'virtual') return model.kind === 'virtual'
  return true
}

const effortPosition = (
  catalog: BuiltInModelCatalog,
  model: BuiltInModelMetadata,
  effort: string,
): number => {
  const family = catalog.reasoning_families.find(
    (candidate) => candidate.id === model.reasoning_family,
  )
  const efforts = family?.levels ?? family?.modes ?? ['default']
  const position = efforts.indexOf(effort)
  return position === -1 ? -1 : position
}

function preferredIndexResult(
  catalog: BuiltInModelCatalog,
  model: BuiltInModelMetadata,
  results: CatalogIndexResult[],
): CatalogIndexResult | undefined {
  return results
    .filter((result) => result.status === 'available' && result.score !== null)
    .sort(
      (left, right) =>
        effortPosition(catalog, model, right.reasoning_effort) -
          effortPosition(catalog, model, left.reasoning_effort) ||
        (right.score ?? 0) - (left.score ?? 0) ||
        left.reasoning_effort.localeCompare(right.reasoning_effort),
    )[0]
}

function preferredEvaluation(
  catalog: BuiltInModelCatalog,
  model: BuiltInModelMetadata,
  evaluations: CatalogEvaluation[],
  profiles: string[],
  metric: string,
): CatalogEvaluation | undefined {
  return evaluations
    .filter(
      (evaluation) =>
        evaluation.status === 'available' &&
        profiles.includes(evaluation.benchmark_profile) &&
        typeof evaluation.metrics[metric] === 'number',
    )
    .sort(
      (left, right) =>
        effortPosition(catalog, model, right.reasoning_effort) -
          effortPosition(catalog, model, left.reasoning_effort) ||
        profiles.indexOf(left.benchmark_profile) - profiles.indexOf(right.benchmark_profile) ||
        (right.measured_at ?? right.observed_at ?? '').localeCompare(
          left.measured_at ?? left.observed_at ?? '',
        ) ||
        left.id.localeCompare(right.id),
    )[0]
}

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value))

function normalizeMetric(value: number, normalization: CatalogMetricNormalization): number {
  switch (normalization.type) {
    case 'one_minus':
      return clamp01(1 - value)
    case 'linear_clamp': {
      const minimum = normalization.min ?? 0
      const maximum = normalization.max ?? 1
      return maximum === minimum ? 0 : clamp01((value - minimum) / (maximum - minimum))
    }
    case 'piecewise_linear': {
      const points = [...(normalization.points ?? [])].sort(
        (left, right) => left.input - right.input,
      )
      if (!points.length) return clamp01(value)
      if (value <= points[0].input) return clamp01(points[0].output)
      if (value >= points[points.length - 1].input) return clamp01(points[points.length - 1].output)
      const upperIndex = points.findIndex((point) => point.input >= value)
      const lower = points[upperIndex - 1]
      const upper = points[upperIndex]
      const position = (value - lower.input) / (upper.input - lower.input)
      return clamp01(lower.output + position * (upper.output - lower.output))
    }
    case 'logistic':
      return clamp01(
        1 / (1 + Math.exp(-(normalization.k ?? 1) * (value - (normalization.x0 ?? 0)))),
      )
    case 'lookup':
      return clamp01(normalization.values?.[String(value)] ?? value)
    case 'identity':
    default:
      return clamp01(value)
  }
}

function rankedRows(rows: Omit<ModelHubArenaRow, 'rank'>[]): ModelHubArenaRow[] {
  const sorted = rows.sort(
    (left, right) =>
      right.score - left.score || left.model.display_name.localeCompare(right.model.display_name),
  )
  let previousScore: number | null = null
  let previousRank = 0
  return sorted.map((row, position) => {
    if (previousScore === null || row.score !== previousScore) previousRank = position + 1
    previousScore = row.score
    return { ...row, rank: previousRank }
  })
}

function indexSurface(
  catalog: BuiltInModelCatalog,
  models: BuiltInModelMetadata[],
  index: CatalogIndex,
  kind: 'overall' | 'capability',
): ModelHubArenaSurface {
  const resultsByModel = new Map<string, CatalogIndexResult[]>()
  catalog.index_results
    .filter((result) => result.index === index.id)
    .forEach((result) => {
      resultsByModel.set(result.model, [...(resultsByModel.get(result.model) ?? []), result])
    })
  return {
    id: index.id,
    displayName: index.display_name,
    description: index.description,
    kind,
    rows: rankedRows(
      models.flatMap<Omit<ModelHubArenaRow, 'rank'>>((model) => {
        const result = preferredIndexResult(catalog, model, resultsByModel.get(model.id) ?? [])
        if (!result || result.score === null) return []
        return [
          {
            model,
            score: result.score,
            reasoningEffort: result.reasoning_effort,
            evidence: `${result.index}:${result.reasoning_effort}`,
          },
        ]
      }),
    ),
  }
}

function benchmarkSurface(
  catalog: BuiltInModelCatalog,
  models: BuiltInModelMetadata[],
  component: CatalogIndexComponent,
  capabilityName: string,
): ModelHubArenaSurface | null {
  const benchmark = catalog.benchmarks.find((candidate) => candidate.id === component.benchmark)
  if (!benchmark || !component.metric) return null
  const profiles =
    component.benchmark_profiles ??
    (component.benchmark_profile ? [component.benchmark_profile] : [benchmark.default_profile])
  const evaluationsByModel = new Map<string, CatalogEvaluation[]>()
  catalog.evaluations
    .filter((evaluation) => evaluation.benchmark === benchmark.id)
    .forEach((evaluation) => {
      evaluationsByModel.set(evaluation.model, [
        ...(evaluationsByModel.get(evaluation.model) ?? []),
        evaluation,
      ])
    })
  return {
    id: benchmark.id,
    displayName: benchmark.display_name,
    description: `${capabilityName} · ${component.metric.split('_').join(' ')}`,
    kind: 'benchmark',
    rows: rankedRows(
      models.flatMap<Omit<ModelHubArenaRow, 'rank'>>((model) => {
        const evaluation = preferredEvaluation(
          catalog,
          model,
          evaluationsByModel.get(model.id) ?? [],
          profiles,
          component.metric ?? '',
        )
        const value = evaluation?.metrics[component.metric ?? '']
        if (!evaluation || typeof value !== 'number') return []
        return [
          {
            model,
            score: normalizeMetric(value, component.normalization) * 100,
            reasoningEffort: evaluation.reasoning_effort,
            evidence: evaluation.id,
          },
        ]
      }),
    ),
  }
}

export function modelHubArenaData(
  catalog: BuiltInModelCatalog,
  scope: ModelHubArenaScope,
): ModelHubArenaData | null {
  const defaultIndex = catalog.catalogs[0]?.default_intelligence_index
  const index = catalog.indices.find((candidate) => candidate.id === defaultIndex)
  if (!index) return null
  const models = catalog.models.filter((model) => matchesScope(model, scope))
  const capabilityIndices = index.components.flatMap((component) => {
    const capability = catalog.indices.find((candidate) => candidate.id === component.index)
    return capability ? [capability] : []
  })
  const benchmarkIDs = new Set<string>()
  const benchmarks = capabilityIndices.flatMap((capability) =>
    capability.components.flatMap((component) => {
      if (!component.benchmark || benchmarkIDs.has(component.benchmark)) return []
      benchmarkIDs.add(component.benchmark)
      const surface = benchmarkSurface(catalog, models, component, capability.display_name)
      return surface ? [surface] : []
    }),
  )
  return {
    index,
    overall: indexSurface(catalog, models, index, 'overall'),
    capabilities: capabilityIndices.map((capability) =>
      indexSurface(catalog, models, capability, 'capability'),
    ),
    benchmarks,
  }
}
