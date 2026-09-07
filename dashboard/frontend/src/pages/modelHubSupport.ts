import type {
  BuiltInModelCatalog,
  BuiltInModelMetadata,
  CatalogBenchmarkMetric,
  CatalogModelBinding,
  CatalogIndexResult,
  CatalogProvider,
} from '../types/modelCatalog'
import { modelHubBenchmarkNormalizedValue } from './modelHubBenchmarkNormalization'

export type ModelHubKindFilter = 'all' | 'physical' | 'virtual'
export type ModelHubDistributionFilter = 'all' | BuiltInModelMetadata['distribution']['type']
export type ModelHubLifecycleFilter = 'supported' | 'all' | BuiltInModelMetadata['lifecycle']
export type ModelHubSort = 'name' | 'released' | 'context' | 'providers'
export type ModelHubView = 'list' | 'table' | 'benchmarks'

export interface ModelHubFilters {
  query: string
  kind: ModelHubKindFilter
  distribution: ModelHubDistributionFilter
  lifecycle: ModelHubLifecycleFilter
  publisher: string
  provider: string
  capability: string
  sort: ModelHubSort
}

export interface ModelHubRow {
  model: BuiltInModelMetadata
  providers: Array<{ provider: CatalogProvider; model: CatalogModelBinding }>
  evaluationCount: number
  benchmarkCount: number
}

export interface ModelHubStats {
  models: number
  physicalModels: number
  virtualModels: number
  mappedProviders: number
  providerContracts: number
  creators: number
  evaluatedModels: number
  evaluations: number
}

export interface ModelHubBenchmarkSelection {
  benchmark: string
  profile: string
  metric: string
}

export interface ModelHubBenchmarkPoint {
  model: BuiltInModelMetadata
  reasoningEffort: string
  value: number
  evaluation: string
}

export const MODEL_HUB_MIN_BENCHMARK_MODELS = 10

const benchmarkSelectionKey = (benchmark: string, profile: string, metric: string): string =>
  `${benchmark}\u0000${profile}\u0000${metric}`

export interface ModelHubChartColor {
  lightness: number
  chroma: number
  hue: number
}

export interface ModelHubPagination<T> {
  items: T[]
  page: number
  pageSize: number
  totalItems: number
  totalPages: number
  start: number
  end: number
}

type ModelHubEvaluationStats = Map<string, { evaluations: number; benchmarks: Set<string> }>
type ModelHubProviderBindings = Map<string, ModelHubRow['providers']>

export function resolveModelHubSelection(
  rows: ModelHubRow[],
  selectedID: string | null,
): ModelHubRow | null {
  return rows.find((row) => row.model.id === selectedID) ?? rows[0] ?? null
}

export const modelHubDistributionLabel: Record<
  BuiltInModelMetadata['distribution']['type'],
  string
> = {
  open_weights: 'Open weights',
  proprietary_api: 'Proprietary API',
  router_recipe: 'Router recipe',
}

export const modelHubRelationshipLabel: Record<CatalogModelBinding['relationship'], string> = {
  first_party: 'First-party',
  managed_cloud: 'Managed cloud',
  gateway: 'Gateway',
  self_hosted: 'Self-hosted',
}

export const readableModelHubValue = (value: string): string => value.split('_').join(' ')

export const modelHubRowElementID = (modelID: string): string =>
  `model-hub-row-${encodeURIComponent(modelID)}`

export function modelHubPageForModel<T extends { model: { id: string } }>(
  rows: T[],
  modelID: string,
  pageSize: number,
): number {
  const index = rows.findIndex((row) => row.model.id === modelID)
  return index < 0 ? 1 : Math.floor(index / Math.max(1, Math.floor(pageSize))) + 1
}

const modelHubPublishedConditionLabels: Record<string, string> = {
  unspecified: 'Effort not reported',
  default: 'Published default',
  enabled: 'Reasoning enabled',
  disabled: 'Non-reasoning run',
  adaptive: 'Adaptive reasoning',
  none: 'No reasoning',
  no_think: 'No reasoning',
}

export function modelHubEvaluationConditionLabel(
  model: Pick<BuiltInModelMetadata, 'reasoning_family'>,
  condition: string,
): string {
  const readable = readableModelHubValue(condition.trim())
  if (model.reasoning_family) return `${readable} effort`
  const publishedLabel = modelHubPublishedConditionLabels[condition.trim()]
  if (publishedLabel) return publishedLabel
  if (!readable) return modelHubPublishedConditionLabels.unspecified
  return `${readable.charAt(0).toLocaleUpperCase()}${readable.slice(1)} run`
}

const normalizedSearch = (value: string): string => value.trim().toLocaleLowerCase()

export function modelHubStats(catalog: BuiltInModelCatalog): ModelHubStats {
  const publicEvaluations = modelHubPublicEvaluations(catalog)
  const evaluatedModels = new Set(publicEvaluations.map((result) => result.model))
  return {
    models: catalog.models.length,
    physicalModels: catalog.models.filter((model) => model.kind === 'physical').length,
    virtualModels: catalog.models.filter((model) => model.kind === 'virtual').length,
    mappedProviders: modelHubProviders(catalog).length,
    providerContracts: catalog.providers.length,
    creators: new Set(
      catalog.models.filter((model) => model.kind === 'physical').map((model) => model.publisher),
    ).size,
    evaluatedModels: evaluatedModels.size,
    evaluations: publicEvaluations.length,
  }
}

export function modelHubCreators(catalog: BuiltInModelCatalog): string[] {
  return [
    ...new Set(
      catalog.models.filter((model) => model.kind === 'physical').map((model) => model.publisher),
    ),
  ].sort((left, right) => left.localeCompare(right))
}

export function modelHubProviders(catalog: BuiltInModelCatalog): CatalogProvider[] {
  const modelIDs = new Set(catalog.models.map((model) => model.id))
  return catalog.providers
    .filter((provider) => provider.models?.some((model) => modelIDs.has(model.catalog)))
    .sort((left, right) => left.display_name.localeCompare(right.display_name))
}

export function modelHubCapabilities(catalog: BuiltInModelCatalog): string[] {
  return [...new Set(catalog.models.flatMap((model) => model.capabilities ?? []))].sort(
    (left, right) => left.localeCompare(right),
  )
}

function modelHubEvaluationStats(catalog: BuiltInModelCatalog): ModelHubEvaluationStats {
  const stats: ModelHubEvaluationStats = new Map()
  modelHubPublicEvaluations(catalog).forEach((evaluation) => {
    const current = stats.get(evaluation.model) ?? {
      evaluations: 0,
      benchmarks: new Set<string>(),
    }
    current.evaluations += 1
    current.benchmarks.add(evaluation.benchmark)
    stats.set(evaluation.model, current)
  })
  return stats
}

export function modelHubPublicEvaluations(catalog: BuiltInModelCatalog) {
  const eligible = new Set(
    modelHubBenchmarkSelections(catalog).map((selection) =>
      benchmarkSelectionKey(selection.benchmark, selection.profile, selection.metric),
    ),
  )
  return catalog.evaluations.filter(
    (evaluation) =>
      evaluation.status === 'available' &&
      Object.keys(evaluation.metrics).some((metric) =>
        eligible.has(
          benchmarkSelectionKey(evaluation.benchmark, evaluation.benchmark_profile, metric),
        ),
      ),
  )
}

function modelHubProviderBindings(catalog: BuiltInModelCatalog): ModelHubProviderBindings {
  const bindings: ModelHubProviderBindings = new Map()
  catalog.providers.forEach((provider) =>
    (provider.models ?? []).forEach((model) => {
      const existing = bindings.get(model.catalog) ?? []
      existing.push({ provider, model })
      bindings.set(model.catalog, existing)
    }),
  )
  return bindings
}

function matchesModelHubModel(
  model: BuiltInModelMetadata,
  filters: ModelHubFilters,
  query: string,
): boolean {
  if (filters.kind !== 'all' && model.kind !== filters.kind) return false
  if (filters.distribution !== 'all' && model.distribution.type !== filters.distribution) {
    return false
  }
  if (filters.publisher !== 'all' && model.publisher !== filters.publisher) return false
  if (filters.lifecycle === 'supported') {
    if (model.lifecycle !== 'active' && model.lifecycle !== 'experimental') return false
  } else if (filters.lifecycle !== 'all' && model.lifecycle !== filters.lifecycle) {
    return false
  }
  if (!query) return true
  return [
    model.id,
    model.display_name,
    model.description,
    model.publisher,
    model.family,
    ...(model.capabilities ?? []),
    ...(model.tags ?? []),
  ]
    .join(' ')
    .toLocaleLowerCase()
    .includes(query)
}

function compareModelHubRows(left: ModelHubRow, right: ModelHubRow, sort: ModelHubSort): number {
  const byName = left.model.display_name.localeCompare(right.model.display_name)
  if (sort === 'name') return byName
  if (sort === 'released') {
    return (right.model.released_at ?? '').localeCompare(left.model.released_at ?? '') || byName
  }
  if (sort === 'context') {
    return (
      (right.model.limits?.context_window_size ?? -1) -
        (left.model.limits?.context_window_size ?? -1) || byName
    )
  }
  if (sort === 'providers') return right.providers.length - left.providers.length || byName
  return byName
}

export function modelHubRows(
  catalog: BuiltInModelCatalog,
  filters: ModelHubFilters,
): ModelHubRow[] {
  const evaluationStats = modelHubEvaluationStats(catalog)
  const providers = modelHubProviderBindings(catalog)
  const query = normalizedSearch(filters.query)

  const rows = catalog.models
    .filter((model) => matchesModelHubModel(model, filters, query))
    .map((model) => {
      const evaluation = evaluationStats.get(model.id)
      return {
        model,
        providers: [...(providers.get(model.id) ?? [])].sort((left, right) =>
          left.provider.display_name.localeCompare(right.provider.display_name),
        ),
        evaluationCount: evaluation?.evaluations ?? 0,
        benchmarkCount: evaluation?.benchmarks.size ?? 0,
      }
    })
    .filter(
      (row) =>
        filters.provider === 'all' ||
        row.providers.some(({ provider }) => provider.id === filters.provider),
    )
    .filter(
      (row) => filters.capability === 'all' || row.model.capabilities.includes(filters.capability),
    )

  return rows.sort((left, right) => compareModelHubRows(left, right, filters.sort))
}

export function paginateModelHubRows<T>(
  rows: T[],
  requestedPage: number,
  requestedPageSize: number,
): ModelHubPagination<T> {
  const pageSize = Math.max(1, Math.floor(requestedPageSize))
  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize))
  const page = Math.min(Math.max(1, Math.floor(requestedPage)), totalPages)
  const startIndex = (page - 1) * pageSize
  const items = rows.slice(startIndex, startIndex + pageSize)
  return {
    items,
    page,
    pageSize,
    totalItems: rows.length,
    totalPages,
    start: rows.length ? startIndex + 1 : 0,
    end: Math.min(startIndex + pageSize, rows.length),
  }
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
  [0.52, 0.62, 0.72].flatMap((lightness) =>
    [0.12, 0.18].flatMap((chroma) =>
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
  return Math.hypot(left.lightness - right.lightness, leftA - rightA, leftB - rightB)
}

const fallbackModelHubColor = (id: string): ModelHubChartColor => {
  const hash = stableModelHubHash(id)
  return {
    lightness: 0.52 + ((hash >>> 24) % 20) / 100,
    chroma: 0.12 + ((hash >>> 16) % 7) / 100,
    hue: (hash / 0x1_0000_0000) * 360,
  }
}

/**
 * Allocates colors against the complete catalog so filtering and pagination never recolor a model.
 * The fixed OKLCH candidates are consumed farthest-first to keep neighboring bars distinguishable.
 */
export function modelHubChartColors(
  modelIDs: string[],
  catalogModelIDs: string[] = modelIDs,
): Map<string, ModelHubChartColor> {
  const requested = new Set(modelIDs)
  const universe = [...new Set([...catalogModelIDs, ...modelIDs])].sort((left, right) =>
    left.localeCompare(right),
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
        ? Math.min(...assigned.map((color) => modelHubChartColorDistance(candidate, color)))
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

  return new Map([...allocation].filter(([id]) => requested.has(id)))
}

export const modelHubChartColorToken = (color: ModelHubChartColor): string =>
  `oklch(${(color.lightness * 100).toFixed(0)}% ${color.chroma.toFixed(2)} ${color.hue})`

export function modelHubBenchmarkDomain(
  values: number[],
  metric: CatalogBenchmarkMetric,
): [number, number] {
  const finiteValues = values
    .filter(Number.isFinite)
    .map((value) => modelHubBenchmarkNormalizedValue(value, metric))
    .filter(Number.isFinite)
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

export function modelHubBenchmarkSelections(
  catalog: BuiltInModelCatalog,
  minimumModels = MODEL_HUB_MIN_BENCHMARK_MODELS,
): Array<ModelHubBenchmarkSelection & { modelCount: number }> {
  const modelsBySelection = new Map<string, Set<string>>()
  catalog.evaluations.forEach((evaluation) => {
    if (evaluation.status !== 'available') return
    Object.keys(evaluation.metrics).forEach((metric) => {
      const key = benchmarkSelectionKey(evaluation.benchmark, evaluation.benchmark_profile, metric)
      const models = modelsBySelection.get(key) ?? new Set<string>()
      models.add(evaluation.model)
      modelsBySelection.set(key, models)
    })
  })
  return Array.from(modelsBySelection.entries())
    .filter(([, models]) => models.size >= minimumModels)
    .map(([key, models]) => {
      const [benchmark, profile, metric] = key.split('\u0000')
      return { benchmark, profile, metric, modelCount: models.size }
    })
    .sort(
      (left, right) =>
        right.modelCount - left.modelCount ||
        `${left.benchmark}\u0000${left.profile}\u0000${left.metric}`.localeCompare(
          `${right.benchmark}\u0000${right.profile}\u0000${right.metric}`,
        ),
    )
}

/** Selects the broadest published profile/metric for each benchmark card. */
export function modelHubBenchmarkOverviewSelections(
  catalog: BuiltInModelCatalog,
  minimumModels = MODEL_HUB_MIN_BENCHMARK_MODELS,
): Array<ModelHubBenchmarkSelection & { modelCount: number }> {
  const seen = new Set<string>()
  return modelHubBenchmarkSelections(catalog, minimumModels).filter((selection) => {
    if (seen.has(selection.benchmark)) return false
    seen.add(selection.benchmark)
    return true
  })
}

export function modelHubDefaultBenchmark(
  catalog: BuiltInModelCatalog,
): ModelHubBenchmarkSelection | null {
  const best = modelHubBenchmarkSelections(catalog)[0]
  return best ? { benchmark: best.benchmark, profile: best.profile, metric: best.metric } : null
}

export function modelHubBenchmarkPoints(
  catalog: BuiltInModelCatalog,
  selection: ModelHubBenchmarkSelection,
  modelIDs?: Set<string>,
): ModelHubBenchmarkPoint[] {
  const models = new Map(catalog.models.map((model) => [model.id, model]))
  const direction = catalog.benchmarks
    .find((benchmark) => benchmark.id === selection.benchmark)
    ?.metrics.find((metric) => metric.id === selection.metric)?.direction
  const compareValues =
    direction === 'lower_is_better'
      ? (left: number, right: number) => left - right
      : (left: number, right: number) => right - left
  return catalog.evaluations
    .filter(
      (evaluation) =>
        evaluation.status === 'available' &&
        evaluation.benchmark === selection.benchmark &&
        evaluation.benchmark_profile === selection.profile &&
        typeof evaluation.metrics[selection.metric] === 'number' &&
        (!modelIDs || modelIDs.has(evaluation.model)),
    )
    .flatMap((evaluation) => {
      const model = models.get(evaluation.model)
      if (!model) return []
      return [
        {
          model,
          reasoningEffort: evaluation.reasoning_effort,
          value: evaluation.metrics[selection.metric],
          evaluation: evaluation.id,
        },
      ]
    })
    .sort(
      (left, right) =>
        compareValues(left.value, right.value) ||
        left.model.display_name.localeCompare(right.model.display_name),
    )
}

export function modelIndexResults(
  catalog: BuiltInModelCatalog,
  model: BuiltInModelMetadata,
  index?: string,
): CatalogIndexResult[] {
  const results = catalog.index_results.filter(
    (result) => result.model === model.id && result.index === index,
  )
  const family = catalog.reasoning_families.find(
    (candidate) => candidate.id === model.reasoning_family,
  )
  const order = new Map(
    (family?.levels ?? ['default']).map((effort, position) => [effort, position]),
  )
  return [...results].sort(
    (left, right) =>
      (order.get(left.reasoning_effort) ?? Number.MAX_SAFE_INTEGER) -
        (order.get(right.reasoning_effort) ?? Number.MAX_SAFE_INTEGER) ||
      left.reasoning_effort.localeCompare(right.reasoning_effort),
  )
}

export function preferredIndexResult(
  catalog: BuiltInModelCatalog,
  model: BuiltInModelMetadata,
  results: CatalogIndexResult[],
): CatalogIndexResult | null {
  const family = catalog.reasoning_families.find(
    (candidate) => candidate.id === model.reasoning_family,
  )
  const preferredEffort = family?.default ?? 'default'
  const preferred = results.find((result) => result.reasoning_effort === preferredEffort)
  if (preferred?.status === 'available') return preferred
  return (
    results
      .filter((result) => result.status === 'available')
      .sort((left, right) => right.coverage - left.coverage)[0] ??
    preferred ??
    results[0] ??
    null
  )
}

export function formatContextWindow(tokens?: number): string {
  if (!tokens) return 'Not published'
  const binaryMillions = tokens / 1_048_576
  if (tokens >= 1_048_576 && Number.isInteger(binaryMillions)) return `${binaryMillions}M`
  if (tokens >= 1_000_000) {
    const millions = tokens / 1_000_000
    return `${Number(millions.toFixed(2))}M`
  }
  const binaryThousands = tokens / 1_024
  if (tokens % 1_000 !== 0 && Number.isInteger(binaryThousands)) {
    return `${binaryThousands}K`
  }
  if (tokens >= 1_000) {
    const thousands = tokens / 1_000
    return `${Number.isInteger(thousands) ? thousands : thousands.toFixed(1)}K`
  }
  return String(tokens)
}

export function modelHubContextLabel(model: BuiltInModelMetadata): string {
  if (model.limits?.context_window_size) {
    return formatContextWindow(model.limits.context_window_size)
  }
  if (model.kind === 'virtual') return 'By backend'
  return model.distribution.type === 'proprietary_api' ? 'Undisclosed' : 'Not cataloged'
}

export function modelHubMaxOutputLabel(model: BuiltInModelMetadata): string {
  if (model.limits?.max_output_tokens) {
    return formatContextWindow(model.limits.max_output_tokens)
  }
  if (model.kind === 'virtual') return 'By backend'
  return model.distribution.type === 'open_weights' ? 'Runtime-set' : 'Provider-defined'
}

export function modelHubParameterLabel(model: BuiltInModelMetadata): string {
  if (model.parameter_size) return model.parameter_size
  if (model.kind === 'virtual') return 'Varies by pool'
  return model.distribution.type === 'proprietary_api' ? 'Undisclosed' : 'Not cataloged'
}

export function formatIntelligence(result: CatalogIndexResult | null): string {
  if (!result || result.status !== 'available' || result.score === null) {
    return 'Not yet measured'
  }
  return result.score.toFixed(1)
}

export function benchmarkName(
  benchmarkID: string | undefined,
  metricID: string,
  catalog: BuiltInModelCatalog,
): string {
  const benchmark = catalog.benchmarks.find((candidate) => candidate.id === benchmarkID)
  return benchmark ? `${benchmark.display_name} · ${metricID}` : metricID
}
