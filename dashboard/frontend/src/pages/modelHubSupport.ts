import type {
  BuiltInModelCatalog,
  BuiltInModelMetadata,
  CatalogModelBinding,
  CatalogProvider,
} from '../types/modelCatalog'

export type ModelHubDistributionFilter = 'all' | BuiltInModelMetadata['distribution']['type']
export type ModelHubLifecycleFilter = 'supported' | 'all' | BuiltInModelMetadata['lifecycle']
export type ModelHubSort = 'name' | 'released' | 'context' | 'providers'

export interface ModelHubFilters {
  query: string
  distribution: ModelHubDistributionFilter
  lifecycle: ModelHubLifecycleFilter
  /** Creators to keep; empty keeps all. */
  publishers: string[]
  /** Serving providers to keep; empty keeps all. */
  providers: string[]
  /** Capabilities a model must all have. */
  capabilities: string[]
  sort: ModelHubSort
}

export const MODEL_HUB_DEFAULT_FILTERS: ModelHubFilters = {
  query: '',
  distribution: 'all',
  lifecycle: 'supported',
  publishers: [],
  providers: [],
  capabilities: [],
  sort: 'released',
}

export interface ModelHubRow {
  model: BuiltInModelMetadata
  providers: Array<{ provider: CatalogProvider; model: CatalogModelBinding }>
  evaluationCount: number
  benchmarkCount: number
}

export interface ModelHubStats {
  models: number
  creators: number
  providers: number
}

interface ModelHubBenchmarkSelection {
  benchmark: string
  profile: string
  metric: string
}

const MODEL_HUB_MIN_BENCHMARK_MODELS = 10

const benchmarkSelectionKey = (benchmark: string, profile: string, metric: string): string =>
  `${benchmark}\u0000${profile}\u0000${metric}`

type ModelHubEvaluationStats = Map<string, { evaluations: number; benchmarks: Set<string> }>
type ModelHubProviderBindings = Map<string, ModelHubRow['providers']>

export const modelHubDistributionLabel: Record<
  BuiltInModelMetadata['distribution']['type'],
  string
> = {
  open_weights: 'Open weights',
  proprietary_api: 'API',
  router_recipe: 'Router recipe',
}

export const modelHubRelationshipLabel: Record<CatalogModelBinding['relationship'], string> = {
  first_party: 'First-party',
  managed_cloud: 'Managed cloud',
  gateway: 'Gateway',
  self_hosted: 'Self-hosted',
}

export const readableModelHubValue = (value: string): string => value.split('_').join(' ')

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
  return {
    models: catalog.models.length,
    creators: modelHubCreators(catalog).length,
    providers: catalog.providers.length,
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

/** Published evidence for benchmarks that cover enough models to be comparable. */
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
  if (filters.distribution !== 'all' && model.distribution.type !== filters.distribution) {
    return false
  }
  if (filters.publishers.length && !filters.publishers.includes(model.publisher)) return false
  if (filters.lifecycle === 'supported') {
    if (model.lifecycle !== 'active' && model.lifecycle !== 'experimental') return false
  } else if (filters.lifecycle !== 'all' && model.lifecycle !== filters.lifecycle) {
    return false
  }
  if (!filters.capabilities.every((capability) => model.capabilities.includes(capability))) {
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

  return catalog.models
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
        !filters.providers.length ||
        row.providers.some(({ provider }) => filters.providers.includes(provider.id)),
    )
    .sort((left, right) => compareModelHubRows(left, right, filters.sort))
}

function modelHubBenchmarkSelections(
  catalog: BuiltInModelCatalog,
  minimumModels = MODEL_HUB_MIN_BENCHMARK_MODELS,
): ModelHubBenchmarkSelection[] {
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
    .map(([key]) => {
      const [benchmark, profile, metric] = key.split('\u0000')
      return { benchmark, profile, metric }
    })
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

/** The backends page, with this catalog model proposed for a new backend. */
export function modelHubBackendLink(modelId: string): string {
  return `/models/backends?add=${encodeURIComponent(modelId)}`
}
