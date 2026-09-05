import type {
  BuiltInModelCatalog,
  BuiltInModelMetadata,
  CatalogModelBinding,
  CatalogIndexResult,
  CatalogProvider,
} from '../types/modelCatalog'

export type ModelHubKindFilter = 'all' | 'physical' | 'virtual'
export type ModelHubDistributionFilter =
  | 'all'
  | BuiltInModelMetadata['distribution']['type']
export type ModelHubLifecycleFilter =
  | 'supported'
  | 'all'
  | BuiltInModelMetadata['lifecycle']
export type ModelHubSort = 'intelligence' | 'name'

export interface ModelHubFilters {
  query: string
  kind: ModelHubKindFilter
  distribution: ModelHubDistributionFilter
  lifecycle: ModelHubLifecycleFilter
  publisher: string
  sort: ModelHubSort
}

export interface ModelHubRow {
  model: BuiltInModelMetadata
  providers: Array<{ provider: CatalogProvider; model: CatalogModelBinding }>
  intelligence: CatalogIndexResult | null
  intelligenceByEffort: CatalogIndexResult[]
}

export interface ModelHubStats {
  models: number
  physicalModels: number
  virtualModels: number
  providers: number
  publishers: number
  scoredModels: number
}

const normalizedSearch = (value: string): string => value.trim().toLocaleLowerCase()

export function modelHubStats(catalog: BuiltInModelCatalog): ModelHubStats {
  const defaultIndex = catalog.catalogs[0]?.default_intelligence_index
  const scoredModels = new Set(
    catalog.index_results
      .filter((result) => result.index === defaultIndex && result.status === 'available')
      .map((result) => result.model),
  )
  return {
    models: catalog.models.length,
    physicalModels: catalog.models.filter((model) => model.kind === 'physical').length,
    virtualModels: catalog.models.filter((model) => model.kind === 'virtual').length,
    providers: catalog.providers.length,
    publishers: new Set(catalog.models.map((model) => model.publisher)).size,
    scoredModels: scoredModels.size,
  }
}

export function modelHubPublishers(catalog: BuiltInModelCatalog): string[] {
  return [...new Set(catalog.models.map((model) => model.publisher))].sort((left, right) =>
    left.localeCompare(right),
  )
}

export function modelHubRows(
  catalog: BuiltInModelCatalog,
  filters: ModelHubFilters,
): ModelHubRow[] {
  const defaultIndex = catalog.catalogs[0]?.default_intelligence_index
  const scores = new Map(
    catalog.models.map((model) => [model.id, modelIndexResults(catalog, model, defaultIndex)]),
  )
  const providers = new Map<
    string,
    Array<{ provider: CatalogProvider; model: CatalogModelBinding }>
  >()
  catalog.providers.forEach((provider) =>
    (provider.models ?? []).forEach((model) => {
      const existing = providers.get(model.catalog) ?? []
      existing.push({ provider, model })
      providers.set(model.catalog, existing)
    }),
  )
  const query = normalizedSearch(filters.query)

  const rows = catalog.models
    .filter((model) => filters.kind === 'all' || model.kind === filters.kind)
    .filter(
      (model) =>
        filters.distribution === 'all' || model.distribution.type === filters.distribution,
    )
    .filter((model) => filters.publisher === 'all' || model.publisher === filters.publisher)
    .filter((model) => {
      if (filters.lifecycle === 'all') return true
      if (filters.lifecycle === 'supported') {
        return model.lifecycle === 'active' || model.lifecycle === 'experimental'
      }
      return model.lifecycle === filters.lifecycle
    })
    .filter((model) => {
      if (!query) return true
      const haystack = [
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
      return haystack.includes(query)
    })
    .map((model) => {
      const intelligenceByEffort = scores.get(model.id) ?? []
      return {
        model,
        providers: [...(providers.get(model.id) ?? [])].sort((left, right) =>
          left.provider.display_name.localeCompare(right.provider.display_name),
        ),
        intelligence: preferredIndexResult(catalog, model, intelligenceByEffort),
        intelligenceByEffort,
      }
    })

  return rows.sort((left, right) => {
    if (filters.sort === 'name') {
      return left.model.display_name.localeCompare(right.model.display_name)
    }
    const leftScore =
      left.intelligence?.status === 'available' ? (left.intelligence.score ?? -1) : -1
    const rightScore =
      right.intelligence?.status === 'available' ? (right.intelligence.score ?? -1) : -1
    return (
      rightScore - leftScore ||
      left.model.display_name.localeCompare(right.model.display_name)
    )
  })
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
  const order = new Map((family?.levels ?? ['default']).map((effort, position) => [effort, position]))
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
  if (tokens >= 1_000_000) {
    const millions = tokens / 1_000_000
    return `${Number.isInteger(millions) ? millions : millions.toFixed(2)}M`
  }
  if (tokens >= 1_000) {
    const thousands = tokens / 1_000
    return `${Number.isInteger(thousands) ? thousands : thousands.toFixed(1)}K`
  }
  return String(tokens)
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
