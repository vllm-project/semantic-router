import type {
  BuiltInModelCatalog,
  BuiltInModelMetadata,
  CatalogIndexResult,
  CatalogOffering,
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
  offerings: CatalogOffering[]
  intelligence: CatalogIndexResult | null
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
  return {
    models: catalog.models.length,
    physicalModels: catalog.models.filter((model) => model.kind === 'physical').length,
    virtualModels: catalog.models.filter((model) => model.kind === 'virtual').length,
    providers: catalog.providers.length,
    publishers: new Set(catalog.models.map((model) => model.publisher)).size,
    scoredModels: catalog.index_results.filter(
      (result) => result.index === defaultIndex && result.status === 'available',
    ).length,
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
    catalog.index_results
      .filter((result) => result.index === defaultIndex)
      .map((result) => [result.model, result]),
  )
  const offerings = new Map<string, CatalogOffering[]>()
  catalog.offerings.forEach((offering) => {
    const existing = offerings.get(offering.model) ?? []
    existing.push(offering)
    offerings.set(offering.model, existing)
  })
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
    .map((model) => ({
      model,
      offerings: [...(offerings.get(model.id) ?? [])].sort((left, right) =>
        left.provider.localeCompare(right.provider),
      ),
      intelligence: scores.get(model.id) ?? null,
    }))

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

export function benchmarkName(metric: string, catalog: BuiltInModelCatalog): string {
  const [benchmarkID, metricID] = metric.split('#', 2)
  const benchmark = catalog.benchmarks.find((candidate) => candidate.id === benchmarkID)
  return benchmark ? `${benchmark.display_name} · ${metricID}` : metric
}
