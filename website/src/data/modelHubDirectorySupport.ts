import type { CatalogModel, Distribution, ModelKind } from './modelHubCatalogTypes'

export interface ModelHubDirectoryFilters {
  search: string
  kind: 'all' | ModelKind
  distribution: 'all' | Distribution
  publisher: string
  provider: string
  capability: string
  lifecycle: 'supported' | 'all' | CatalogModel['lifecycle']
  sort: 'newest' | 'name' | 'context'
}

export const modelHubDirectoryDefaults: ModelHubDirectoryFilters = {
  search: '',
  kind: 'all',
  distribution: 'all',
  publisher: 'all',
  provider: 'all',
  capability: 'all',
  lifecycle: 'supported',
  sort: 'newest',
}

export const modelHubActiveFilterCount = (filters: ModelHubDirectoryFilters): number =>
  [
    filters.search.trim(),
    filters.kind !== modelHubDirectoryDefaults.kind,
    filters.distribution !== modelHubDirectoryDefaults.distribution,
    filters.publisher !== modelHubDirectoryDefaults.publisher,
    filters.provider !== modelHubDirectoryDefaults.provider,
    filters.capability !== modelHubDirectoryDefaults.capability,
    filters.lifecycle !== modelHubDirectoryDefaults.lifecycle,
    filters.sort !== modelHubDirectoryDefaults.sort,
  ].filter(Boolean).length
