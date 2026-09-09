import type { ModelView } from './modelHubCatalogTypes'
import {
  modelHubDirectoryDefaults,
  type ModelHubDirectoryFilters,
} from './modelHubDirectorySupport'

export interface ModelHubBenchmarkUrlState {
  filter: string
  query: string
  publisher: string
}

export interface ModelHubUrlState {
  filters: ModelHubDirectoryFilters
  view: ModelView
  page: number
  benchmark: ModelHubBenchmarkUrlState
  selectedModelID: string | null
}

const DEFAULT_VIEW: ModelView = 'list'
const DEFAULT_PAGE = 1
const DEFAULT_BENCHMARK: ModelHubBenchmarkUrlState = {
  filter: 'all',
  query: '',
  publisher: 'all',
}

const filterParameters: Record<keyof ModelHubDirectoryFilters, string> = {
  search: 'q',
  kind: 'kind',
  distribution: 'distribution',
  publisher: 'creator',
  provider: 'provider',
  capability: 'capability',
  lifecycle: 'lifecycle',
  sort: 'sort',
}

const knownParameters = [
  ...Object.values(filterParameters),
  'view',
  'page',
  'benchmark',
  'benchmark_q',
  'benchmark_creator',
  'model',
]

const oneOf = <T extends string>(
  value: string | null,
  allowed: readonly T[],
  fallback: T,
): T => (value && allowed.includes(value as T) ? value as T : fallback)

const positivePage = (value: string | null): number => {
  if (!value || !/^\d+$/.test(value)) return DEFAULT_PAGE
  const page = Number(value)
  return Number.isSafeInteger(page) && page > 0 ? page : DEFAULT_PAGE
}

export function parseModelHubUrlState(search: string): ModelHubUrlState {
  const parameters = new URLSearchParams(search)
  return {
    filters: {
      search: parameters.get(filterParameters.search) ?? modelHubDirectoryDefaults.search,
      kind: oneOf(
        parameters.get(filterParameters.kind),
        ['all', 'physical', 'virtual'] as const,
        modelHubDirectoryDefaults.kind,
      ),
      distribution: oneOf(
        parameters.get(filterParameters.distribution),
        ['all', 'open_weights', 'proprietary_api', 'router_recipe'] as const,
        modelHubDirectoryDefaults.distribution,
      ),
      publisher: parameters.get(filterParameters.publisher) || modelHubDirectoryDefaults.publisher,
      provider: parameters.get(filterParameters.provider) || modelHubDirectoryDefaults.provider,
      capability: parameters.get(filterParameters.capability) || modelHubDirectoryDefaults.capability,
      lifecycle: oneOf(
        parameters.get(filterParameters.lifecycle),
        ['supported', 'all', 'active', 'experimental', 'deprecated', 'removed'] as const,
        modelHubDirectoryDefaults.lifecycle,
      ),
      sort: oneOf(
        parameters.get(filterParameters.sort),
        ['newest', 'name', 'context'] as const,
        modelHubDirectoryDefaults.sort,
      ),
    },
    view: oneOf(parameters.get('view'), ['list', 'table'] as const, DEFAULT_VIEW),
    page: positivePage(parameters.get('page')),
    benchmark: {
      filter: parameters.get('benchmark') || DEFAULT_BENCHMARK.filter,
      query: parameters.get('benchmark_q') ?? DEFAULT_BENCHMARK.query,
      publisher: parameters.get('benchmark_creator') || DEFAULT_BENCHMARK.publisher,
    },
    selectedModelID: parameters.get('model') || null,
  }
}

const setWhenDifferent = (
  parameters: URLSearchParams,
  key: string,
  value: string,
  defaultValue: string,
) => {
  if (value === defaultValue) return
  parameters.set(key, value)
}

export function serializeModelHubUrlState(
  state: ModelHubUrlState,
  existingSearch = '',
): string {
  const parameters = new URLSearchParams(existingSearch)
  knownParameters.forEach(parameter => parameters.delete(parameter))

  const filterFields = Object.keys(filterParameters) as Array<keyof ModelHubDirectoryFilters>
  filterFields.forEach((field) => {
    const value = state.filters[field]
    const defaultValue = modelHubDirectoryDefaults[field]
    setWhenDifferent(parameters, filterParameters[field], value, defaultValue)
  })
  setWhenDifferent(parameters, 'view', state.view, DEFAULT_VIEW)
  if (state.page !== DEFAULT_PAGE) parameters.set('page', String(state.page))
  setWhenDifferent(
    parameters,
    'benchmark',
    state.benchmark.filter,
    DEFAULT_BENCHMARK.filter,
  )
  setWhenDifferent(parameters, 'benchmark_q', state.benchmark.query, DEFAULT_BENCHMARK.query)
  setWhenDifferent(
    parameters,
    'benchmark_creator',
    state.benchmark.publisher,
    DEFAULT_BENCHMARK.publisher,
  )
  if (state.selectedModelID) parameters.set('model', state.selectedModelID)

  const serialized = parameters.toString()
  return serialized ? `?${serialized}` : ''
}
