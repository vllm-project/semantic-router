import { getRouterModelState, type RouterModelInfo } from '../utils/routerRuntime'

export type ModelInventoryStateFilter = 'all' | 'ready' | 'loading' | 'not_loaded'
export type ModelInventorySort = 'state' | 'name' | 'type'

const LOADING_STATES = new Set(['downloading', 'pending', 'initializing'])

function modelSearchText(model: RouterModelInfo): string {
  return [
    model.name,
    model.type,
    model.model_path,
    model.resolved_model_path,
    model.registry?.local_path,
    model.registry?.repo_id,
    model.registry?.purpose,
    model.registry?.description,
    model.registry?.pipeline_tag,
    ...(model.registry?.tags ?? []),
    ...(model.categories ?? []),
  ]
    .filter(Boolean)
    .join(' ')
    .toLocaleLowerCase()
}

function matchesState(model: RouterModelInfo, filter: ModelInventoryStateFilter): boolean {
  if (filter === 'all') return true

  const state = getRouterModelState(model)
  if (filter === 'loading') return LOADING_STATES.has(state)
  return state === filter
}

export function filterAndSortRouterModels(
  models: RouterModelInfo[],
  query: string,
  stateFilter: ModelInventoryStateFilter,
  sort: ModelInventorySort,
): RouterModelInfo[] {
  const normalizedQuery = query.trim().toLocaleLowerCase()
  const filtered = models.filter(
    (model) =>
      matchesState(model, stateFilter) &&
      (!normalizedQuery || modelSearchText(model).includes(normalizedQuery)),
  )

  return [...filtered].sort((left, right) => {
    if (sort === 'name') return left.name.localeCompare(right.name)
    if (sort === 'type') {
      const byType = (left.registry?.purpose || left.type).localeCompare(
        right.registry?.purpose || right.type,
      )
      return byType || left.name.localeCompare(right.name)
    }

    const rank: Record<string, number> = {
      ready: 0,
      downloading: 1,
      pending: 2,
      initializing: 3,
      not_loaded: 4,
    }
    const byState =
      (rank[getRouterModelState(left)] ?? 99) - (rank[getRouterModelState(right)] ?? 99)
    return byState || left.name.localeCompare(right.name)
  })
}

export function clampInventoryPage(page: number, itemCount: number, pageSize: number): number {
  const totalPages = Math.max(1, Math.ceil(itemCount / pageSize))
  return Math.min(Math.max(1, page), totalPages)
}
