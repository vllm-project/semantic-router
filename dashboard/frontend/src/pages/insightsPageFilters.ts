import type { InsightsFilterType, InsightsRecord } from './insightsPageTypes'

interface InsightsFilterState {
  searchTerm: string
  filter: InsightsFilterType
  recipeFilter: string
  decisionFilter: string
  modelFilter: string
}

// The list's filters as the replay API reads them. turn is the Codex turn from the
// page URL, which the API matches on turn_id or parent_turn_id; since is an RFC 3339
// lower bound on the record time.
export interface ReplayQueryFilters extends InsightsFilterState {
  turn: string
  session?: string
  since?: string
}

export function buildReplayQueryString(
  filters: ReplayQueryFilters,
  pagination?: { limit: number; offset: number },
) {
  const params = new URLSearchParams()

  if (pagination) {
    params.set('limit', String(pagination.limit))
    params.set('offset', String(pagination.offset))
    params.set('showDetails', 'false')
  }

  const search = filters.searchTerm.trim()
  if (search) {
    params.set('search', search)
  }
  if (filters.filter !== 'all') {
    params.set('cache_status', filters.filter)
  }
  if (filters.recipeFilter !== 'all') {
    params.set('recipe', filters.recipeFilter)
  }
  if (filters.decisionFilter !== 'all') {
    params.set('decision', filters.decisionFilter)
  }
  if (filters.modelFilter !== 'all') {
    params.set('model', filters.modelFilter)
  }
  if (filters.turn) {
    params.set('turn_id', filters.turn)
  }
  if (filters.session) {
    params.set('session_id', filters.session)
  }
  if (filters.since) {
    params.set('since', filters.since)
  }

  const query = params.toString()
  return query ? `?${query}` : ''
}

export function filterInsightsRecords(records: InsightsRecord[], filters: InsightsFilterState) {
  const searchTerm = filters.searchTerm.trim().toLowerCase()

  return records.filter((record) => {
    if (filters.filter === 'cached' && !record.from_cache) {
      return false
    }
    if (filters.filter === 'streamed' && !record.streaming) {
      return false
    }
    if (filters.recipeFilter !== 'all' && record.recipe !== filters.recipeFilter) {
      return false
    }
    if (filters.decisionFilter !== 'all' && record.decision !== filters.decisionFilter) {
      return false
    }
    if (
      filters.modelFilter !== 'all' &&
      record.selected_model !== filters.modelFilter &&
      record.original_model !== filters.modelFilter
    ) {
      return false
    }
    if (searchTerm && !record.request_id?.toLowerCase().includes(searchTerm)) {
      return false
    }

    return true
  })
}
