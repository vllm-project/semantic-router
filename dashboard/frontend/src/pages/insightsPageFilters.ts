import type { InsightsFilterType, InsightsRecord } from './insightsPageTypes'

interface InsightsFilterState {
  searchTerm: string
  filter: InsightsFilterType
  recipeFilter: string
  decisionFilter: string
  modelFilter: string
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
