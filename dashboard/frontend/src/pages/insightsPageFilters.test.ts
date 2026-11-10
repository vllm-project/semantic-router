import { describe, expect, it } from 'vitest'

import { filterInsightsRecords } from './insightsPageFilters'
import type { InsightsRecord } from './insightsPageTypes'

const all = {
  filter: 'all',
  recipeFilter: 'all',
  decisionFilter: 'all',
  modelFilter: 'all',
} as const

describe('filterInsightsRecords search', () => {
  const records = [
    { id: 'a', request_id: 'req-1', recipe: 'default', session_id: '6582a295-0801-4f55' },
    { id: 'b', request_id: 'req-2', recipe: 'coding' },
  ] as InsightsRecord[]

  it('finds an agent session by the id its harness sent', () => {
    expect(
      filterInsightsRecords(records, { ...all, searchTerm: '6582A295' }).map((r) => r.id),
    ).toEqual(['a'])
  })

  it('still finds a request id or a recipe', () => {
    expect(
      filterInsightsRecords(records, { ...all, searchTerm: 'req-2' }).map((r) => r.id),
    ).toEqual(['b'])
    expect(
      filterInsightsRecords(records, { ...all, searchTerm: 'coding' }).map((r) => r.id),
    ).toEqual(['b'])
  })
})
