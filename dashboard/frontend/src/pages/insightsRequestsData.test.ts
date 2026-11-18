import { describe, expect, it } from 'vitest'

import { codexSubagentCall } from './insightsAgentFixtures'
import {
  defaultRequestRange,
  harnessCountsQuery,
  harnessFilterValues,
  hasAnyFilter,
  hasClientFilters,
  matchesClientFilters,
  readRequestFilters,
  requestRangeSince,
  serverQuery,
  writeRequestFilters,
} from './insightsRequestsData'

const read = (query: string) => readRequestFilters(new URLSearchParams(query))

describe('request list filters', () => {
  it('opens on the last day, and a session or a Codex turn in full', () => {
    expect(read('').range).toBe('24h')
    expect(read('session=s-1').range).toBe('all')
    expect(read('turn=t-1').range).toBe('all')
    expect(read('session=s-1&range=1h').range).toBe('1h')
    expect(read('range=forever').range).toBe('24h')
    expect(defaultRequestRange({})).toBe('24h')
  })

  it('keeps a range out of the URL when it is the default for the other filters', () => {
    const paged = new URLSearchParams('page=3')
    expect(writeRequestFilters(paged, { range: 'all' }).toString()).toBe('range=all')
    expect(writeRequestFilters(paged, { range: '24h' }).toString()).toBe('')
    expect(
      writeRequestFilters(new URLSearchParams('range=all'), { session: 's-1' }).toString(),
    ).toBe('session=s-1')
    expect(
      writeRequestFilters(new URLSearchParams('session=s-1'), { session: '' }).toString(),
    ).toBe('')
    expect(
      writeRequestFilters(new URLSearchParams('session=s-1&range=7d'), { session: '' }).toString(),
    ).toBe('range=7d')
    expect(writeRequestFilters(paged, { cache: 'all', harness: ['codex'] }).toString()).toBe(
      'harness=codex',
    )
  })

  it('counts every filter but the range as narrowing the list', () => {
    expect(hasAnyFilter(read('range=7d'))).toBe(false)
    expect(hasAnyFilter(read('housekeeping=hide'))).toBe(true)
    expect(hasAnyFilter(read('session=s-1'))).toBe(true)
    expect(hasAnyFilter(read('harness=codex'))).toBe(true)
  })

  it('bounds a range to the minute, so a live list keeps one query a minute', () => {
    const minute = Math.floor(Date.parse('2026-11-18T10:42:30Z') / 60_000)
    expect(requestRangeSince('24h', minute)).toBe('2026-11-17T10:42:00.000Z')
    expect(requestRangeSince('all', minute)).toBeUndefined()
    const query = new URLSearchParams(serverQuery(read('model=m'), requestRangeSince('1h', minute)))
    expect(Object.fromEntries(query)).toEqual({ model: 'm', since: '2026-11-18T09:42:00.000Z' })
  })
})

describe('the harness filter', () => {
  it('reads the harnesses the replay API filters by, and API clients as api_client', () => {
    // "api" names API clients in links from before the server filtered harnesses
    expect(read('harness=codex,claudecode,api,cursor,gemini,codex').harness).toEqual([
      'codex',
      'claude_code',
      'api_client',
      'cursor',
    ])
    expect(read('harness=api_client').harness).toEqual(['api_client'])
  })

  it('asks the replay API for the harnesses and leaves the records it returns alone', () => {
    const filters = read('harness=codex,api_client&model=m')
    expect(Object.fromEntries(new URLSearchParams(serverQuery(filters, undefined)))).toEqual({
      model: 'm',
      harness: 'api_client,codex',
    })
    expect(hasClientFilters(filters)).toBe(false)
    expect(hasClientFilters(read('class=main'))).toBe(true)
    expect(matchesClientFilters(codexSubagentCall, read('harness=claude_code'))).toBe(true)
  })

  it('counts the harnesses under every other server filter', () => {
    const since = '2026-11-18T09:42:00.000Z'
    const query = new URLSearchParams(harnessCountsQuery(read('harness=codex&model=m'), since))
    expect(Object.fromEntries(query)).toEqual({
      model: 'm',
      since,
      limit: '1',
      offset: '0',
      showDetails: 'false',
    })
  })

  it('offers every harness before the counts load, then the counted and the selected ones', () => {
    expect(harnessFilterValues(null, []).map((option) => option.value)).toEqual([
      'claude_code',
      'codex',
      'opencode',
      'cursor',
      'openclaw',
      'deepseek_harness',
      'api_client',
    ])
    expect(harnessFilterValues(null, [])[0].count).toBeUndefined()
    const counted = harnessFilterValues({ api_client: 51234, codex: 977, claude_code: 0 }, [
      'opencode',
    ])
    expect(counted).toEqual([
      { value: 'codex', count: 977 },
      { value: 'opencode', count: 0 },
      { value: 'api_client', count: 51234 },
    ])
  })
})
