import { describe, expect, it } from 'vitest'

import {
  defaultRequestRange,
  hasAnyFilter,
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
  })

  it('bounds a range to the minute, so a live list keeps one query a minute', () => {
    const minute = Math.floor(Date.parse('2026-11-18T10:42:30Z') / 60_000)
    expect(requestRangeSince('24h', minute)).toBe('2026-11-17T10:42:00.000Z')
    expect(requestRangeSince('all', minute)).toBeUndefined()
    const query = new URLSearchParams(serverQuery(read('model=m'), requestRangeSince('1h', minute)))
    expect(Object.fromEntries(query)).toEqual({ model: 'm', since: '2026-11-18T09:42:00.000Z' })
  })
})
