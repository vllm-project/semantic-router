import { describe, expect, it } from 'vitest'

import {
  hasRunClientFilters,
  matchesRunFilters,
  readRunFilters,
  runHarnessValue,
  runsQuery,
  writeRunFilters,
} from './insightsRunsData'
import type { RunSummary } from './insightsRunTypes'

const cost = {
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
}

const run: RunSummary = {
  session_id: 'run-1',
  harness: 'Claude Code',
  started_at: '2026-11-18T10:00:00Z',
  ended_at: '2026-11-18T10:05:00Z',
  duration_ms: 300_000,
  calls: 4,
  agents: [],
  models: [],
  outcome: { status: 'failed', tool_results: 0, tool_failures: 0 },
  cost,
}

describe('agent run filters', () => {
  it('reads the runs a URL asks for, the last day by default', () => {
    expect(
      readRunFilters(new URLSearchParams('tab=runs&q=retry&harness=codex,api&status=failed')),
    ).toEqual({
      search: 'retry',
      range: '24h',
      harness: ['codex', 'api'],
      status: ['failed'],
    })
    expect(readRunFilters(new URLSearchParams('range=all')).range).toBe('all')
  })

  it('writes filters back at the first page and leaves the default range out', () => {
    const params = new URLSearchParams('tab=runs&page=3&range=7d')
    expect(writeRunFilters(params, { range: '24h' }).toString()).toBe('tab=runs')
    expect(writeRunFilters(params, { search: ' loop ', harness: ['codex'] }).toString()).toBe(
      'tab=runs&range=7d&q=loop&harness=codex',
    )
  })

  it('asks the runs API for the search and for the runs with a call in the range', () => {
    const minute = Math.floor(Date.parse('2026-11-18T10:42:30Z') / 60_000)
    expect(
      Object.fromEntries(new URLSearchParams(runsQuery({ search: 'loop', range: '1h' }, minute))),
    ).toEqual({
      search: 'loop',
      since: '2026-11-18T09:42:00.000Z',
    })
    expect(runsQuery({ search: ' ', range: 'all' }, minute)).toBe('')
  })

  it('filters harness and status in the browser, a run with no harness as an API client', () => {
    expect(runHarnessValue(run)).toBe('claude_code')
    expect(runHarnessValue({ harness: undefined })).toBe('api')
    const filters = readRunFilters(new URLSearchParams('harness=claude_code'))
    expect(hasRunClientFilters(filters)).toBe(true)
    expect(matchesRunFilters(run, filters)).toBe(true)
    expect(matchesRunFilters(run, { ...filters, status: ['answered'] })).toBe(false)
    expect(hasRunClientFilters(readRunFilters(new URLSearchParams('q=loop')))).toBe(false)
  })
})
