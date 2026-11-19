import { afterEach, describe, expect, it, vi } from 'vitest'

import { chosenSession, isScopedSessionId, sessionParam } from './insightsAgentIdentity'
import type { RunSummary } from './insightsRunTypes'
import { lookUpSession, sessionMatches } from './insightsSessionScope'

const run = (session_id: string): RunSummary => ({
  session_id,
  started_at: '2026-11-19T02:36:38Z',
  ended_at: '2026-11-19T02:36:42Z',
  duration_ms: 4_000,
  calls: 5,
  agents: [],
  models: [],
  outcome: { status: 'answered', tool_results: 0, tool_failures: 0 },
  cost: { unpriced_records: 0, unpriced_router_calls: 0, prompt_tokens: 0, completion_tokens: 0 },
})

describe('session ids by the harness', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('links by the id the harness sent and remembers the full id for this tab', () => {
    expect(sessionParam('key_3fa9c2:ses_e0e5a1YCOSIf')).toBe('ses_e0e5a1YCOSIf')
    expect(chosenSession('ses_e0e5a1YCOSIf')).toBe('key_3fa9c2:ses_e0e5a1YCOSIf')
    expect(sessionParam('session-1')).toBe('session-1')
    expect(chosenSession('session-1')).toBeUndefined()
    expect(isScopedSessionId('key_3fa9c2:ses_e0e5a1YCOSIf')).toBe(true)
    expect(isScopedSessionId('ses_e0e5a1YCOSIf')).toBe(false)
  })

  it('matches the whole id after the key scope, not a part of it', () => {
    const runs = [
      run('key_3fa9c2:ses_e0e5a1YCOSIf'),
      run('key_77b104:ses_e0e5a1YCOSIfX'),
      run('ses_e0e5a1YCOSIf'),
    ]
    expect(sessionMatches('ses_e0e5a1YCOSIf', runs).map((entry) => entry.session_id)).toEqual([
      'key_3fa9c2:ses_e0e5a1YCOSIf',
      'ses_e0e5a1YCOSIf',
    ])
  })

  it('looks a harness id up among the runs the view reads, and again while none matches', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ object: 'list', data: [] }))
      .mockResolvedValueOnce(
        Response.json({ object: 'list', data: [run('key_3fa9c2:ses_lookup1'), run('x')] }),
      )
    vi.stubGlobal('fetch', fetchMock)

    expect(await lookUpSession('ses_lookup1')).toEqual([])
    const found = await lookUpSession('ses_lookup1')
    expect(found.map((entry) => entry.session_id)).toEqual(['key_3fa9c2:ses_lookup1'])
    expect(await lookUpSession('ses_lookup1')).toBe(found)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    const url = new URL(String(fetchMock.mock.calls[1][0]), 'http://dashboard.test')
    expect(url.pathname).toBe('/api/router/api/v1/observability/replays/runs')
    expect(url.searchParams.get('search')).toBe('ses_lookup1')
  })
})
