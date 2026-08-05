import { afterEach, describe, expect, it, vi } from 'vitest'

import { isInsightsReplayUnavailableError } from './insightsPageApi'
import { fetchAbortableInsightsJSON, isAbortError } from './insightsPageRequestSupport'

describe('insights page request support', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('passes the active abort signal to fetch', async () => {
    const signal = new AbortController().signal
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ count: 2 }),
    }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(fetchAbortableInsightsJSON('/api/insights', 'insights', signal)).resolves.toEqual({
      count: 2,
    })
    expect(fetchMock).toHaveBeenCalledWith('/api/insights', { signal })
  })

  it('preserves replay-unavailable errors and recognizes aborted requests', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      })),
    )

    const request = fetchAbortableInsightsJSON(
      '/api/insights',
      'insights',
      new AbortController().signal,
    )
    await expect(request).rejects.toSatisfy(isInsightsReplayUnavailableError)
    expect(isAbortError({ name: 'AbortError' })).toBe(true)
    expect(isAbortError(new Error('network failed'))).toBe(false)
  })
})
