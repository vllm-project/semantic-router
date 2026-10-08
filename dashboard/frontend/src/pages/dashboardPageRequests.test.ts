import { describe, expect, it, vi } from 'vitest'

import { fetchDashboardJson, settleDashboardRequests } from './dashboardPageRequests'

describe('fetchDashboardJson', () => {
  it('rejects an HTTP error instead of treating it as empty data', async () => {
    const fetcher = vi.fn(async () => new Response('Failed to read config', { status: 500 }))

    await expect(
      fetchDashboardJson('/api/router/config/all', 'Router config', fetcher),
    ).rejects.toThrow('Router config request failed (HTTP 500)')
    expect(fetcher).toHaveBeenCalledWith('/api/router/config/all')
  })

  it('returns the decoded body of a successful response', async () => {
    const fetcher = vi.fn(async () => Response.json({ overall: 'healthy' }))

    await expect(fetchDashboardJson('/api/status', 'System status', fetcher)).resolves.toEqual({
      overall: 'healthy',
    })
  })
})

describe('settleDashboardRequests', () => {
  it('reports no failure when every request succeeds', async () => {
    await expect(
      settleDashboardRequests([Promise.resolve(), Promise.resolve()]),
    ).resolves.toBeNull()
  })

  it('reports a failure even when the other request succeeds', async () => {
    await expect(
      settleDashboardRequests([
        Promise.reject(new Error('Router config request failed (HTTP 500)')),
        Promise.resolve(),
      ]),
    ).resolves.toBe('Router config request failed (HTTP 500)')
  })

  it('names every failed request', async () => {
    await expect(
      settleDashboardRequests([
        Promise.reject(new Error('Router config request failed (HTTP 500)')),
        Promise.reject(new Error('System status request failed (HTTP 502)')),
      ]),
    ).resolves.toBe(
      'Router config request failed (HTTP 500); System status request failed (HTTP 502)',
    )
  })

  it('falls back to a generic message for non-Error rejections', async () => {
    await expect(settleDashboardRequests([Promise.reject('offline')])).resolves.toBe(
      'Failed to load dashboard data',
    )
  })
})
