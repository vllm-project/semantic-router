import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  clearDashboardObservations,
  dashboardObservation,
  invalidateDashboardObservations,
} from './dashboardObservationCache'

describe('authenticated observation cache', () => {
  afterEach(() => {
    clearDashboardObservations()
    vi.unstubAllGlobals()
  })

  it('shares a request for the same API path and erases it on session change', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ model: 'private' }))),
    )
    const first = dashboardObservation<{ model: string }>('/api/cache-session-test', { pollMs: 0 })
    const second = dashboardObservation<{ model: string }>('/api/cache-session-test')
    expect(second).toBe(first)
    const stopA = first.subscribe(vi.fn())
    const stopB = second.subscribe(vi.fn())
    await first.refresh()
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(second.getSnapshot().data?.model).toBe('private')
    clearDashboardObservations()
    expect(first.getSnapshot()).toMatchObject({ data: null, updatedAt: null })
    expect(fetch).toHaveBeenCalledTimes(1)
    invalidateDashboardObservations()
    await first.refresh()
    expect(fetch).toHaveBeenCalledTimes(2)
    stopA()
    stopB()
  })

  it('does not invalidate unrelated observations when only saved configuration changed', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('{}')),
    )
    const config = dashboardObservation('/api/cache-config-test', { pollMs: 0 })
    const status = dashboardObservation('/api/cache-status-test', { pollMs: 0 })
    const stopConfig = config.subscribe(vi.fn())
    const stopStatus = status.subscribe(vi.fn())
    await Promise.all([config.refresh(), status.refresh()])
    invalidateDashboardObservations(['/api/cache-config-test'])
    await config.refresh()
    expect(fetch).toHaveBeenCalledTimes(3)
    expect(status.getSnapshot().stale).toBe(false)
    stopConfig()
    stopStatus()
  })
})
