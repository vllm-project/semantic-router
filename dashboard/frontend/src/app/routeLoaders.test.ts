import { describe, expect, it, vi } from 'vitest'

import {
  lazyRoutePage,
  loadDashboardPage,
  loadDecisionModelsWorkspace,
  preloadDashboardRoute,
  resetDashboardRouteLoader,
} from './routeLoaders'

describe('lazy route pages', () => {
  it('keeps one lazy page per loader until the route is reset for a retry', () => {
    const loader = () => Promise.resolve({ default: () => null })
    const first = lazyRoutePage(loader)

    expect(lazyRoutePage(loader)).toBe(first)

    resetDashboardRouteLoader(loader)
    expect(lazyRoutePage(loader)).not.toBe(first)
  })
})

describe('route preloading', () => {
  it('can fetch the requested static route before authentication without requesting protected data', async () => {
    const fetcher = vi.fn()
    vi.stubGlobal('fetch', fetcher)
    try {
      resetDashboardRouteLoader(loadDecisionModelsWorkspace)
      await preloadDashboardRoute('/models/decision')
      expect(fetcher).not.toHaveBeenCalled()
    } finally {
      vi.unstubAllGlobals()
    }
  })
  it('ignores paths outside the dashboard route registry', () => {
    for (const path of [
      '/not-a-dashboard-route',
      '/knowledge-bases/bases',
      '/knowledge-bases/privacy_kb/map',
      '/taxonomy',
    ]) {
      expect(preloadDashboardRoute(path)).toBeUndefined()
    }
  })

  it('deduplicates repeated route preload requests', () => {
    resetDashboardRouteLoader(loadDashboardPage)
    const first = preloadDashboardRoute('/')
    const second = preloadDashboardRoute('/')

    expect(first).toBeDefined()
    expect(second).toBe(first)
  })

  it('allows a failed route boundary to retry through a fresh preload entry', () => {
    resetDashboardRouteLoader(loadDashboardPage)
    const first = preloadDashboardRoute('/')

    resetDashboardRouteLoader(loadDashboardPage)
    const retried = preloadDashboardRoute('/')

    expect(retried).toBeDefined()
    expect(retried).not.toBe(first)
  })
})
