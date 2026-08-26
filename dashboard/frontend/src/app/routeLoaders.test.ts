import { describe, expect, it } from 'vitest'

import {
  loadAccessControlPage,
  loadDashboardPage,
  loadOpenClawPage,
  preloadDashboardRoute,
  resetDashboardRouteLoader,
} from './routeLoaders'

describe('route preloading', () => {
  it('ignores paths outside the dashboard route registry', () => {
    expect(preloadDashboardRoute('/not-a-dashboard-route')).toBeUndefined()
    expect(preloadDashboardRoute('/plugins/context-compression')).toBeUndefined()
  })

  it('deduplicates repeated route preload requests', () => {
    resetDashboardRouteLoader(loadDashboardPage)
    const first = preloadDashboardRoute('/dashboard')
    const second = preloadDashboardRoute('/dashboard')

    expect(first).toBeDefined()
    expect(second).toBe(first)
  })

  it('preloads OpenClaw from its canonical product route', () => {
    resetDashboardRouteLoader(loadOpenClawPage)

    expect(preloadDashboardRoute('/openclaw')).toBeDefined()
  })

  it('preloads Request Logs through the Access surface', () => {
    resetDashboardRouteLoader(loadAccessControlPage)

    const logs = preloadDashboardRoute('/logs')
    const access = preloadDashboardRoute('/access/usage')

    expect(logs).toBeDefined()
    expect(logs).toBe(access)
  })

  it('allows a failed route boundary to retry through a fresh preload entry', () => {
    resetDashboardRouteLoader(loadDashboardPage)
    const first = preloadDashboardRoute('/dashboard')

    resetDashboardRouteLoader(loadDashboardPage)
    const retried = preloadDashboardRoute('/dashboard')

    expect(retried).toBeDefined()
    expect(retried).not.toBe(first)
  })
})
