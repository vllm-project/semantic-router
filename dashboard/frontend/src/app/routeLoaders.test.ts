import { describe, expect, it } from 'vitest'

import {
  loadAccountSecurityPage,
  loadDashboardPage,
  preloadDashboardRoute,
  resetDashboardRouteLoader,
} from './routeLoaders'

describe('route preloading', () => {
  it('ignores paths outside the dashboard route registry', () => {
    expect(preloadDashboardRoute('/not-a-dashboard-route')).toBeUndefined()
  })

  it('deduplicates repeated route preload requests', () => {
    resetDashboardRouteLoader(loadDashboardPage)
    const first = preloadDashboardRoute('/dashboard')
    const second = preloadDashboardRoute('/dashboard')

    expect(first).toBeDefined()
    expect(second).toBe(first)
  })

  it('allows a failed route boundary to retry through a fresh preload entry', () => {
    resetDashboardRouteLoader(loadDashboardPage)
    const first = preloadDashboardRoute('/dashboard')

    resetDashboardRouteLoader(loadDashboardPage)
    const retried = preloadDashboardRoute('/dashboard')

    expect(retried).toBeDefined()
    expect(retried).not.toBe(first)
  })

  it('preloads the account security route through its focused page boundary', () => {
    resetDashboardRouteLoader(loadAccountSecurityPage)

    expect(preloadDashboardRoute('/account/security')).toBeDefined()
  })
})
