import { describe, expect, it } from 'vitest'
import {
  fallbackRouteTarget,
  redirectRouteDefinitions,
  shellRouteDefinitions,
} from './routeManifest'

describe('dashboard route manifest', () => {
  it('keeps authenticated shell route paths unique', () => {
    const routePaths = shellRouteDefinitions.map((route) => route.path)

    expect(new Set(routePaths).size).toBe(routePaths.length)
  })

  it('keeps playground in compact fullscreen-friendly shell mode', () => {
    const playgroundRoute = shellRouteDefinitions.find((route) => route.path === '/playground')

    expect(playgroundRoute).toMatchObject({
      hideAccountControl: true,
      hideHeaderOnMobile: true,
      page: 'playground',
    })
  })

  it('keeps legacy redirects pointed at canonical dashboard routes', () => {
    expect(redirectRouteDefinitions).toContainEqual({
      path: '/knowledge-bases',
      to: '/knowledge-bases/bases',
    })
    expect(redirectRouteDefinitions).toContainEqual({
      path: '/taxonomy',
      to: '/knowledge-bases/bases',
    })
    expect(redirectRouteDefinitions).toContainEqual({
      path: '/openclaw',
      to: '/clawos',
    })
  })

  it('routes unknown paths to setup only while setup mode is active', () => {
    expect(fallbackRouteTarget(true)).toBe('/setup')
    expect(fallbackRouteTarget(false)).toBe('/dashboard')
  })
})
