import { describe, expect, it } from 'vitest'
import {
  fallbackRouteTarget,
  legacyConfigSectionTarget,
  legacyRedirects,
  shellRouteDefinitions,
} from './routeManifest'

describe('dashboard route manifest', () => {
  it('keeps authenticated shell route paths unique', () => {
    const routePaths = shellRouteDefinitions.map((route) => route.path)

    expect(new Set(routePaths).size).toBe(routePaths.length)
  })

  it('registers the model pages under /models', () => {
    expect(shellRouteDefinitions).toContainEqual({ path: '/models/hub', page: 'model-hub' })
    expect(shellRouteDefinitions).toContainEqual({
      path: '/models/decision',
      page: 'decision-models',
    })
    expect(shellRouteDefinitions.map((route) => route.path)).toContain('/models/backends')
  })

  it('redirects every retired path to a registered page', () => {
    const registered = new Set(shellRouteDefinitions.map((route) => route.path))
    for (const redirect of legacyRedirects) {
      const target = redirect.to.split('?')[0]
      expect(registered.has(target), `${redirect.from} -> ${redirect.to}`).toBe(true)
      expect(registered.has(redirect.from), `${redirect.from} is still registered`).toBe(false)
    }
  })

  it('keeps the old decision model tabs reachable', () => {
    expect(legacyRedirects).toContainEqual({
      from: '/decision-model/playground',
      to: '/models/decision?tab=playground',
    })
    expect(legacyRedirects).toContainEqual({
      from: '/decision-model/monitoring',
      to: '/models/decision?tab=monitoring',
    })
  })

  it('maps legacy config sections to settings or routing pages', () => {
    expect(legacyConfigSectionTarget('mcp')).toBe('/settings/integrations')
    expect(legacyConfigSectionTarget('unknown-section')).toBe('/settings/router')
    expect(legacyConfigSectionTarget(undefined)).toBe('/settings/router')
  })

  it('routes unknown paths to setup only while setup mode is active', () => {
    expect(fallbackRouteTarget(true)).toBe('/setup')
    expect(fallbackRouteTarget(false)).toBe('/')
  })
})
