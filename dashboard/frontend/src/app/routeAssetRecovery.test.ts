import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { isRouteAssetLoadError, recoverRouteAssetOnce } from './routeAssetRecovery'

describe('route asset recovery', () => {
  beforeEach(() => {
    const values = new Map<string, string>()
    vi.stubGlobal('window', {
      location: { pathname: '/insights' },
      sessionStorage: {
        getItem: (key: string) => values.get(key) || null,
        setItem: (key: string, value: string) => values.set(key, value),
      },
    })
  })
  afterEach(() => vi.unstubAllGlobals())

  it('recognizes stale Vite JavaScript and CSS chunks', () => {
    expect(
      isRouteAssetLoadError(
        new Error('Unable to preload CSS for /assets/insightsPageSupport-D2aHsuGv.css'),
      ),
    ).toBe(true)
    expect(
      isRouteAssetLoadError(new TypeError('Failed to fetch dynamically imported module')),
    ).toBe(true)
    expect(isRouteAssetLoadError(new Error('API request failed'))).toBe(false)
  })

  it('reloads a stale asset only once per route and asset', () => {
    const reload = vi.fn()
    const error = new Error('Unable to preload CSS for /assets/insightsPageSupport-test.css')

    expect(recoverRouteAssetOnce(error, reload)).toBe(true)
    expect(recoverRouteAssetOnce(error, reload)).toBe(false)
    expect(reload).toHaveBeenCalledTimes(1)
  })
})
