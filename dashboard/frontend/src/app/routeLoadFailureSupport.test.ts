import { describe, expect, it } from 'vitest'

import {
  getRouteLoadFailureCopy,
  isRouteChunkLoadError,
  routeLoadErrorMessage,
} from './routeLoadFailureSupport'

describe('route load failure support', () => {
  it('recognizes browser-specific lazy chunk failures', () => {
    expect(isRouteChunkLoadError(new Error('Failed to fetch dynamically imported module'))).toBe(
      true,
    )
    expect(isRouteChunkLoadError(new Error('Loading CSS chunk dashboard failed'))).toBe(true)
    expect(
      isRouteChunkLoadError(Object.assign(new Error('route'), { name: 'ChunkLoadError' })),
    ).toBe(true)
    expect(isRouteChunkLoadError(new Error('The route component threw'))).toBe(false)
  })

  it('returns recovery copy tailored to chunk and component failures', () => {
    expect(
      getRouteLoadFailureCopy(
        new Error('Failed to fetch dynamically imported module'),
        'Dashboard',
      ),
    ).toMatchObject({
      eyebrow: 'Route update interrupted',
      title: 'Dashboard needs to reconnect',
    })
    expect(getRouteLoadFailureCopy(new Error('render failed'), 'Dashboard')).toMatchObject({
      eyebrow: 'Route unavailable',
      title: 'Dashboard could not open',
    })
  })

  it('normalizes unknown errors for technical details', () => {
    expect(routeLoadErrorMessage('offline')).toBe('offline')
    expect(routeLoadErrorMessage({ reason: 'offline' })).toBe('Unknown route error')
  })
})
