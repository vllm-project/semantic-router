import { describe, expect, it } from 'vitest'

import { routerPublicEndpoint, siblingRouterPublicEndpoint } from './routerPublicApi'

describe('Router public API endpoint', () => {
  it('uses the configured browser-reachable origin without a Dashboard proxy prefix', () => {
    expect(routerPublicEndpoint('https://router.example.com/', '/v1/chat/completions')).toBe(
      'https://router.example.com/v1/chat/completions',
    )
    expect(routerPublicEndpoint('https://router.example.com/', '/v1')).toBe(
      'https://router.example.com/v1',
    )
    expect(routerPublicEndpoint('', '/v1/models')).toBe('/v1/models')
    expect(
      siblingRouterPublicEndpoint(
        'https://router.example.com/v1/chat/completions',
        '/v1/router/outcomes',
      ),
    ).toBe('https://router.example.com/v1/router/outcomes')
  })

  it('rejects credentials and non-public API paths', () => {
    expect(() =>
      routerPublicEndpoint('https://user:secret@router.example.com', '/v1/models'),
    ).toThrow('HTTP(S) origin')
    expect(() => routerPublicEndpoint('https://router.example.com/gateway', '/v1/models')).toThrow(
      'without a path',
    )
    expect(() =>
      routerPublicEndpoint('https://router.example.com', '/api/router/v1/models'),
    ).toThrow('must start with /v1')
  })
})
