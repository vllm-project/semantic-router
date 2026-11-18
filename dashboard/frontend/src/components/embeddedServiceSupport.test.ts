import { describe, expect, it } from 'vitest'

import { embedRedirectTarget } from './embeddedServiceSupport'

const origin = 'https://dashboard.example.test'

describe('embedRedirectTarget', () => {
  it('accepts a probe that stays on the proxy path', () => {
    expect(
      embedRedirectTarget(
        { redirected: false, url: `${origin}/embedded/grafana/` },
        '/embedded/grafana/',
        origin,
      ),
    ).toBeNull()
    expect(
      embedRedirectTarget(
        { redirected: true, url: `${origin}/embedded/grafana/login` },
        '/embedded/grafana/',
        origin,
      ),
    ).toBeNull()
  })

  it('names where a probe that left the proxy path landed', () => {
    expect(
      embedRedirectTarget({ redirected: true, url: `${origin}/` }, '/embedded/jaeger/', origin),
    ).toBe('/')
    expect(
      embedRedirectTarget(
        { redirected: true, url: 'https://grafana.example.test/login' },
        '/embedded/grafana/',
        origin,
      ),
    ).toBe('https://grafana.example.test')
  })
})
