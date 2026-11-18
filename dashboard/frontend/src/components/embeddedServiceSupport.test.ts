import { describe, expect, it } from 'vitest'

import { embedRedirectTarget, frameStayedOnEmbed } from './embeddedServiceSupport'

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

describe('frameStayedOnEmbed', () => {
  it('accepts a frame on the proxy path', () => {
    expect(frameStayedOnEmbed('/embedded/grafana/d/router', '/embedded/grafana/')).toBe(true)
    expect(frameStayedOnEmbed('/embedded/jaeger/search', '/embedded/jaeger/')).toBe(true)
  })

  it('rejects a frame that left the proxy path or shows the browser error page', () => {
    expect(frameStayedOnEmbed('/', '/embedded/grafana/')).toBe(false)
    expect(frameStayedOnEmbed('/login', '/embedded/jaeger/')).toBe(false)
    expect(frameStayedOnEmbed(null, '/embedded/grafana/')).toBe(false)
  })
})
