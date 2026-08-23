import { afterEach, describe, expect, it, vi } from 'vitest'

import { installAuthenticatedFetch, UNAUTHORIZED_EVENT, withAuthQuery } from './authFetch'

describe('cookie-backed authenticated fetch', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('never puts a Dashboard credential in a URL', () => {
    expect(withAuthQuery('/embedded/trace?id=one')).toBe('/embedded/trace?id=one')
  })

  it('uses same-origin cookies without adding Authorization', async () => {
    const underlying = vi.fn(async () => new Response(null, { status: 204 }))
    const dispatchEvent = vi.fn()
    vi.stubGlobal('window', {
      location: { origin: 'https://dashboard.example.test' },
      fetch: underlying,
      dispatchEvent,
    })

    installAuthenticatedFetch()
    await window.fetch('/api/settings', { headers: { Accept: 'application/json' } })

    const [, init] = underlying.mock.calls[0] as unknown as [
      RequestInfo | URL,
      RequestInit | undefined,
    ]
    expect(init?.credentials).toBe('same-origin')
    expect(new Headers(init?.headers).has('Authorization')).toBe(false)
    expect(dispatchEvent).not.toHaveBeenCalled()
  })

  it('logs out only when the local session probe is unauthorized', async () => {
    const underlying = vi.fn(async () => new Response(null, { status: 401 }))
    const events: string[] = []
    vi.stubGlobal(
      'CustomEvent',
      class CustomEventStub {
        type: string
        constructor(type: string) {
          this.type = type
        }
      },
    )
    vi.stubGlobal('window', {
      location: { origin: 'https://dashboard.example.test' },
      fetch: underlying,
      dispatchEvent: (event: Event) => events.push(event.type),
    })

    installAuthenticatedFetch()
    await window.fetch('/api/router/management/v1/me')
    expect(events).toEqual([])
    await window.fetch('/api/auth/me')
    expect(events).toEqual([UNAUTHORIZED_EVENT])
  })
})
