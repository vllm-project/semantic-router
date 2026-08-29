import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  CSRF_HEADER_NAME,
  installAuthenticatedFetch,
  normalizeAuthToken,
  UNAUTHORIZED_EVENT,
} from './authFetch'

const ORIGIN = 'https://dashboard.example.test'

describe('auth token normalization', () => {
  it('normalizes bounded cookie-safe tokens', () => {
    expect(normalizeAuthToken(' header.payload.signature ')).toBe('header.payload.signature')
    expect(normalizeAuthToken('')).toBeNull()
    expect(normalizeAuthToken('header payload')).toBeNull()
    expect(normalizeAuthToken('header;payload')).toBeNull()
    expect(normalizeAuthToken(`header\npayload`)).toBeNull()
    expect(normalizeAuthToken('x'.repeat(8193))).toBeNull()
  })
})

describe('csrf header', () => {
  let originalFetch: ReturnType<typeof vi.fn>
  let dispatchEvent: ReturnType<typeof vi.fn>

  function install(cookie: string, status = 200) {
    originalFetch = vi.fn().mockResolvedValue({ status })
    dispatchEvent = vi.fn()
    vi.stubGlobal('window', {
      location: { origin: ORIGIN, protocol: 'https:' },
      fetch: originalFetch,
      open: () => null,
      dispatchEvent,
    })
    vi.stubGlobal('document', { cookie })
    installAuthenticatedFetch()
  }

  async function sentHeaders(input: RequestInfo | URL, init?: RequestInit): Promise<Headers> {
    await window.fetch(input, init)
    return originalFetch.mock.calls[0][1].headers as Headers
  }

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it.each(['POST', 'PUT', 'PATCH', 'DELETE'])('attaches the token on %s', async (method) => {
    install('vsr_csrf=csrf-token-value')

    const headers = await sentHeaders('/api/x', { method })

    expect(headers.get(CSRF_HEADER_NAME)).toBe('csrf-token-value')
  })

  it.each(['GET', 'HEAD'])('sends no token on %s', async (method) => {
    install('vsr_csrf=csrf-token-value')

    const headers = await sentHeaders('/api/x', { method })

    expect(headers.has(CSRF_HEADER_NAME)).toBe(false)
  })

  it('uppercases a lowercase method', async () => {
    install('vsr_csrf=csrf-token-value')

    const headers = await sentHeaders('/api/x', { method: 'post' })

    expect(headers.get(CSRF_HEADER_NAME)).toBe('csrf-token-value')
  })

  it('reads the method from a Request object with no init', async () => {
    install('vsr_csrf=csrf-token-value')

    const headers = await sentHeaders(new Request(`${ORIGIN}/api/x`, { method: 'POST' }))

    expect(headers.get(CSRF_HEADER_NAME)).toBe('csrf-token-value')
  })

  it('lets init.method override a Request object', async () => {
    install('vsr_csrf=csrf-token-value')

    const headers = await sentHeaders(new Request(`${ORIGIN}/api/x`, { method: 'GET' }), {
      method: 'POST',
    })

    expect(headers.get(CSRF_HEADER_NAME)).toBe('csrf-token-value')
  })

  it('sends nothing cross-origin', async () => {
    install('vsr_csrf=csrf-token-value')

    const headers = await sentHeaders('https://other.example/x', { method: 'POST' })

    expect(headers.has(CSRF_HEADER_NAME)).toBe(false)
  })

  it('sends nothing on an unprotected path', async () => {
    install('vsr_csrf=csrf-token-value')

    const headers = await sentHeaders('/login', { method: 'POST' })

    expect(headers.has(CSRF_HEADER_NAME)).toBe(false)
  })

  it('still sends the request when the cookie is missing', async () => {
    install('other=1')

    const headers = await sentHeaders('/api/x', { method: 'POST' })

    expect(headers.has(CSRF_HEADER_NAME)).toBe(false)
    expect(originalFetch).toHaveBeenCalledTimes(1)
  })

  it('ignores an empty cookie value', async () => {
    install('vsr_csrf=')

    const headers = await sentHeaders('/api/x', { method: 'POST' })

    expect(headers.has(CSRF_HEADER_NAME)).toBe(false)
  })

  it('does not match a similarly named cookie', async () => {
    install('vsr_csrf_other=nope')

    const headers = await sentHeaders('/api/x', { method: 'POST' })

    expect(headers.has(CSRF_HEADER_NAME)).toBe(false)
  })

  it('parses the value out of several cookies', async () => {
    install('a=1; vsr_csrf=csrf-token-value; b=2')

    const headers = await sentHeaders('/api/x', { method: 'POST' })

    expect(headers.get(CSRF_HEADER_NAME)).toBe('csrf-token-value')
  })

  it('preserves a header the caller set', async () => {
    install('vsr_csrf=csrf-token-value')

    const headers = await sentHeaders('/api/x', {
      method: 'POST',
      headers: { [CSRF_HEADER_NAME]: 'caller-value' },
    })

    expect(headers.get(CSRF_HEADER_NAME)).toBe('caller-value')
  })

  // The wrapper's second job. Since #2465 there is no stored token to gate this on: a 401
  // from a protected path is itself the signal that the session cookie is gone.
  it('reports a 401 on a protected path', async () => {
    install('vsr_csrf=csrf-token-value', 401)

    await window.fetch('/api/x')

    expect(dispatchEvent).toHaveBeenCalledTimes(1)
    expect(dispatchEvent.mock.calls[0][0]).toMatchObject({ type: UNAUTHORIZED_EVENT })
  })

  it('ignores a 401 from an unprotected path', async () => {
    install('vsr_csrf=csrf-token-value', 401)

    await window.fetch('/login')

    expect(dispatchEvent).not.toHaveBeenCalled()
  })
})

// #2465 removed the four global URL patches. These assert they stay removed: each global
// must be the exact object it was before installAuthenticatedFetch() ran.
describe('browser transports are left alone', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('leaves window.open, WebSocket, EventSource and the iframe prototype untouched', () => {
    const open = () => null
    const WebSocketCtor = function () {} as unknown as typeof window.WebSocket
    const EventSourceCtor = function () {} as unknown as typeof window.EventSource
    const iframePrototype = { setAttribute() {} }
    const originalSetAttribute = iframePrototype.setAttribute

    vi.stubGlobal('window', {
      location: { origin: ORIGIN, protocol: 'https:' },
      fetch: vi.fn().mockResolvedValue({ status: 200 }),
      open,
      WebSocket: WebSocketCtor,
      EventSource: EventSourceCtor,
      HTMLIFrameElement: { prototype: iframePrototype },
    })
    vi.stubGlobal('document', { cookie: '' })

    installAuthenticatedFetch()

    expect(window.open).toBe(open)
    expect(window.WebSocket).toBe(WebSocketCtor)
    expect(window.EventSource).toBe(EventSourceCtor)
    expect(iframePrototype.setAttribute).toBe(originalSetAttribute)
    expect(Object.getOwnPropertyDescriptor(iframePrototype, 'src')).toBeUndefined()
  })
})
