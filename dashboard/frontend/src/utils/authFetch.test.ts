import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  CSRF_HEADER_NAME,
  getStoredAuthToken,
  installAuthenticatedFetch,
  normalizeAuthToken,
  storeAuthToken,
  STORAGE_KEY,
} from './authFetch'

class MemoryStorage {
  private readonly values = new Map<string, string>()

  getItem(key: string): string | null {
    return this.values.get(key) ?? null
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value)
  }

  removeItem(key: string): void {
    this.values.delete(key)
  }
}

describe('auth token storage', () => {
  let storage: MemoryStorage

  beforeEach(() => {
    storage = new MemoryStorage()
    vi.stubGlobal('window', {
      location: {
        origin: 'https://dashboard.example.test',
        protocol: 'https:',
      },
      localStorage: storage,
    })
    vi.stubGlobal('document', { cookie: '' })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('normalizes bounded cookie-safe tokens', () => {
    expect(normalizeAuthToken(' header.payload.signature ')).toBe('header.payload.signature')
    expect(normalizeAuthToken('')).toBeNull()
    expect(normalizeAuthToken('header payload')).toBeNull()
    expect(normalizeAuthToken('header;payload')).toBeNull()
    expect(normalizeAuthToken(`header\npayload`)).toBeNull()
    expect(normalizeAuthToken('x'.repeat(8193))).toBeNull()
  })

  it('stores trimmed tokens in localStorage and session cookie', () => {
    const token = storeAuthToken(' header.payload.signature ')

    expect(token).toBe('header.payload.signature')
    expect(storage.getItem(STORAGE_KEY)).toBe('header.payload.signature')
    expect(document.cookie).toBe(
      'vsr_session=header.payload.signature; Path=/; SameSite=Lax; Secure',
    )
  })

  it('clears malformed stored tokens before reuse', () => {
    storage.setItem(STORAGE_KEY, 'bad token')

    expect(getStoredAuthToken()).toBeNull()
    expect(storage.getItem(STORAGE_KEY)).toBeNull()
    expect(document.cookie).toBe('vsr_session=; Path=/; SameSite=Lax; Max-Age=0; Secure')
  })
})

const ORIGIN = 'https://dashboard.example.test'

describe('csrf header', () => {
  let originalFetch: ReturnType<typeof vi.fn>

  function install(cookie: string, storedToken: string | null = 'header.payload.signature') {
    const storage = new MemoryStorage()
    if (storedToken) {
      storage.setItem(STORAGE_KEY, storedToken)
    }
    originalFetch = vi.fn().mockResolvedValue({ status: 200 })
    vi.stubGlobal('window', {
      location: { origin: ORIGIN, protocol: 'https:' },
      localStorage: storage,
      fetch: originalFetch,
      open: () => null,
    })
    vi.stubGlobal('document', { cookie })
    installAuthenticatedFetch()
  }

  async function sentHeaders(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Headers> {
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

  // The token is independent of localStorage, which PR 2 removes entirely.
  it('attaches the token with no stored auth token', async () => {
    install('vsr_csrf=csrf-token-value', null)

    const headers = await sentHeaders('/api/x', { method: 'POST' })

    expect(headers.get(CSRF_HEADER_NAME)).toBe('csrf-token-value')
  })
})
