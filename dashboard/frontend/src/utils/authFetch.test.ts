import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  getStoredAuthToken,
  getAuthSessionRevision,
  installAuthenticatedFetch,
  normalizeAuthToken,
  shouldClearSessionForUnauthorized,
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

class MemoryCustomEvent<T> {
  readonly type: string
  readonly detail: T | null

  constructor(type: string, init?: CustomEventInit<T>) {
    this.type = type
    this.detail = init?.detail ?? null
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

describe('authenticated fetch session snapshots', () => {
  let storage: MemoryStorage
  let dispatchedEvents: Event[]

  beforeEach(() => {
    storage = new MemoryStorage()
    dispatchedEvents = []
    vi.stubGlobal('document', { cookie: '' })
    vi.stubGlobal('CustomEvent', MemoryCustomEvent)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  function stubBrowserFetch(fetcher: typeof fetch): void {
    vi.stubGlobal('window', {
      location: {
        origin: 'https://dashboard.example.test',
        protocol: 'https:',
      },
      localStorage: storage,
      fetch: fetcher,
      open: vi.fn(),
      dispatchEvent: (event: Event) => {
        dispatchedEvents.push(event)
        return true
      },
    })
  }

  it('does not let an old in-flight request invalidate a rotated token', async () => {
    let resolveFetch: ((response: Response) => void) | undefined
    const originalFetch = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve
        }),
    ) as unknown as typeof fetch
    stubBrowserFetch(originalFetch)
    storeAuthToken('old-session-token')
    const oldSessionRevision = getAuthSessionRevision()
    installAuthenticatedFetch()

    const pendingRequest = window.fetch('/api/status')
    storeAuthToken('rotated-session-token')
    const rotatedSessionRevision = getAuthSessionRevision()
    resolveFetch?.({ status: 401 } as Response)
    await pendingRequest

    expect(dispatchedEvents).toHaveLength(1)
    expect(shouldClearSessionForUnauthorized(rotatedSessionRevision, dispatchedEvents[0])).toBe(
      false,
    )
    expect(shouldClearSessionForUnauthorized(oldSessionRevision, dispatchedEvents[0])).toBe(true)
  })

  it('invalidates the current cookie-only session after a protected 401', async () => {
    const originalFetch = vi.fn(
      async () => ({ status: 401 }) as Response,
    ) as unknown as typeof fetch
    stubBrowserFetch(originalFetch)
    installAuthenticatedFetch()

    await window.fetch('/api/status')

    expect(dispatchedEvents).toHaveLength(1)
    expect(shouldClearSessionForUnauthorized(getAuthSessionRevision(), dispatchedEvents[0])).toBe(
      true,
    )
  })
})
