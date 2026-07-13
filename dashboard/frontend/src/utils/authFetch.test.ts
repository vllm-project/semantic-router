import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  getAuthSessionRevision,
  installAuthenticatedFetch,
  LEGACY_STORAGE_KEY,
  markAuthSessionChanged,
  shouldClearSessionForUnauthorized,
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

describe('cookie-only browser authentication transport', () => {
  let storage: MemoryStorage
  let dispatchedEvents: Event[]

  beforeEach(() => {
    storage = new MemoryStorage()
    dispatchedEvents = []
    vi.stubGlobal('CustomEvent', MemoryCustomEvent)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  function stubBrowserFetch(fetcher: typeof fetch): void {
    vi.stubGlobal('window', {
      location: { origin: 'https://dashboard.example.test', protocol: 'https:' },
      localStorage: storage,
      fetch: fetcher,
      dispatchEvent: (event: Event) => {
        dispatchedEvents.push(event)
        return true
      },
    })
  }

  it('purges a legacy browser credential without writing a script-readable cookie', () => {
    storage.setItem(LEGACY_STORAGE_KEY, 'legacy-session-credential')
    const originalFetch = vi.fn(
      async () => ({ status: 200 }) as Response,
    ) as unknown as typeof fetch
    stubBrowserFetch(originalFetch)
    const cookieBefore = 'unrelated=value'
    vi.stubGlobal('document', { cookie: cookieBefore })

    installAuthenticatedFetch()

    expect(storage.getItem(LEGACY_STORAGE_KEY)).toBeNull()
    expect(document.cookie).toBe(cookieBefore)
  })

  it('initializes cookie authentication when legacy storage is blocked', () => {
    const originalFetch = vi.fn(
      async () => ({ status: 200 }) as Response,
    ) as unknown as typeof fetch
    vi.stubGlobal('window', {
      location: { origin: 'https://dashboard.example.test', protocol: 'https:' },
      get localStorage() {
        throw new DOMException('Storage is disabled', 'SecurityError')
      },
      fetch: originalFetch,
      dispatchEvent: () => true,
    })

    expect(() => installAuthenticatedFetch()).not.toThrow()
    expect(window.fetch).not.toBe(originalFetch)
  })

  it('observes a protected request without adding Authorization or changing its URL', async () => {
    const originalFetch = vi.fn(
      async () => ({ status: 200 }) as Response,
    ) as unknown as typeof fetch
    stubBrowserFetch(originalFetch)
    const init: RequestInit = { headers: { 'X-Request-ID': 'request-1' } }

    installAuthenticatedFetch()
    await window.fetch('/api/status?view=summary', init)

    expect(originalFetch).toHaveBeenCalledWith('/api/status?view=summary', init)
    expect(new Headers(init.headers).has('Authorization')).toBe(false)
  })

  it('does not let an old in-flight request invalidate a newer cookie session', async () => {
    let resolveFetch: ((response: Response) => void) | undefined
    const originalFetch = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve
        }),
    ) as unknown as typeof fetch
    stubBrowserFetch(originalFetch)
    const oldSessionRevision = getAuthSessionRevision()
    installAuthenticatedFetch()

    const pendingRequest = window.fetch('/api/status')
    const rotatedSessionRevision = markAuthSessionChanged()
    resolveFetch?.({ status: 401 } as Response)
    await pendingRequest

    expect(dispatchedEvents).toHaveLength(1)
    expect(shouldClearSessionForUnauthorized(rotatedSessionRevision, dispatchedEvents[0])).toBe(
      false,
    )
    expect(shouldClearSessionForUnauthorized(oldSessionRevision, dispatchedEvents[0])).toBe(true)
  })

  it('invalidates the current cookie session after a protected 401', async () => {
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

  it('ignores a 401 from another origin', async () => {
    const originalFetch = vi.fn(
      async () => ({ status: 401 }) as Response,
    ) as unknown as typeof fetch
    stubBrowserFetch(originalFetch)
    installAuthenticatedFetch()

    await window.fetch('https://api.example.test/status')

    expect(dispatchedEvents).toHaveLength(0)
  })
})
