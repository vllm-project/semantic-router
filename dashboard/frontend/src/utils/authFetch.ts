const LEGACY_STORAGE_KEY = ['vsr', 'auth', 'token'].join('_')
const UNAUTHORIZED_EVENT = 'vsr-auth-unauthorized'

type WrappedFetch = typeof window.fetch & {
  __vsrAuthObserved?: boolean
}

interface UnauthorizedEventDetail {
  requestRevision: number
}

// The revision contains no credential material. It only prevents a late 401
// from an old request from clearing a newer cookie-backed session.
let authSessionRevision = 0

function getRequestUrl(input: RequestInfo | URL): URL | null {
  if (typeof window === 'undefined') {
    return null
  }

  if (input instanceof URL) {
    return input
  }

  if (typeof input === 'string') {
    return new URL(input, window.location.origin)
  }

  if (input instanceof Request) {
    return new URL(input.url, window.location.origin)
  }

  return null
}

function hasCompatibleOrigin(url: URL | null): boolean {
  if (!url || typeof window === 'undefined') {
    return false
  }

  const current = new URL(window.location.origin)
  return url.protocol === current.protocol && url.host === current.host
}

function isProtectedPath(url: URL | null): boolean {
  if (!url || !hasCompatibleOrigin(url)) {
    return false
  }

  return url.pathname.startsWith('/api/') || url.pathname.startsWith('/embedded/')
}

function clearLegacyStoredCredential(): void {
  if (typeof window === 'undefined') {
    return
  }
  try {
    window.localStorage?.removeItem(LEGACY_STORAGE_KEY)
  } catch {
    // Storage can be disabled by browser policy. Cookie authentication must
    // still initialize when the legacy storage area is unavailable.
  }
}

export function getAuthSessionRevision(): number {
  return authSessionRevision
}

export function markAuthSessionChanged(): number {
  authSessionRevision += 1
  return authSessionRevision
}

export function notifyUnauthorized(requestRevision: number = getAuthSessionRevision()): void {
  if (typeof window === 'undefined') {
    return
  }

  window.dispatchEvent(
    new CustomEvent<UnauthorizedEventDetail>(UNAUTHORIZED_EVENT, {
      detail: { requestRevision },
    }),
  )
}

export function shouldClearSessionForUnauthorized(currentRevision: number, event: Event): boolean {
  const detail = (event as CustomEvent<unknown>).detail
  if (!detail || typeof detail !== 'object' || !('requestRevision' in detail)) {
    return false
  }

  const requestRevision = (detail as UnauthorizedEventDetail).requestRevision
  if (!Number.isSafeInteger(requestRevision) || requestRevision < 0) {
    return false
  }
  return requestRevision === currentRevision
}

// Browser authentication is cookie-only. This wrapper observes protected 401s
// but deliberately leaves request headers and URLs untouched.
export function installAuthenticatedFetch(): void {
  if (typeof window === 'undefined' || typeof window.fetch !== 'function') {
    return
  }

  clearLegacyStoredCredential()

  const currentFetch = window.fetch as WrappedFetch
  if (currentFetch.__vsrAuthObserved) {
    return
  }

  const originalFetch = window.fetch.bind(window)
  const wrappedFetch: WrappedFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const protectedRequest = isProtectedPath(getRequestUrl(input))
    const requestRevision = getAuthSessionRevision()
    const response = await originalFetch(input, init)
    if (protectedRequest && response.status === 401) {
      notifyUnauthorized(requestRevision)
    }
    return response
  }) as WrappedFetch

  wrappedFetch.__vsrAuthObserved = true
  window.fetch = wrappedFetch
}

export { LEGACY_STORAGE_KEY, UNAUTHORIZED_EVENT }
