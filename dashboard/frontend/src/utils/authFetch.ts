const UNAUTHORIZED_EVENT = 'vsr-auth-unauthorized'

type WrappedFetch = typeof window.fetch & { __vsrAuthWrapped?: boolean }

function sameOriginProtectedPath(input: RequestInfo | URL): boolean {
  if (typeof window === 'undefined') return false
  try {
    const url =
      input instanceof Request
        ? new URL(input.url, window.location.origin)
        : new URL(input.toString(), window.location.origin)
    return (
      url.origin === window.location.origin &&
      (url.pathname.startsWith('/api/') || url.pathname.startsWith('/embedded/'))
    )
  } catch {
    return false
  }
}

function isSessionProbe(input: RequestInfo | URL): boolean {
  if (typeof window === 'undefined') return false
  try {
    const url =
      input instanceof Request
        ? new URL(input.url, window.location.origin)
        : new URL(input.toString(), window.location.origin)
    return url.origin === window.location.origin && url.pathname === '/api/auth/me'
  } catch {
    return false
  }
}

export function notifyUnauthorized(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT))
  }
}

export function withAuthQuery(path: string): string {
  return path
}

export function installAuthenticatedFetch(): void {
  if (typeof window === 'undefined' || typeof window.fetch !== 'function') return
  const current = window.fetch as WrappedFetch
  if (current.__vsrAuthWrapped) return

  const originalFetch = window.fetch.bind(window)
  const wrapped: WrappedFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const protectedRequest = sameOriginProtectedPath(input)
    const response = await originalFetch(input, {
      ...init,
      credentials: protectedRequest ? 'same-origin' : init?.credentials,
    })
    // Router Management may correctly return 401 for an expired scoped
    // Management session while the Dashboard browser session remains valid.
    // Only the dedicated local-session probe is authoritative for logout.
    if (isSessionProbe(input) && response.status === 401) notifyUnauthorized()
    return response
  }) as WrappedFetch
  wrapped.__vsrAuthWrapped = true
  window.fetch = wrapped
}

export { UNAUTHORIZED_EVENT }
