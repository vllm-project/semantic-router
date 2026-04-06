const UNAUTHORIZED_EVENT = 'vsr-auth-unauthorized'

type WrappedFetch = typeof window.fetch & {
  __vsrAuthWrapped?: boolean
}

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
  const compatibleProtocols = new Set([current.protocol])

  if (current.protocol === 'http:') {
    compatibleProtocols.add('ws:')
  }
  if (current.protocol === 'https:') {
    compatibleProtocols.add('wss:')
  }

  if (url.host !== current.host) {
    return false
  }

  return compatibleProtocols.has(url.protocol)
}

function isProtectedPath(url: URL | null): boolean {
  if (!url || !hasCompatibleOrigin(url)) {
    return false
  }

  return url.pathname.startsWith('/api/') || url.pathname.startsWith('/embedded/')
}

export function notifyUnauthorized(): void {
  if (typeof window === 'undefined') {
    return
  }

  window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT))
}

export function installAuthenticatedFetch(): void {
  if (typeof window === 'undefined' || typeof window.fetch !== 'function') {
    return
  }

  const currentFetch = window.fetch as WrappedFetch
  if (currentFetch.__vsrAuthWrapped) {
    return
  }

  const originalFetch = window.fetch.bind(window)
  const wrappedFetch: WrappedFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = getRequestUrl(input)
    const protectedPath = isProtectedPath(url)
    const headers = input instanceof Request ? new Headers(input.headers) : new Headers()
    new Headers(init?.headers).forEach((value, key) => {
      headers.set(key, value)
    })

    const response = await originalFetch(input, {
      ...init,
      headers,
      credentials: protectedPath ? (init?.credentials ?? 'same-origin') : init?.credentials,
    })

    if (protectedPath && response.status === 401) {
      notifyUnauthorized()
    }

    return response
  }) as WrappedFetch

  wrappedFetch.__vsrAuthWrapped = true
  window.fetch = wrappedFetch
}

export { UNAUTHORIZED_EVENT }
