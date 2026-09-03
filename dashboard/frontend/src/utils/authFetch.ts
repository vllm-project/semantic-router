// Auth plumbing. The session credential is an HttpOnly cookie (vsr_session) the browser
// attaches to same-origin requests itself, so nothing here holds or forwards it: the
// wrapper only attaches the CSRF token to writes and turns a 401 into an app-wide logout.
// #2465 removed the localStorage copy, the ?authToken= rewriting and four global patches.

const UNAUTHORIZED_EVENT = 'vsr-auth-unauthorized'
const MAX_AUTH_TOKEN_LENGTH = 8192
const UNSAFE_AUTH_TOKEN_CHARS = /[\s;]/
const CSRF_COOKIE_NAME = 'vsr_csrf'
const CSRF_HEADER_NAME = 'X-CSRF-Token'
const UNSAFE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE'])

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

// Not HttpOnly, unlike vsr_session: the page has to read it, and it is not a credential on
// its own. Read fresh every time, because it changes at each login. See #2465.
function readCSRFToken(): string | null {
  if (typeof document === 'undefined') {
    return null
  }

  const prefix = `${CSRF_COOKIE_NAME}=`
  for (const part of document.cookie.split(';')) {
    const entry = part.trim()
    if (entry.startsWith(prefix)) {
      return entry.slice(prefix.length).trim() || null
    }
  }
  return null
}

// init.method wins over a Request's own, matching fetch(request, {method}) semantics.
function requestMethod(input: RequestInfo | URL, init?: RequestInit): string {
  const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
  return method.toUpperCase()
}

// Nothing stores tokens now; AuthContext.login still sanity-checks the login response.
export function normalizeAuthToken(token: string | null | undefined): string | null {
  if (typeof token !== 'string') {
    return null
  }

  const next = token.trim()
  if (
    !next ||
    next.length > MAX_AUTH_TOKEN_LENGTH ||
    UNSAFE_AUTH_TOKEN_CHARS.test(next) ||
    hasControlCharacter(next)
  ) {
    return null
  }

  return next
}

function hasControlCharacter(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index)
    if (code < 32 || code === 127) {
      return true
    }
  }
  return false
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

    // With no cookie, send anyway and let the server answer 403.
    if (
      UNSAFE_METHODS.has(requestMethod(input, init)) &&
      protectedPath &&
      !headers.has(CSRF_HEADER_NAME)
    ) {
      const csrfToken = readCSRFToken()
      if (csrfToken) {
        headers.set(CSRF_HEADER_NAME, csrfToken)
      }
    }

    const response = await originalFetch(input, { ...init, headers })
    if (protectedPath && response.status === 401) {
      notifyUnauthorized()
    }

    return response
  }) as WrappedFetch

  wrappedFetch.__vsrAuthWrapped = true
  window.fetch = wrappedFetch
}

export { CSRF_COOKIE_NAME, CSRF_HEADER_NAME, UNAUTHORIZED_EVENT }
