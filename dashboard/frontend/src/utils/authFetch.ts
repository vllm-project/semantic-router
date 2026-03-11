const STORAGE_KEY = 'vsr_auth_token'
const COOKIE_NAME = 'vsr_session'
const COOKIE_PATH = 'Path=/'
const COOKIE_SAME_SITE = 'SameSite=Lax'
const AUTH_QUERY_PARAM = 'authToken'
const UNAUTHORIZED_EVENT = 'vsr-auth-unauthorized'

type WrappedFetch = typeof window.fetch & {
  __vsrAuthWrapped?: boolean
}

type WrappedWebSocket = typeof window.WebSocket & {
  __vsrAuthWrapped?: boolean
}

type WrappedEventSource = typeof window.EventSource & {
  __vsrAuthWrapped?: boolean
}

type PatchedIframePrototype = HTMLIFrameElement & {
  __vsrAuthSrcWrapped?: boolean
  __vsrAuthSetAttributeWrapped?: boolean
}

type PatchedWindow = Window & {
  __vsrAuthWindowOpenWrapped?: boolean
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

function withToken(url: URL | null): URL | null {
  if (!url) {
    return null
  }

  const token = getStoredAuthToken()
  if (!token || !isProtectedPath(url)) {
    return url
  }

  const next = new URL(url.toString())
  next.searchParams.set(AUTH_QUERY_PARAM, token)
  return next
}

function toProtectedUrlString(input: string | URL): string {
  if (typeof window === 'undefined') {
    return typeof input === 'string' ? input : input.toString()
  }

  const url = input instanceof URL ? input : new URL(input, window.location.origin)
  return withToken(url)?.toString() ?? url.toString()
}

function patchProtectedResourceUrl(value: string): string {
  if (typeof window === 'undefined') {
    return value
  }

  try {
    return toProtectedUrlString(value)
  } catch {
    return value
  }
}

export function getStoredAuthToken(): string | null {
  if (typeof window === 'undefined') {
    return null
  }

  return window.localStorage.getItem(STORAGE_KEY)
}

export function storeAuthToken(token: string): void {
  if (typeof window === 'undefined') {
    return
  }

  window.localStorage.setItem(STORAGE_KEY, token)
  const secure = window.location.protocol === 'https:' ? '; Secure' : ''
  document.cookie = `${COOKIE_NAME}=${token}; ${COOKIE_PATH}; ${COOKIE_SAME_SITE}${secure}`
}

export function clearStoredAuthToken(): void {
  if (typeof window === 'undefined') {
    return
  }

  window.localStorage.removeItem(STORAGE_KEY)
  const secure = window.location.protocol === 'https:' ? '; Secure' : ''
  document.cookie = `${COOKIE_NAME}=; ${COOKIE_PATH}; ${COOKIE_SAME_SITE}; Max-Age=0${secure}`
}

export function notifyUnauthorized(): void {
  if (typeof window === 'undefined') {
    return
  }

  window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT))
}

export function withAuthQuery(path: string): string {
  if (typeof window === 'undefined') {
    return path
  }

  const url = withToken(new URL(path, window.location.origin))
  if (!url) {
    return path
  }

  return `${url.pathname}${url.search}${url.hash}`
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
    const shouldAttachAuth = Boolean(getStoredAuthToken()) && isProtectedPath(url)
    const headers = input instanceof Request ? new Headers(input.headers) : new Headers()
    new Headers(init?.headers).forEach((value, key) => {
      headers.set(key, value)
    })

    if (shouldAttachAuth && !headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${getStoredAuthToken()}`)
    }

    const response = await originalFetch(input, { ...init, headers })
    if (shouldAttachAuth && response.status === 401) {
      notifyUnauthorized()
    }

    return response
  }) as WrappedFetch

  wrappedFetch.__vsrAuthWrapped = true
  window.fetch = wrappedFetch

  installAuthenticatedBrowserTransports()
}

function installAuthenticatedBrowserTransports(): void {
  installAuthenticatedWindowOpen()
  installAuthenticatedWebSocket()
  installAuthenticatedEventSource()
  installAuthenticatedIframe()
}

function installAuthenticatedWindowOpen(): void {
  const patchedWindow = window as PatchedWindow
  if (patchedWindow.__vsrAuthWindowOpenWrapped) {
    return
  }

  const originalOpen = window.open.bind(window)
  window.open = ((url?: string | URL, target?: string, features?: string) => {
    const nextUrl =
      typeof url === 'string' || url instanceof URL
        ? toProtectedUrlString(url)
        : url

    return originalOpen(nextUrl as string | undefined, target, features)
  }) as typeof window.open

  patchedWindow.__vsrAuthWindowOpenWrapped = true
}

function installAuthenticatedWebSocket(): void {
  if (typeof window.WebSocket !== 'function') {
    return
  }

  const CurrentWebSocket = window.WebSocket as WrappedWebSocket
  if (CurrentWebSocket.__vsrAuthWrapped) {
    return
  }

  const OriginalWebSocket = window.WebSocket
  const WrappedConstructor = function (
    this: WebSocket,
    url: string | URL,
    protocols?: string | string[],
  ) {
    const nextUrl = toProtectedUrlString(url)
    return protocols === undefined
      ? new OriginalWebSocket(nextUrl)
      : new OriginalWebSocket(nextUrl, protocols)
  } as unknown as WrappedWebSocket

  Object.assign(WrappedConstructor, OriginalWebSocket, { __vsrAuthWrapped: true })
  WrappedConstructor.prototype = OriginalWebSocket.prototype
  window.WebSocket = WrappedConstructor
}

function installAuthenticatedEventSource(): void {
  if (typeof window.EventSource !== 'function') {
    return
  }

  const CurrentEventSource = window.EventSource as WrappedEventSource
  if (CurrentEventSource.__vsrAuthWrapped) {
    return
  }

  const OriginalEventSource = window.EventSource
  const WrappedConstructor = function (
    this: EventSource,
    url: string | URL,
    eventSourceInitDict?: EventSourceInit,
  ) {
    return new OriginalEventSource(toProtectedUrlString(url), eventSourceInitDict)
  } as unknown as WrappedEventSource

  Object.assign(WrappedConstructor, OriginalEventSource, { __vsrAuthWrapped: true })
  WrappedConstructor.prototype = OriginalEventSource.prototype
  window.EventSource = WrappedConstructor
}

function installAuthenticatedIframe(): void {
  const iframePrototype = window.HTMLIFrameElement?.prototype as PatchedIframePrototype | undefined
  if (!iframePrototype) {
    return
  }

  const srcDescriptor = Object.getOwnPropertyDescriptor(iframePrototype, 'src')
  if (srcDescriptor?.set && !iframePrototype.__vsrAuthSrcWrapped) {
    Object.defineProperty(iframePrototype, 'src', {
      ...srcDescriptor,
      set(value: string) {
        srcDescriptor.set?.call(this, patchProtectedResourceUrl(value))
      },
    })
    iframePrototype.__vsrAuthSrcWrapped = true
  }

  if (!iframePrototype.__vsrAuthSetAttributeWrapped) {
    const originalSetAttribute = iframePrototype.setAttribute
    iframePrototype.setAttribute = function (name: string, value: string) {
      if (name.toLowerCase() === 'src') {
        originalSetAttribute.call(this, name, patchProtectedResourceUrl(value))
        return
      }

      originalSetAttribute.call(this, name, value)
    }
    iframePrototype.__vsrAuthSetAttributeWrapped = true
  }
}

export { AUTH_QUERY_PARAM, STORAGE_KEY, UNAUTHORIZED_EVENT }
