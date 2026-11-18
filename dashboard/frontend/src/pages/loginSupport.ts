import { fallbackRouteTarget } from '../app/routeManifest'

const DASHBOARD_ORIGIN = 'http://vsr-dashboard.local'

export const MIN_PASSWORD_LENGTH = 9

/** How long a finished sign-in waits for the destination route chunk before navigating anyway. */
export const ROUTE_PRELOAD_GRACE_MS = 1500

function isAuthSurface(pathname: string) {
  return (
    pathname === '/login' ||
    pathname.startsWith('/invite/') ||
    pathname === '/auth' ||
    pathname.startsWith('/auth/')
  )
}

/** Accepts only same-origin, non-auth paths so a crafted `from` can never leave the dashboard. */
export function sanitizePostAuthTarget(
  candidate: string | null | undefined,
  fallback: string,
): string {
  if (!candidate || !candidate.startsWith('/') || candidate.startsWith('//')) {
    return fallback
  }

  try {
    const url = new URL(candidate, DASHBOARD_ORIGIN)
    if (url.origin !== DASHBOARD_ORIGIN || isAuthSurface(url.pathname)) {
      return fallback
    }
    return `${url.pathname}${url.search}${url.hash}`
  } catch {
    return fallback
  }
}

/** Setup mode always continues to the setup wizard; otherwise return to where sign-in started. */
export function resolvePostAuthTarget(setupMode: boolean, from?: string | null): string {
  const fallback = fallbackRouteTarget(setupMode)
  return sanitizePostAuthTarget(setupMode ? fallback : (from ?? fallback), fallback)
}

/** Reads a plain-text or JSON (`message`/`error`) error body as a sentence for the form. */
export function formatServerMessage(body: string, fallback: string): string {
  let text = body.trim()
  try {
    const payload = JSON.parse(text) as { message?: unknown; error?: unknown } | null
    const candidate =
      typeof payload?.message === 'string'
        ? payload.message
        : typeof payload?.error === 'string'
          ? payload.error
          : null
    if (candidate) text = candidate.trim()
  } catch {
    // Plain-text bodies are the common case.
  }
  if (!text) return fallback
  return text.charAt(0).toUpperCase() + text.slice(1)
}

export function targetPathname(target: string): string {
  return new URL(target, DASHBOARD_ORIGIN).pathname
}

/** Resolves once the promise settles or the grace period ends, whichever comes first. */
export function settleWithin(promise: Promise<unknown> | undefined, ms: number): Promise<void> {
  if (!promise) return Promise.resolve()
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms)
    void promise.finally(() => {
      clearTimeout(timer)
      resolve()
    })
  })
}
