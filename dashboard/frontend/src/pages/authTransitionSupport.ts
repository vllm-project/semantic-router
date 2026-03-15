export const AUTH_TRANSITION_PATH = '/auth/transition'
export const AUTH_TRANSITION_MIN_DURATION_MS = 3800

const DASHBOARD_ORIGIN = 'http://vsr-dashboard.local'

export function sanitizeAuthTransitionTarget(
  candidate: string | null | undefined,
  fallback: string,
): string {
  if (!candidate || !candidate.startsWith('/') || candidate.startsWith('//')) {
    return fallback
  }

  try {
    const url = new URL(candidate, DASHBOARD_ORIGIN)
    if (url.origin !== DASHBOARD_ORIGIN) {
      return fallback
    }

    const path = `${url.pathname}${url.search}${url.hash}`
    if (
      url.pathname === '/login' ||
      url.pathname === AUTH_TRANSITION_PATH ||
      url.pathname.startsWith(`${AUTH_TRANSITION_PATH}/`)
    ) {
      return fallback
    }

    return path
  } catch {
    return fallback
  }
}

export function resolvePostAuthTarget(setupMode: boolean, from?: string | null): string {
  const fallback = setupMode ? '/setup' : '/dashboard'
  const requestedTarget = setupMode ? '/setup' : from ?? '/dashboard'

  return sanitizeAuthTransitionTarget(requestedTarget, fallback)
}

export function buildAuthTransitionPath(target: string): string {
  const searchParams = new URLSearchParams({ to: target })
  return `${AUTH_TRANSITION_PATH}?${searchParams.toString()}`
}
