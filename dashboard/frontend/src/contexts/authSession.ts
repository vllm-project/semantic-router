export interface AuthUser {
  id: string
  email: string
  name: string
  role?: string
  permissions?: string[]
}

export type AuthSessionRefreshResult =
  | { status: 'authenticated'; user: AuthUser }
  | { status: 'unauthenticated' }
  | { status: 'unavailable'; message: string }

export function hasAuthenticatedSession(token: string | null, user: AuthUser | null): boolean {
  return Boolean(token || user)
}

export async function fetchCurrentAuthUser(
  fetcher: typeof fetch = fetch,
): Promise<AuthSessionRefreshResult> {
  try {
    const response = await fetcher('/api/auth/me', { credentials: 'same-origin' })

    if (response.status === 401) {
      return { status: 'unauthenticated' }
    }

    if (!response.ok) {
      return {
        status: 'unavailable',
        message:
          response.status === 403
            ? 'Session verification was denied. Retry or contact your administrator.'
            : 'Unable to verify your session. Please try again.',
      }
    }

    const payload = (await response.json()) as { user?: AuthUser | null }
    if (!payload?.user?.id) {
      return { status: 'unavailable', message: 'Unable to verify your session. Please try again.' }
    }
    return { status: 'authenticated', user: payload.user }
  } catch {
    return {
      status: 'unavailable',
      message: 'Unable to connect to the Dashboard. Please try again.',
    }
  }
}
