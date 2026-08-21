export interface AuthUser {
  id: string
  email: string
  name: string
  role?: string
  permissions?: string[]
}

export interface AuthSessionRefreshResult {
  user: AuthUser | null
  clearLocalToken: boolean
}

export function hasAuthenticatedSession(token: string | null, user: AuthUser | null): boolean {
  return Boolean(token || user)
}

export async function fetchCurrentAuthUser(
  fetcher: typeof fetch = fetch,
): Promise<AuthSessionRefreshResult> {
  const response = await fetcher('/api/auth/me', { credentials: 'same-origin' })

  if (response.status === 401) {
    return { user: null, clearLocalToken: true }
  }

  if (!response.ok) {
    return { user: null, clearLocalToken: false }
  }

  const payload = (await response.json()) as { user?: AuthUser | null }
  return { user: payload?.user ?? null, clearLocalToken: false }
}
