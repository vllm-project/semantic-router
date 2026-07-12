import { normalizeAuthToken } from '../utils/authFetch'

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

export type AuthSessionWriter = (token: string, user?: AuthUser | null) => void

const MIN_EXACT_PASSWORD_REDACTION_LENGTH = 8
export const SAFE_PASSWORD_CHANGE_ERROR =
  'Password change was rejected. Review the password requirements and try again.'

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

export async function readAuthResponseError(response: Response): Promise<string> {
  const body = await response.text()
  if (!body) {
    return `HTTP ${response.status}: ${response.statusText}`
  }

  try {
    const payload = JSON.parse(body) as { message?: unknown; error?: unknown }
    if (typeof payload.message === 'string' && payload.message) {
      return payload.message
    }
    if (typeof payload.error === 'string' && payload.error) {
      return payload.error
    }
  } catch {
    // Plain-text responses are already suitable for display.
  }

  return body
}

function redactSubmittedPasswords(message: string, passwords: readonly string[]): string {
  // Replace longer overlapping values first so redacting a current-password
  // prefix cannot leave the remainder of a reflected new password visible.
  const submittedPasswords = [...new Set(passwords)]
    .filter(Boolean)
    .sort((left, right) => right.length - left.length)
  const reflectedShortPassword = submittedPasswords.some(
    (password) =>
      password.length < MIN_EXACT_PASSWORD_REDACTION_LENGTH && message.includes(password),
  )
  if (reflectedShortPassword) {
    return SAFE_PASSWORD_CHANGE_ERROR
  }

  return submittedPasswords.reduce(
    (redacted, password) => redacted.split(password).join('[REDACTED]'),
    message,
  )
}

export async function changePasswordAndRotateSession(
  currentPassword: string,
  newPassword: string,
  writeSession: AuthSessionWriter,
  fallbackUser: AuthUser | null = null,
  fetcher: typeof fetch = fetch,
): Promise<void> {
  const response = await fetcher('/api/auth/password', {
    method: 'POST',
    credentials: 'same-origin',
    cache: 'no-store',
    redirect: 'error',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ currentPassword, newPassword }),
  })

  if (!response.ok) {
    const message = await readAuthResponseError(response)
    throw new Error(redactSubmittedPasswords(message, [currentPassword, newPassword]))
  }

  const payload = (await response.json()) as { token?: unknown; user?: AuthUser | null }
  const nextToken = normalizeAuthToken(
    typeof payload.token === 'string' ? payload.token : undefined,
  )
  if (!nextToken) {
    throw new Error('Password change response did not include a valid session token')
  }

  writeSession(nextToken, payload.user ?? fallbackUser)
}
