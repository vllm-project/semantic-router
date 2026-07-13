export interface AuthUser {
  id: string
  email: string
  name: string
  role?: string
  permissions?: string[]
}

export interface AuthSessionRefreshResult {
  user: AuthUser | null
  unauthorized: boolean
}

export type AuthSessionWriter = (user: AuthUser) => void | Promise<void>

// Maintained browser clients explicitly request an HttpOnly-cookie-only auth
// response. Cookie-only is also the server default; a metadata-free non-browser
// client must explicitly request X-VSR-Auth-Mode: bearer when it needs the JWT.
export const COOKIE_AUTH_RESPONSE_HEADERS: Readonly<Record<string, string>> = Object.freeze({
  'X-VSR-Auth-Mode': 'cookie',
})

const MIN_EXACT_PASSWORD_REDACTION_LENGTH = 8
export const SAFE_PASSWORD_CHANGE_ERROR =
  'Password change was rejected. Review the password requirements and try again.'

export function hasAuthenticatedSession(user: AuthUser | null): boolean {
  return Boolean(user)
}

export async function fetchCurrentAuthUser(
  fetcher: typeof fetch = fetch,
): Promise<AuthSessionRefreshResult> {
  const response = await fetcher('/api/auth/me', { credentials: 'same-origin' })

  if (response.status === 401) {
    return { user: null, unauthorized: true }
  }

  if (!response.ok) {
    return { user: null, unauthorized: false }
  }

  const payload = (await response.json()) as { user?: AuthUser | null }
  return { user: payload?.user ?? null, unauthorized: false }
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
      ...COOKIE_AUTH_RESPONSE_HEADERS,
    },
    body: JSON.stringify({ currentPassword, newPassword }),
  })

  if (!response.ok) {
    const message = await readAuthResponseError(response)
    throw new Error(redactSubmittedPasswords(message, [currentPassword, newPassword]))
  }

  const payload = (await response.json()) as { user?: AuthUser | null }
  const nextUser = payload.user ?? fallbackUser
  if (!nextUser) {
    throw new Error('Password change response did not include an authenticated user')
  }

  await writeSession(nextUser)
}
