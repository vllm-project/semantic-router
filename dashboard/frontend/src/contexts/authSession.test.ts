import { describe, expect, it, vi } from 'vitest'
import {
  changePasswordAndRotateSession,
  COOKIE_AUTH_RESPONSE_HEADERS,
  fetchCurrentAuthUser,
  hasAuthenticatedSession,
  SAFE_PASSWORD_CHANGE_ERROR,
  type AuthUser,
} from './authSession'

function response(status: number, body?: unknown): Response {
  const serializedBody =
    typeof body === 'string' ? body : body === undefined ? '' : JSON.stringify(body)
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Bad Request',
    json: async () => body,
    text: async () => serializedBody,
  } as Response
}

describe('authSession', () => {
  it('uses the explicit maintained-browser cookie response contract', () => {
    expect(COOKIE_AUTH_RESPONSE_HEADERS).toEqual({ 'X-VSR-Auth-Mode': 'cookie' })
    expect(Object.isFrozen(COOKIE_AUTH_RESPONSE_HEADERS)).toBe(true)
  })

  it('treats a server-resolved cookie-backed user as authenticated', () => {
    const user: AuthUser = {
      id: 'user-1',
      email: 'user@example.test',
      name: 'User One',
    }

    expect(hasAuthenticatedSession(user)).toBe(true)
    expect(hasAuthenticatedSession(null)).toBe(false)
  })

  it('refreshes the current user through the server session cookie path', async () => {
    const calls: Array<{ input: RequestInfo | URL; init?: RequestInit }> = []
    const fetcher: typeof fetch = async (input, init) => {
      calls.push({ input, init })
      return response(200, {
        user: {
          id: 'user-1',
          email: 'user@example.test',
          name: 'User One',
        },
      })
    }

    await expect(fetchCurrentAuthUser(fetcher)).resolves.toEqual({
      user: {
        id: 'user-1',
        email: 'user@example.test',
        name: 'User One',
      },
      unauthorized: false,
    })
    expect(calls).toEqual([
      {
        input: '/api/auth/me',
        init: { credentials: 'same-origin' },
      },
    ])
  })

  it('marks the cookie-backed session unauthorized when the server rejects it', async () => {
    const fetcher: typeof fetch = async () => response(401)

    await expect(fetchCurrentAuthUser(fetcher)).resolves.toEqual({
      user: null,
      unauthorized: true,
    })
  })

  it('posts only the password-change payload and adopts the cookie-backed user', async () => {
    const calls: Array<{ input: RequestInfo | URL; init?: RequestInit }> = []
    const fetcher: typeof fetch = async (input, init) => {
      calls.push({ input, init })
      return response(200, {
        user: {
          id: 'user-1',
          email: 'user@example.test',
          name: 'User One',
        },
      })
    }
    const writeSession = vi.fn()

    await changePasswordAndRotateSession(
      'old-password-value',
      'new-password-value',
      writeSession,
      null,
      fetcher,
    )

    expect(calls).toEqual([
      {
        input: '/api/auth/password',
        init: {
          method: 'POST',
          credentials: 'same-origin',
          cache: 'no-store',
          redirect: 'error',
          headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
            'X-VSR-Auth-Mode': 'cookie',
          },
          body: JSON.stringify({
            currentPassword: 'old-password-value',
            newPassword: 'new-password-value',
          }),
        },
      },
    ])
    expect(writeSession).toHaveBeenCalledWith({
      id: 'user-1',
      email: 'user@example.test',
      name: 'User One',
    })
  })

  it('preserves the precise server policy error without reflecting either password', async () => {
    const fetcher: typeof fetch = async () =>
      response(400, { error: 'Password must not match a recently used password.' })
    const writeSession = vi.fn()

    const request = changePasswordAndRotateSession(
      'old-password-value',
      'new-password-value',
      writeSession,
      null,
      fetcher,
    )

    await expect(request).rejects.toThrow('Password must not match a recently used password.')
    await expect(request).rejects.not.toThrow('old-password-value')
    await expect(request).rejects.not.toThrow('new-password-value')
    expect(writeSession).not.toHaveBeenCalled()
  })

  it('redacts a submitted password if an upstream error accidentally reflects it', async () => {
    const fetcher: typeof fetch = async () =>
      response(400, {
        error: 'The value new-password-value is not allowed; old-password-value was rejected.',
      })

    const request = changePasswordAndRotateSession(
      'old-password-value',
      'new-password-value',
      vi.fn(),
      null,
      fetcher,
    )

    await expect(request).rejects.toThrow(
      'The value [REDACTED] is not allowed; [REDACTED] was rejected.',
    )
  })

  it('redacts the longer value first when submitted passwords overlap', async () => {
    const fetcher: typeof fetch = async () =>
      response(400, {
        error: 'The value current-password-value-with-suffix is not allowed.',
      })

    const request = changePasswordAndRotateSession(
      'current-password-value',
      'current-password-value-with-suffix',
      vi.fn(),
      null,
      fetcher,
    )

    await expect(request).rejects.toThrow('The value [REDACTED] is not allowed.')
    await expect(request).rejects.not.toThrow('with-suffix')
  })

  it('falls back to a fixed safe error when a short password overlaps policy prose', async () => {
    const fetcher: typeof fetch = async () =>
      response(400, { error: 'Password must be at least 15 characters.' })

    const request = changePasswordAndRotateSession('z', 'a', vi.fn(), null, fetcher)

    await expect(request).rejects.toThrow(SAFE_PASSWORD_CHANGE_ERROR)
    await expect(request).rejects.not.toThrow('P[REDACTED]ssword')
  })

  it('preserves the known user when a successful rotation omits the user payload', async () => {
    const existingUser: AuthUser = {
      id: 'user-1',
      email: 'user@example.test',
      name: 'User One',
    }
    const fetcher: typeof fetch = async () => response(200, {})
    const writeSession = vi.fn()

    await changePasswordAndRotateSession(
      'old-password-value',
      'new-password-value',
      writeSession,
      existingUser,
      fetcher,
    )

    expect(writeSession).toHaveBeenCalledWith(existingUser)
  })
})
