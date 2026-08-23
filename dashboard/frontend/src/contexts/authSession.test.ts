import { describe, expect, it } from 'vitest'
import { fetchCurrentAuthUser, hasAuthenticatedSession, type AuthUser } from './authSession'

function response(status: number, body?: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response
}

describe('authSession', () => {
  it('treats cookie-backed users as authenticated even without a local token', () => {
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

  it('marks the browser session unauthorized when the server rejects it', async () => {
    const fetcher: typeof fetch = async () => response(401)

    await expect(fetchCurrentAuthUser(fetcher)).resolves.toEqual({
      user: null,
      unauthorized: true,
    })
  })
})
