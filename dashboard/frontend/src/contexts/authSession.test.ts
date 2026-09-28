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

    expect(hasAuthenticatedSession(null, user)).toBe(true)
    expect(hasAuthenticatedSession('token', null)).toBe(true)
    expect(hasAuthenticatedSession(null, null)).toBe(false)
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
      status: 'authenticated',
    })
    expect(calls).toEqual([
      {
        input: '/api/auth/me',
        init: { credentials: 'same-origin' },
      },
    ])
  })

  it('reports only a definitive unauthorized response as unauthenticated', async () => {
    const fetcher: typeof fetch = async () => response(401)

    await expect(fetchCurrentAuthUser(fetcher)).resolves.toEqual({
      status: 'unauthenticated',
    })
  })

  it.each([403, 429, 500, 503])(
    'keeps HTTP %i separate from an invalid session',
    async (status) => {
      const fetcher: typeof fetch = async () => response(status)
      await expect(fetchCurrentAuthUser(fetcher)).resolves.toMatchObject({ status: 'unavailable' })
    },
  )

  it('reports network failures without invalidating the session', async () => {
    const fetcher: typeof fetch = async () => {
      throw new TypeError('Failed to fetch')
    }
    await expect(fetchCurrentAuthUser(fetcher)).resolves.toMatchObject({ status: 'unavailable' })
  })

  it.each([null, {}, { user: null }])(
    'does not accept an incomplete successful response',
    async (body) => {
      const fetcher: typeof fetch = async () => response(200, body)
      await expect(fetchCurrentAuthUser(fetcher)).resolves.toMatchObject({ status: 'unavailable' })
    },
  )

  it('treats an unreadable response as unavailable rather than signed out', async () => {
    const fetcher: typeof fetch = async () =>
      ({
        ...response(200),
        json: async () => {
          throw new SyntaxError('Invalid JSON')
        },
      }) as Response
    await expect(fetchCurrentAuthUser(fetcher)).resolves.toMatchObject({ status: 'unavailable' })
  })
})
