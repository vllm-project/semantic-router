import { afterEach, describe, expect, it, vi } from 'vitest'
import { MANAGEMENT_API_MEDIA_TYPE } from '../generated/managementApiContract'

import { setManagementNamespace } from './managementApiContract'
import { inferenceAccessApi } from './inferenceAccessApi'

const response = (payload: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  headers: new Headers({ 'Content-Type': MANAGEMENT_API_MEDIA_TYPE }),
  json: async () => payload,
})

afterEach(() => {
  setManagementNamespace(null)
  vi.unstubAllGlobals()
})

describe('Router Management access client', () => {
  it('uses the canonical Management base and keyset cursor pagination', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(response({ data: [], page: { hasMore: false, pageSize: 20 } }))
    vi.stubGlobal('fetch', fetchMock)

    await inferenceAccessApi.users({ limit: 20, cursor: 'next-page' })

    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/router/management/v1/users?cursor=next-page&pageSize=20',
    )
    expect(fetchMock.mock.calls[0][1]).toMatchObject({
      method: 'GET',
      cache: 'no-store',
      credentials: 'same-origin',
      headers: expect.objectContaining({ Accept: MANAGEMENT_API_MEDIA_TYPE }),
    })
  })

  it('sends bounded collection search with the same opaque selector cursor', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(response({ data: [], page: { hasMore: false, pageSize: 20 } }))
    vi.stubGlobal('fetch', fetchMock)

    await inferenceAccessApi.keys({ q: '  Production  ', limit: 20, cursor: 'selector-page-2' })

    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/router/management/v1/api-keys?cursor=selector-page-2&pageSize=20&search=Production',
    )
  })

  it('creates a Team and its complete policy selection in one request', async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/teams') && init?.method === 'POST') {
        return Promise.resolve(
          response({ resource: { kind: 'team', id: 'team-1', revision: 1 } }, 201),
        )
      }
      if (url.endsWith('/teams/team-1')) {
        return Promise.resolve(
          response({
            data: {
              teamId: 'team-1',
              name: 'Platform',
              description: 'Product inference',
              status: 'active',
              revision: 1,
              createdAt: '2026-08-23T00:00:00Z',
              updatedAt: '2026-08-23T00:00:00Z',
            },
          }),
        )
      }
      if (url.includes('/teams/team-1/members')) {
        return Promise.resolve(response({ data: [], page: { hasMore: false, pageSize: 200 } }))
      }
      if (url.includes('/access-policy-bindings')) {
        return Promise.resolve(
          response({
            data: [
              {
                bindingId: 'access-binding-1',
                policyId: 'access-1',
                subject: { type: 'team', id: 'team-1' },
                status: 'active',
                revision: 1,
                createdAt: '2026-08-23T00:00:00Z',
                updatedAt: '2026-08-23T00:00:00Z',
              },
              {
                bindingId: 'access-binding-2',
                policyId: 'access-2',
                subject: { type: 'team', id: 'team-1' },
                status: 'active',
                revision: 1,
                createdAt: '2026-08-23T00:00:00Z',
                updatedAt: '2026-08-23T00:00:00Z',
              },
            ],
            page: { hasMore: false, pageSize: 200 },
          }),
        )
      }
      if (url.includes('/rate-limit-bindings')) {
        return Promise.resolve(
          response({
            data: [
              {
                bindingId: 'rate-binding-1',
                policyId: 'budget-1',
                subject: { type: 'team', id: 'team-1' },
                mode: 'allocation',
                quotaPartitionId: 'quota-team-1',
                status: 'active',
                revision: 1,
                createdAt: '2026-08-23T00:00:00Z',
                updatedAt: '2026-08-23T00:00:00Z',
              },
            ],
            page: { hasMore: false, pageSize: 200 },
          }),
        )
      }
      throw new Error(`Unexpected request: ${init?.method || 'GET'} ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    await inferenceAccessApi.saveTeam({
      name: 'Platform',
      description: 'Product inference',
      accessGroupIds: ['access-1', 'access-2'],
      budgetId: 'budget-1',
      members: [],
    })

    const createCall = fetchMock.mock.calls.find(
      ([input, init]) => String(input).endsWith('/teams') && init?.method === 'POST',
    )
    if (!createCall?.[1]?.body) throw new Error('Team create request was not sent')
    expect(JSON.parse(String(createCall[1].body))).toEqual({
      name: 'Platform',
      description: 'Product inference',
      accessPolicyIds: ['access-1', 'access-2'],
      rateLimitPolicyId: 'budget-1',
    })
    const postCalls = fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST')
    expect(postCalls).toHaveLength(1)
    expect(String(postCalls[0][0])).toBe('/api/router/management/v1/teams')
    expect(
      fetchMock.mock.calls.some(
        ([input, init]) =>
          init?.method === 'POST' &&
          /\/(access-policy-bindings|rate-limit-bindings)$/.test(String(input)),
      ),
    ).toBe(false)
  })

  it('keeps inference access reads on the Router Management namespace', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(response({ data: [], page: { hasMore: false, pageSize: 100 } }))
    vi.stubGlobal('fetch', fetchMock)
    await inferenceAccessApi.keys({ limit: 100 })
    expect(String(fetchMock.mock.calls[0][0])).toBe(
      '/api/router/management/v1/api-keys?pageSize=100',
    )
  })

  it('loads direct relationship pages with exact totals and cursor continuation', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        response({
          data: [
            {
              teamId: 'team-1',
              userId: 'user-1',
              role: 'admin',
              status: 'active',
              revision: 1,
              createdAt: '2026-08-23T00:00:00Z',
              updatedAt: '2026-08-23T00:00:00Z',
              teamName: 'Platform',
              teamStatus: 'active',
            },
          ],
          page: { hasMore: true, nextCursor: 'page-2', pageSize: 1, totalCount: '2' },
        }),
      )
      .mockResolvedValueOnce(
        response({
          data: [
            {
              teamId: 'team-2',
              userId: 'user-1',
              role: 'member',
              status: 'active',
              revision: 1,
              createdAt: '2026-08-22T00:00:00Z',
              updatedAt: '2026-08-22T00:00:00Z',
              teamName: 'Research',
              teamStatus: 'active',
            },
          ],
          page: { hasMore: false, pageSize: 1 },
        }),
      )
    vi.stubGlobal('fetch', fetchMock)

    const first = await inferenceAccessApi.userMemberships('user-1', { limit: 1 })
    const second = await inferenceAccessApi.userMemberships('user-1', {
      limit: 1,
      cursor: first.nextCursor,
      includeTotal: false,
    })

    expect(first).toMatchObject({ total: 2, hasMore: true, items: [{ teamName: 'Platform' }] })
    expect(second).toMatchObject({ hasMore: false, items: [{ teamName: 'Research' }] })
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/router/management/v1/users/user-1/memberships?pageSize=1&includeTotal=true',
    )
    expect(fetchMock.mock.calls[1][0]).toBe(
      '/api/router/management/v1/users/user-1/memberships?cursor=page-2&pageSize=1',
    )
  })
})
