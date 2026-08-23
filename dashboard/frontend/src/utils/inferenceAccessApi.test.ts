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

  it('hydrates a selected API key without reading credential material', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      response({
        data: {
          keyId: 'key-42',
          name: 'Search result',
          owner: { type: 'user', id: 'user-1' },
          status: 'active',
          revision: 1,
          createdAt: '2026-08-23T00:00:00Z',
          updatedAt: '2026-08-23T00:00:00Z',
        },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(inferenceAccessApi.keySummary('key-42')).resolves.toMatchObject({
      id: 'key-42',
      name: 'Search result',
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/router/management/v1/api-keys/key-42')
  })

  it('creates a key with a typed one-of owner and policy references', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      response({
        data: {
          keyId: 'key-1',
          name: 'Production',
          owner: { type: 'team', id: 'team-1' },
          status: 'active',
          revision: 1,
          createdAt: '2026-08-23T00:00:00Z',
          updatedAt: '2026-08-23T00:00:00Z',
        },
        credential: {
          credentialId: 'credential-1',
          keyId: 'key-1',
          kid: 'vsr_live_123',
          status: 'active',
          revealable: true,
          notBefore: '2026-08-23T00:00:00Z',
          createdAt: '2026-08-23T00:00:00Z',
        },
        secret: 'vsr-secret',
        deliveryExpiresAt: '2026-08-23T00:05:00Z',
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await inferenceAccessApi.createKey({
      name: 'Production',
      ownerType: 'team',
      ownerId: 'team-1',
      accessGroupIds: ['policy-1'],
      budgetId: 'rate-1',
      revision: 0,
    })

    expect(fetchMock.mock.calls[0][0]).toBe('/api/router/management/v1/api-keys')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      name: 'Production',
      owner: { type: 'team', id: 'team-1' },
      revealable: true,
      accessPolicyIds: ['policy-1'],
      rateLimitOverride: { policyId: 'rate-1' },
    })
    expect(fetchMock.mock.calls[0][1].headers).toEqual(
      expect.objectContaining({ 'Idempotency-Key': expect.any(String) }),
    )
  })

  it('creates a key and an ordinary inline quota in one atomic request', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      response({
        data: {
          keyId: 'key-2',
          name: 'Agent key',
          owner: { type: 'user', id: 'user-1' },
          status: 'active',
          revision: 1,
          createdAt: '2026-08-23T00:00:00Z',
          updatedAt: '2026-08-23T00:00:00Z',
        },
        credential: {
          credentialId: 'credential-2',
          keyId: 'key-2',
          kid: 'vsr_live_456',
          status: 'active',
          revealable: true,
          notBefore: '2026-08-23T00:00:00Z',
          createdAt: '2026-08-23T00:00:00Z',
        },
        secret: 'vsr-secret-2',
        deliveryExpiresAt: '2026-08-23T00:05:00Z',
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await inferenceAccessApi.createKey(
      {
        name: 'Agent key',
        ownerType: 'user',
        ownerId: 'user-1',
        accessGroupIds: [],
      },
      {
        name: 'Agent quota',
        description: 'Bounded agent traffic',
        rules: [
          {
            metric: 'cost',
            algorithm: 'sliding_log',
            limit: '5',
            window: 'PT8H',
            accounting: 'response_actual',
            enforcement: 'enforce',
            ordinal: 0,
          },
        ],
      },
    )

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      name: 'Agent key',
      owner: { type: 'user', id: 'user-1' },
      revealable: true,
      accessPolicyIds: [],
      rateLimitOverride: {
        inlinePolicy: {
          name: 'Agent quota',
          description: 'Bounded agent traffic',
          rules: [
            {
              metric: 'cost',
              algorithm: 'sliding_log',
              limit: '5',
              window: 'PT8H',
              accounting: 'response_actual',
              enforcement: 'enforce',
            },
          ],
        },
      },
    })
  })

  it('creates a Team and its complete policy selection in one request', async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/teams') && init?.method === 'POST') {
        return Promise.resolve(
          response({ resource: { type: 'team', id: 'team-1', revision: 1 } }, 201),
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
              { policyId: 'access-1', status: 'active', revision: 1 },
              { policyId: 'access-2', status: 'active', revision: 1 },
            ],
            page: { hasMore: false, pageSize: 200 },
          }),
        )
      }
      if (url.includes('/rate-limit-bindings')) {
        return Promise.resolve(
          response({
            data: [{ policyId: 'budget-1', mode: 'allocation', status: 'active', revision: 1 }],
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

  it('reads effective access and live quota from one Router policy snapshot', async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api-keys/key-1')) {
        return Promise.resolve(
          response({
            data: {
              keyId: 'key-1',
              name: 'Ada',
              owner: { type: 'user', id: 'user-1' },
              status: 'active',
              revision: 1,
              createdAt: '2026-08-23T00:00:00Z',
              updatedAt: '2026-08-23T00:00:00Z',
            },
          }),
        )
      }
      if (url.includes('/credentials')) {
        return Promise.resolve(response({ data: [], page: { hasMore: false, pageSize: 200 } }))
      }
      if (url.includes('/access-policy-bindings') || url.includes('/rate-limit-bindings')) {
        return Promise.resolve(response({ data: [], page: { hasMore: false, pageSize: 200 } }))
      }
      if (url.endsWith('/api-keys/key-1/effective-policy')) {
        return Promise.resolve(
          response({
            subject: { type: 'api_key', id: 'key-1' },
            revision: 8,
            appliedRevision: 8,
            access: {
              grants: [
                {
                  resourceType: 'entrypoint',
                  resourceId: 'blend',
                  permissions: ['discover', 'invoke'],
                  effect: 'allow',
                  source: { subjectType: 'team', subjectId: 'team-1', bindingId: 'binding-1' },
                },
              ],
            },
            quota: {
              meters: [],
              unknownUsageFences: [],
              asOf: '2026-08-23T00:00:00Z',
            },
          }),
        )
      }
      throw new Error(`Unexpected URL: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    await expect(inferenceAccessApi.key('key-1')).resolves.toMatchObject({
      effectiveAccess: [{ resourceType: 'entrypoint', resourceId: 'blend' }],
      accessPolicySources: ['team'],
    })
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toContain(
      '/api/router/management/v1/api-keys/key-1/effective-policy',
    )
  })

  it('creates exact-cost windows without translating them into fixed RPM fields', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        response({ resource: { type: 'rate_limit_policy', id: 'rate-1', revision: 1 } }),
      )
      .mockResolvedValueOnce(
        response({
          data: {
            policyId: 'rate-1',
            name: 'Eight hour spend',
            description: '',
            status: 'active',
            revision: 1,
            rules: [
              {
                ruleId: 'rule-1',
                metric: 'cost',
                algorithm: 'sliding_log',
                limit: '20.000000000000001',
                window: 'PT8H',
                accounting: 'response_actual',
                enforcement: 'enforce',
                ordinal: 0,
              },
            ],
            createdAt: '2026-08-23T00:00:00Z',
            updatedAt: '2026-08-23T00:00:00Z',
          },
        }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await inferenceAccessApi.saveBudget({
      name: 'Eight hour spend',
      enabled: true,
      rules: [
        {
          ruleId: 'rule-1',
          metric: 'cost',
          algorithm: 'sliding_log',
          limit: '20.000000000000001',
          window: 'PT8H',
          accounting: 'response_actual',
          enforcement: 'enforce',
          ordinal: 7,
        },
      ],
    })

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      name: 'Eight hour spend',
      status: 'active',
      rules: [
        {
          ruleId: 'rule-1',
          metric: 'cost',
          algorithm: 'sliding_log',
          limit: '20.000000000000001',
          window: 'PT8H',
          accounting: 'response_actual',
          enforcement: 'enforce',
        },
      ],
    })
  })

  it('creates access groups from exact Router resource identities', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        response({ resource: { type: 'access_policy', id: 'policy-1', revision: 1 } }),
      )
      .mockResolvedValueOnce(
        response({
          data: {
            policyId: 'policy-1',
            name: 'Product models',
            description: '',
            status: 'active',
            revision: 1,
            grants: [
              {
                resourceType: 'entrypoint',
                resourceId: 'blend',
                permission: 'discover',
                effect: 'allow',
              },
              {
                resourceType: 'entrypoint',
                resourceId: 'blend',
                permission: 'invoke',
                effect: 'allow',
              },
            ],
            createdAt: '2026-08-23T00:00:00Z',
            updatedAt: '2026-08-23T00:00:00Z',
          },
        }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await inferenceAccessApi.saveGroup({
      name: 'Product models',
      resources: [{ resourceType: 'entrypoint', resourceId: 'blend' }],
    })

    expect(fetchMock.mock.calls[0][0]).toBe('/api/router/management/v1/access-policies')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      name: 'Product models',
      status: 'active',
      grants: [
        {
          resourceType: 'entrypoint',
          resourceId: 'blend',
          permission: 'discover',
          effect: 'allow',
        },
        {
          resourceType: 'entrypoint',
          resourceId: 'blend',
          permission: 'invoke',
          effect: 'allow',
        },
      ],
    })
  })

  it('uses the exact key usage summary and scopes every supporting series query', async () => {
    const totals = {
      requests: '12',
      successfulRequests: '12',
      inputTokens: '100',
      outputTokens: '40',
      totalTokens: '140',
      incompleteDispatches: '0',
      completeness: 'complete',
      latency: {
        sampleCount: '12',
        totalMilliseconds: '4920',
        averageMilliseconds: 410,
        p50Milliseconds: 250,
        p95Milliseconds: 750,
        p99Milliseconds: 1000,
        percentilesAreEstimated: true,
      },
      ttft: {
        sampleCount: '12',
        totalMilliseconds: '1104',
        averageMilliseconds: 92,
        p50Milliseconds: 64,
        p95Milliseconds: 125,
        p99Milliseconds: 250,
        percentilesAreEstimated: true,
      },
      costs: [
        {
          currency: 'USD',
          knownAmount: '0.0014',
          completeness: 'complete',
          knownDispatches: '12',
          incompleteDispatches: '0',
        },
      ],
    }
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api-keys/key-1/usage')) {
        return Promise.resolve(response({ totals, grain: 'minute', final: true }))
      }
      if (url.includes('/usage/series')) {
        return Promise.resolve(response({ points: [], grain: 'minute', final: true }))
      }
      return Promise.resolve(
        response({ dimension: 'api_key', rows: [], grain: 'minute', final: true }),
      )
    })
    vi.stubGlobal('fetch', fetchMock)

    const usage = await inferenceAccessApi.keyUsage('key-1', {
      from: '2026-08-23T00:00:00Z',
      granularity: 'minute',
    })

    const urls = fetchMock.mock.calls.map(([input]) => String(input))
    const exact = urls.find((url) => url.includes('/api-keys/key-1/usage'))
    expect(exact).not.toContain('apiKeyId=')
    expect(
      urls.filter((url) => url.includes('/usage/series') || url.includes('/usage/breakdowns')),
    ).toHaveLength(5)
    expect(
      urls
        .filter((url) => url.includes('/usage/series') || url.includes('/usage/breakdowns'))
        .every((url) => url.includes('apiKeyId=key-1')),
    ).toBe(true)
    expect(usage.costs[0]).toMatchObject({ currency: 'USD', knownAmount: '0.0014' })
  })

  it('builds overview from bounded statistics plus the existing usage ledger queries', async () => {
    const timing = {
      sampleCount: '0',
      totalMilliseconds: '0',
      averageMilliseconds: 0,
      p50Milliseconds: 0,
      p95Milliseconds: 0,
      p99Milliseconds: 0,
      percentilesAreEstimated: false,
    }
    const totals = {
      requests: '12',
      successfulRequests: '11',
      inputTokens: '100',
      outputTokens: '40',
      totalTokens: '140',
      incompleteDispatches: '0',
      completeness: 'complete',
      costs: [],
      latency: timing,
      ttft: timing,
    }
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/statistics')) {
        return Promise.resolve(
          response({
            asOf: '2026-08-23T00:00:00Z',
            expiringBefore: '2026-09-22T00:00:00Z',
            teams: '3',
            activeApiKeys: '10000',
            expiringApiKeys: '8',
            accessPolicies: '4',
            activeRatePolicies: '5',
          }),
        )
      }
      if (url.includes('/usage/series')) {
        return Promise.resolve(response({ points: [], grain: 'hour', final: true }))
      }
      if (url.includes('/usage/breakdowns')) {
        return Promise.resolve(response({ dimension: 'api_key', rows: [], grain: 'hour' }))
      }
      return Promise.resolve(response({ totals, grain: 'hour', final: true }))
    })
    vi.stubGlobal('fetch', fetchMock)

    await expect(inferenceAccessApi.overview()).resolves.toMatchObject({
      users: null,
      teams: '3',
      activeKeys: '10000',
      requestsToday: 12,
      tokensToday: 140,
    })
    const urls = fetchMock.mock.calls.map(([input]) => String(input))
    expect(urls).toHaveLength(7)
    expect(
      urls.some((url) =>
        /\/(users|teams|api-keys|access-policies|rate-limit-policies)(\?|$)/.test(url),
      ),
    ).toBe(false)
  })

  it('builds self-service team ownership only from the Router /me identity', async () => {
    setManagementNamespace('namespace-1')
    const fetchMock = vi.fn().mockResolvedValue(
      response({
        principal: {
          principalId: 'principal-1',
          displayName: 'Ada',
          kind: 'human',
          status: 'active',
        },
        session: {
          sessionId: 'session-1',
          authenticatedAt: '2026-08-23T00:00:00Z',
          expiresAt: '2026-08-23T01:00:00Z',
          evidenceKind: 'human',
        },
        clusterPermissions: [],
        namespaces: [
          {
            namespace: {
              namespaceId: 'namespace-1',
              name: 'Production',
              status: 'active',
              desiredRevision: 1,
              appliedRevision: 1,
            },
            permissions: ['delegation.use'],
            roleBindings: [],
            user: {
              userId: 'router-user-1',
              email: 'ada@example.com',
              displayName: 'Ada',
              status: 'active',
            },
            teams: [{ teamId: 'team-1', name: 'Platform', role: 'admin', status: 'active' }],
            selfServicePolicy: {
              maxKeysPerUser: 1,
              maxDelegatedSessions: 3,
              delegatedSessionTtlSeconds: 900,
              allowTeamKeyDelegation: true,
              automaticFirstKey: true,
              revision: 1,
            },
          },
        ],
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(inferenceAccessApi.selfTeams()).resolves.toMatchObject({
      members: [{ id: 'router-user-1' }],
      items: [
        {
          id: 'team-1',
          members: [{ teamId: 'team-1', userId: 'router-user-1', role: 'admin' }],
        },
      ],
    })
    expect(fetchMock.mock.calls[0][0]).toBe('/api/router/management/v1/me')
    expect(fetchMock.mock.calls[0][1].headers).not.toHaveProperty('VLLM-SR-Namespace')
  })
})
