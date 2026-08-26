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
      response(
        {
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
        },
        201,
      ),
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
      response(
        {
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
        },
        201,
      ),
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
})
