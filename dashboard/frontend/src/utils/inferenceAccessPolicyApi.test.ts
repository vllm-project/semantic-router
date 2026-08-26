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
  it('creates exact-cost windows without translating them into fixed RPM fields', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        response({ resource: { kind: 'rate_limit_policy', id: 'rate-1', revision: 1 } }, 201),
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
      .mockResolvedValueOnce(
        response({ data: [], page: { hasMore: false, pageSize: 1, totalCount: '0' } }),
      )
      .mockResolvedValue(
        response({ data: [], page: { hasMore: false, pageSize: 1, totalCount: '0' } }),
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

    expect(fetchMock).toHaveBeenCalledTimes(3)

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      name: 'Eight hour spend',
      status: 'active',
      rules: [
        {
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
        response({ resource: { kind: 'access_policy', id: 'policy-1', revision: 1 } }, 201),
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
      .mockResolvedValueOnce(
        response({ data: [], page: { hasMore: false, pageSize: 1, totalCount: '0' } }),
      )
      .mockResolvedValue(
        response({ data: [], page: { hasMore: false, pageSize: 1, totalCount: '0' } }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await inferenceAccessApi.saveGroup({
      name: 'Product models',
      resources: [{ resourceType: 'entrypoint', resourceId: 'blend' }],
    })

    expect(fetchMock).toHaveBeenCalledTimes(3)

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

  it('derives policy detail assignment counts from the authoritative binding collection', async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/access-policies/policy-1')) {
        return Promise.resolve(
          response({
            data: {
              policyId: 'policy-1',
              name: 'Developers',
              description: '',
              status: 'active',
              grants: [],
              revision: 1,
              createdAt: '2026-08-23T00:00:00Z',
              updatedAt: '2026-08-23T00:00:00Z',
            },
          }),
        )
      }
      return Promise.resolve(
        response({ data: [], page: { hasMore: false, pageSize: 1, totalCount: '37' } }),
      )
    })
    vi.stubGlobal('fetch', fetchMock)

    await expect(inferenceAccessApi.group('policy-1')).resolves.toMatchObject({
      id: 'policy-1',
      assignmentCount: 37,
    })
    expect(
      fetchMock.mock.calls
        .map(([input]) => String(input))
        .some(
          (url) =>
            url.includes('/access-policy-bindings?') &&
            url.includes('policyId=policy-1') &&
            url.includes('includeTotal=true'),
        ),
    ).toBe(true)
  })
})
