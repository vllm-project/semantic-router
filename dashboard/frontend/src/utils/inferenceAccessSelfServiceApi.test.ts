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
