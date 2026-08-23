import { describe, expect, it } from 'vitest'

import { assertManagementMe } from './managementApiContract'

const identity = {
  principal: { principalId: 'principal-1', displayName: 'Ada', kind: 'human', status: 'active' },
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
        userId: 'user-1',
        email: 'ada@example.com',
        displayName: 'Ada',
        status: 'active',
      },
      teams: [{ teamId: 'team-1', name: 'Platform', role: 'member', status: 'active' }],
      selfServicePolicy: {
        maxKeysPerUser: 1,
        maxDelegatedSessions: 3,
        delegatedSessionTtlSeconds: 900,
        allowTeamKeyDelegation: false,
        automaticFirstKey: true,
        revision: 1,
      },
    },
  ],
}

describe('Management identity contract', () => {
  it('accepts the canonical /me projection', () => {
    expect(assertManagementMe(identity)).toEqual(identity)
  })

  it('rejects noncanonical or malformed identity ownership shapes', () => {
    expect(() =>
      assertManagementMe({
        ...identity,
        namespaces: [{ ...identity.namespaces[0], user: { id: 'dashboard-user-1' } }],
      }),
    ).toThrow('invalid linked Management user')
    expect(() =>
      assertManagementMe({
        ...identity,
        namespaces: [{ ...identity.namespaces[0], selfServicePolicy: undefined }],
      }),
    ).toThrow('invalid Management namespace scope')
  })
})
