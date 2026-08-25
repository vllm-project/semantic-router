import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  assertManagementMe,
  managementOperationRequest,
  managementOperationStream,
  setManagementNamespace,
} from './managementApiContract'

afterEach(() => {
  setManagementNamespace(null)
  vi.unstubAllGlobals()
})

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
    ).toThrow('Me')
    expect(() =>
      assertManagementMe({
        ...identity,
        namespaces: [{ ...identity.namespaces[0], selfServicePolicy: undefined }],
      }),
    ).toThrow('Me')
  })
})

describe('Management event stream transport', () => {
  it('returns the SSE body without decoding or buffering it', async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('event: progress\ndata: {}\n\n'))
        controller.close()
      },
    })
    const response = new Response(stream, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    })
    const json = vi.spyOn(response, 'json')
    const fetchMock = vi.fn().mockResolvedValue(response)
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      managementOperationStream('getAgentSessionsBySessionEvents', {
        pathParameters: { session: 'session-1' },
        query: { afterSequence: 4 },
      }),
    ).resolves.toBe(response)
    expect(json).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/router/management/v1/agent-sessions/session-1/events?afterSequence=4',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({ Accept: 'text/event-stream' }),
      }),
    )
  })

  it('rejects a non-stream operation before opening a transport', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    await expect(managementOperationStream('getUsers')).rejects.toThrow(
      'generated streaming operation',
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('Management text transport', () => {
  it('decodes the generated YAML export as text without attempting JSON', async () => {
    const response = new Response('version: v0.3\n', {
      status: 200,
      headers: { 'Content-Type': 'application/yaml; charset=utf-8' },
    })
    const json = vi.spyOn(response, 'json')
    const text = vi.spyOn(response, 'text')
    const fetchMock = vi.fn().mockResolvedValue(response)
    vi.stubGlobal('fetch', fetchMock)
    setManagementNamespace('namespace-1')

    await expect(
      managementOperationRequest('getRoutingExportsCurrent', {
        headers: {
          Accept: 'application/json',
          'VLLM-SR-Namespace': 'untrusted-namespace',
        } as never,
      }),
    ).resolves.toBe('version: v0.3\n')
    expect(json).not.toHaveBeenCalled()
    expect(text).toHaveBeenCalledOnce()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/router/management/v1/routing/exports/current',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          Accept: 'application/yaml',
          'VLLM-SR-Namespace': 'namespace-1',
        }),
      }),
    )
  })
})

describe('Management empty response transport', () => {
  it('rejects response media metadata on an OpenAPI-declared empty success', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(null, {
          status: 204,
          headers: {
            'Content-Type': 'application/vnd.vllm-semantic-router.management.v1+json',
          },
        }),
      ),
    )

    await expect(
      managementOperationRequest('deleteAccessPoliciesByPolicyId', {
        pathParameters: { policyId: '10000000-0000-4000-8000-000000000001' },
        headers: { 'If-Match': '"access-policy:1"' },
      }),
    ).rejects.toMatchObject({ status: 502 })
  })
})
