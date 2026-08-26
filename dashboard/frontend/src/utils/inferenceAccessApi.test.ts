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
  it('keeps audit free-text search client-side and sends only supported pagination', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      response({
        data: [
          {
            id: '11111111-1111-4111-8111-111111111111',
            namespaceId: '22222222-2222-4222-8222-222222222222',
            chainSequence: 1,
            actorChain: [],
            action: 'key.disable',
            resourceType: 'api_key',
            resourceId: '33333333-3333-4333-8333-333333333333',
            requestId: '44444444-4444-4444-8444-444444444444',
            outcome: 'allowed',
            reason: 'requested',
            details: {},
            eventHash: 'a'.repeat(64),
            createdAt: '2026-08-25T00:00:00Z',
          },
        ],
        page: { hasMore: false, pageSize: 20 },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      inferenceAccessApi.auditLogs({
        q: '  key.disable  ',
        cursor: 'audit-page-2',
        limit: 20,
        status: 'active',
        includeTotal: true,
      }),
    ).resolves.toMatchObject({
      items: [expect.objectContaining({ action: 'key.disable' })],
    })

    const requestedURL = new URL(String(fetchMock.mock.calls[0][0]), 'http://dashboard.local')
    expect(requestedURL.pathname).toBe('/api/router/management/v1/audit-events')
    expect([...requestedURL.searchParams.entries()]).toEqual([
      ['cursor', 'audit-page-2'],
      ['pageSize', '20'],
    ])
  })
})
