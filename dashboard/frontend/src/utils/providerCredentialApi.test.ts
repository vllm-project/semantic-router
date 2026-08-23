import { afterEach, describe, expect, it, vi } from 'vitest'

import { MANAGEMENT_API_MEDIA_TYPE } from '../generated/managementApiContract'
import { createProviderCredential } from './providerCredentialApi'

afterEach(() => vi.unstubAllGlobals())

describe('provider credential management seam', () => {
  it('submits the secret only to the credential endpoint and returns metadata', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          data: {
            id: 'credential-1',
            providerId: 'example',
            name: 'Example connection',
            revision: 1,
            status: 'active',
          },
        }),
        { status: 200, headers: { 'Content-Type': MANAGEMENT_API_MEDIA_TYPE } },
      ),
    )
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('crypto', { randomUUID: () => 'idempotency-1' })

    await expect(
      createProviderCredential({
        providerId: 'example',
        catalogRevision: `sha256:${'a'.repeat(64)}`,
        name: 'Example connection',
        secret: 'provider-secret',
      }),
    ).resolves.toMatchObject({ data: { id: 'credential-1' } })

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/router/management/v1/provider-credentials',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('provider-secret'),
        headers: expect.objectContaining({
          'Content-Type': MANAGEMENT_API_MEDIA_TYPE,
          'Idempotency-Key': 'idempotency-1',
        }),
      }),
    )
  })
})
