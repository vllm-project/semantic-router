import { afterEach, describe, expect, it, vi } from 'vitest'

import { MANAGEMENT_API_MEDIA_TYPE } from '../generated/managementApiContract'
import { createProviderCredential } from './providerCredentialApi'

afterEach(() => vi.unstubAllGlobals())

const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': MANAGEMENT_API_MEDIA_TYPE },
  })

describe('provider credential management seam', () => {
  it('submits the secret only to the credential endpoint and resolves canonical metadata', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        response(
          { resource: { kind: 'provider_credential', id: 'credential-1', revision: 1 } },
          201,
        ),
      )
      .mockResolvedValueOnce(
        response({
          data: {
            credentialId: 'credential-1',
            providerId: 'example',
            catalogRevision: `sha256:${'a'.repeat(64)}`,
            name: 'Example connection',
            normalizedOrigin: 'https://api.example.test/v1',
            revision: 1,
            status: 'active',
            createdAt: '2026-08-25T00:00:00Z',
            updatedAt: '2026-08-25T00:00:00Z',
          },
        }),
      )
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('crypto', { randomUUID: () => 'idempotency-1' })

    await expect(
      createProviderCredential({
        providerId: 'example',
        name: 'Example connection',
        secret: 'provider-secret',
      }),
    ).resolves.toMatchObject({ data: { credentialId: 'credential-1' } })

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/router/management/v1/provider-credentials',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          providerId: 'example',
          name: 'Example connection',
          secret: 'provider-secret',
        }),
        headers: expect.objectContaining({
          'Content-Type': MANAGEMENT_API_MEDIA_TYPE,
          'Idempotency-Key': 'idempotency-1',
        }),
      }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/router/management/v1/provider-credentials/credential-1',
      expect.objectContaining({ method: 'GET' }),
    )
  })
})
