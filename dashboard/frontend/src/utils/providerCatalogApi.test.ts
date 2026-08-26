import { afterEach, describe, expect, it, vi } from 'vitest'

import { MANAGEMENT_API_MEDIA_TYPE } from '../generated/managementApiContract'
import {
  discoverProviderModels,
  getProviderCatalogDetail,
  listProviderCatalog,
} from './providerCatalogApi'

const revision = `sha256:${'a'.repeat(64)}`
const provider = {
  providerId: 'example',
  revision,
  display: {
    name: 'Example',
    description: 'Connect Example models.',
    category: 'Model APIs',
    icon: { source: 'lobe', value: 'example', color: false },
    monogram: 'E',
    accent: '#5b8cff',
  },
  credential: { mode: 'required', label: 'API key' },
  origin: { mode: 'fixed', defaultUrl: 'https://api.example.test/v1', baseUrlRequired: false },
  discoverySupported: true,
  capabilities: ['streaming', 'tools'],
  connectionFields: [
    {
      name: 'region',
      label: 'Region',
      kind: 'select',
      required: false,
      advanced: true,
      options: [{ value: 'global', label: 'Global' }],
    },
  ],
  interfaces: [
    {
      id: 'chat',
      label: 'Chat Completions',
      default: true,
      capabilities: ['streaming', 'tools'],
    },
  ],
}

const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': MANAGEMENT_API_MEDIA_TYPE },
  })

afterEach(() => vi.unstubAllGlobals())

describe('provider catalog management client', () => {
  it('lists providers with keyset, search, and category filters using the vendor media type', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      response({
        data: [provider],
        page: { nextCursor: 'next-page', hasMore: true, pageSize: 40 },
        catalogRevision: revision,
        categories: ['Model APIs'],
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      listProviderCatalog({
        cursor: 'cursor-1',
        pageSize: 40,
        search: 'exa',
        category: 'Model APIs',
        capability: 'tools',
      }),
    ).resolves.toMatchObject({ data: [{ providerId: 'example' }], catalogRevision: revision })

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/router/management/v1/providers?cursor=cursor-1&pageSize=40&search=exa&category=Model+APIs&capability=tools',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({ Accept: MANAGEMENT_API_MEDIA_TYPE }),
      }),
    )
  })

  it('loads provider detail across additive Management response fields', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response({ data: provider, catalogRevision: revision }))
      .mockResolvedValueOnce(
        response({
          data: { ...provider, protocolAdapterId: 'openai.chat.v1' },
          catalogRevision: revision,
        }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await expect(getProviderCatalogDetail('example')).resolves.toMatchObject({
      data: { providerId: 'example' },
    })
    await expect(getProviderCatalogDetail('example')).resolves.toMatchObject({
      data: { providerId: 'example' },
    })
  })

  it('discovers by credential reference and typed fields without sending a raw key', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      response({
        data: [
          {
            catalogItemId: 'item-1',
            providerModelId: 'example-3',
            displayName: 'Example 3',
            capabilities: ['tools'],
          },
          {
            catalogItemId: 'item-2',
            providerModelId: 'example-unknown',
            displayName: 'Example Unknown',
          },
        ],
        page: { hasMore: false, pageSize: 50 },
        catalogRevision: revision,
        discoveryRevision: 'signed-discovery-claim',
        expiresAt: '2026-08-22T12:00:00Z',
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const result = await discoverProviderModels('example', {
      credentialId: 'credential-1',
      connectionFields: { region: 'global', private: false, replicas: 2 },
      search: 'example',
      pageSize: 50,
    })
    expect(result).toMatchObject({
      data: [{ catalogItemId: 'item-1' }, { catalogItemId: 'item-2' }],
    })
    expect(result.data[1]).not.toHaveProperty('capabilities')

    const request = fetchMock.mock.calls[0][1] as RequestInit
    expect(request.headers).toMatchObject({
      Accept: MANAGEMENT_API_MEDIA_TYPE,
      'Content-Type': MANAGEMENT_API_MEDIA_TYPE,
    })
    expect(request.body).toContain('credential-1')
    expect(request.body).not.toContain('apiKey')
    expect(request.body).not.toContain('secret')
    expect(request.body).not.toContain('authHeader')
  })

  it('fails closed on a non-vendor success response', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({ data: [provider], page: { hasMore: false, pageSize: 1 } }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        ),
    )
    await expect(listProviderCatalog()).rejects.toMatchObject({ status: 502 })
  })
})
