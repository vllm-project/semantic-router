import { afterEach, describe, expect, it, vi } from 'vitest'

import { getBuiltInModelCatalog, ModelCatalogApiError } from './modelCatalogApi'

afterEach(() => {
  vi.unstubAllGlobals()
})

const validCatalog = {
  catalogs: [
    {
      catalog_version: 'latest',
      channel: 'latest',
      default_model: 'vllm-sr/mom-v1-blend',
      enabled_models: ['vllm-sr/mom-v1-blend'],
    },
  ],
  models: [
    {
      id: 'vllm-sr/mom-v1-blend',
      display_name: 'MoM V1 Blend',
      description: 'Balanced built-in mixture-of-models policy.',
      kind: 'virtual',
      family: 'mom-v1',
      generation: 1,
      policy_version: '1.0.0',
      entrypoint: 'vllm-sr/mom-v1-blend',
      recipe: 'balance',
      protocols: ['openai_chat'],
      compatible: true,
      compatibility_reason: 'compatible',
      catalog_version: 'latest',
      channel: 'latest',
      enabled_by_default: true,
      default: true,
      verification: {
        status: 'verified',
        authority: 'vllm-sr-maintainers',
        asset_sha256: `sha256:${'a'.repeat(64)}`,
      },
      traits: ['balanced'],
      roles: [
        {
          name: 'balanced',
          required: true,
          minimum_candidates: 1,
          traits: ['chat'],
          recommended_pool: ['local/model'],
        },
      ],
    },
  ],
}

describe('built-in model catalog API', () => {
  it('loads the authenticated read-only catalog endpoint with abort support', async () => {
    const controller = new AbortController()
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(validCatalog), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(getBuiltInModelCatalog(controller.signal)).resolves.toEqual(validCatalog)
    expect(fetchMock).toHaveBeenCalledWith('/api/models/catalog', { signal: controller.signal })
  })

  it('fails closed when the server omits version or model inventory', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () => new Response(JSON.stringify({ catalogs: [], models: [] }), { status: 200 }),
      ),
    )

    await expect(getBuiltInModelCatalog()).rejects.toMatchObject({
      name: 'ModelCatalogApiError',
      status: 502,
    })
  })

  it('fails closed when verification provenance is malformed', async () => {
    const malformed = structuredClone(validCatalog)
    malformed.models[0].verification.asset_sha256 = 'sha256:not-a-digest'
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(malformed), { status: 200 })),
    )

    await expect(getBuiltInModelCatalog()).rejects.toMatchObject({
      name: 'ModelCatalogApiError',
      status: 502,
    })
  })

  it.each([
    ['protocols', (model: Record<string, unknown>) => delete model.protocols],
    [
      'role pool',
      (model: Record<string, unknown>) => {
        const roles = model.roles as Array<Record<string, unknown>>
        roles[0].recommended_pool = []
      },
    ],
  ])('fails closed when required nested %s metadata is malformed', async (_name, mutate) => {
    const malformed = structuredClone(validCatalog)
    mutate(malformed.models[0] as unknown as Record<string, unknown>)
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(malformed), { status: 200 })),
    )

    await expect(getBuiltInModelCatalog()).rejects.toMatchObject({
      name: 'ModelCatalogApiError',
      status: 502,
    })
  })

  it('does not echo an arbitrary backend error body', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response('private backend command and credentials', {
            status: 503,
            statusText: 'Service Unavailable',
          }),
      ),
    )

    const request = getBuiltInModelCatalog()
    await expect(request).rejects.toBeInstanceOf(ModelCatalogApiError)
    await expect(request).rejects.not.toThrow(/private backend|credentials/)
  })
})
