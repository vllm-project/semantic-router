import { afterEach, describe, expect, it, vi } from 'vitest'

import generatedCatalog from '../generated/modelCatalog.json'
import { getBuiltInModelCatalog, ModelCatalogApiError } from './modelCatalogApi'

afterEach(() => {
  vi.unstubAllGlobals()
})

const validCatalog = generatedCatalog as unknown as Record<string, unknown>

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
    const models = malformed.models as Array<Record<string, unknown>>
    const verification = models[0].verification as Record<string, unknown>
    verification.asset_sha256 = 'sha256:not-a-digest'
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
    [
      'protocol operations',
      (payload: Record<string, unknown>) => {
        const protocols = payload.protocols as Array<Record<string, unknown>>
        protocols[0].operations = []
      },
    ],
    [
      'reasoning levels',
      (payload: Record<string, unknown>) => {
        const families = payload.reasoning_families as Array<Record<string, unknown>>
        families[0].levels = []
      },
    ],
    [
      'index normalization',
      (payload: Record<string, unknown>) => {
        const indices = payload.indices as Array<Record<string, unknown>>
        const components = indices[0].components as Array<Record<string, unknown>>
        components[0].normalization = { type: 'linear_clamp', min: 1, max: 0 }
      },
    ],
  ])('fails closed when required nested %s metadata is malformed', async (_name, mutate) => {
    const malformed = structuredClone(validCatalog)
    mutate(malformed)
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
