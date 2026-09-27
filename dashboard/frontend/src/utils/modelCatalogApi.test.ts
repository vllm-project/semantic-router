import { afterEach, describe, expect, it, vi } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import { getBuiltInModelCatalog, ModelCatalogApiError } from './modelCatalogApi'

afterEach(() => {
  vi.unstubAllGlobals()
})

const validCatalog = generatedCatalog as unknown as Record<string, unknown>

describe('built-in model catalog API transport', () => {
  it('loads the authenticated read-only catalog endpoint with abort support', async () => {
    const controller = new AbortController()
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(validCatalog), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(getBuiltInModelCatalog(controller.signal)).resolves.toEqual(validCatalog)
    expect(fetchMock).toHaveBeenCalledWith('/api/models/catalog', { signal: controller.signal })
  })
})

describe('built-in model catalog API snapshot identity', () => {
  it('keeps virtual-model protocols explicit and SystemOne provider-scoped', () => {
    const protocols = new Set(
      (validCatalog.protocols as Array<Record<string, unknown>>).map((protocol) => protocol.id),
    )
    const models = validCatalog.models as Array<Record<string, unknown>>
    const virtualModels = models.filter((model) => model.kind === 'virtual')

    expect(virtualModels.length).toBeGreaterThan(0)
    expect(
      virtualModels.every(
        (model) =>
          Array.isArray(model.protocols) &&
          model.protocols.length > 0 &&
          model.protocols.every(
            (protocol) => protocol !== 'typesafe/systemone@1' && protocols.has(protocol),
          ),
      ),
    ).toBe(true)
  })

  it('publishes the six Decision cards with an explicit evaluation class', () => {
    const models = validCatalog.models as Array<Record<string, unknown>>
    const decisionModels = models.filter((model) => model.evaluation_class === 'decision')

    expect(decisionModels.map((model) => model.id).sort()).toEqual(
      [
        'llm-semantic-router/decision-1.0-eos-0.8b',
        'llm-semantic-router/decision-1.0-kai-0.6b',
        'llm-semantic-router/decision-1.0-lex-0.6b',
        'llm-semantic-router/decision-1.0-lux-9b',
        'llm-semantic-router/decision-1.0-nox-4b',
        'llm-semantic-router/decision-1.0-sol-2b',
      ].sort(),
    )
  })

  it('rejects unknown and virtual-only evaluation classes', async () => {
    const unknown = structuredClone(validCatalog)
    const unknownModels = unknown.models as Array<Record<string, unknown>>
    unknownModels.find((model) => model.kind === 'physical')!.evaluation_class = 'other'
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(unknown), { status: 200 })),
    )
    await expect(getBuiltInModelCatalog()).rejects.toBeInstanceOf(ModelCatalogApiError)

    const virtual = structuredClone(validCatalog)
    const virtualModels = virtual.models as Array<Record<string, unknown>>
    virtualModels.find((model) => model.kind === 'virtual')!.evaluation_class = 'decision'
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(virtual), { status: 200 })),
    )
    await expect(getBuiltInModelCatalog()).rejects.toBeInstanceOf(ModelCatalogApiError)
  })

  it('rejects an unregistered virtual-model protocol', async () => {
    const malformed = structuredClone(validCatalog)
    const models = malformed.models as Array<Record<string, unknown>>
    const model = models.find((item) => item.kind === 'virtual')!
    model.protocols = ['example/missing@1']
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(malformed), { status: 200 })),
    )

    await expect(getBuiltInModelCatalog()).rejects.toBeInstanceOf(ModelCatalogApiError)
  })

  it.each(['physical', 'virtual'])('rejects an empty %s verification date', async (kind) => {
    const catalog = structuredClone(validCatalog)
    const models = catalog.models as Array<Record<string, unknown>>
    const model = models.find((item) => item.kind === kind)!
    const verification = model.verification as Record<string, unknown>
    verification.verified_at = ''
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(catalog), { status: 200 })),
    )

    await expect(getBuiltInModelCatalog()).rejects.toMatchObject({
      name: 'ModelCatalogApiError',
      status: 502,
    })
  })

  it('accepts undated claimed policies and empty advisory pools without weakening assignments', async () => {
    const catalog = structuredClone(validCatalog)
    const models = catalog.models as Array<Record<string, unknown>>
    const model = models.find((item) => item.kind === 'virtual')!
    const verification = model.verification as Record<string, unknown>
    verification.status = 'claimed'
    delete verification.verified_at
    const roles = model.roles as Array<Record<string, unknown>>
    roles[0].recommended_pool = []
    roles[0].required = true
    roles[0].minimum_candidates = 2
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(catalog), { status: 200 })),
    )

    await expect(getBuiltInModelCatalog()).resolves.toEqual(catalog)
    roles[0].minimum_candidates = 0
    await expect(getBuiltInModelCatalog()).rejects.toBeInstanceOf(ModelCatalogApiError)
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

  it('accepts explicit partial and missing index coverage without a score', async () => {
    const catalog = structuredClone(validCatalog)
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(catalog), { status: 200 })),
    )

    await expect(getBuiltInModelCatalog()).resolves.toEqual(catalog)
  })

  it.each([
    ['a partial result with a score', 'partial', 42, undefined],
    ['an available result without a score', 'available', null, undefined],
    ['a partial result with zero coverage', 'partial', null, 0],
    ['a missing result with nonzero coverage', 'missing', null, 0.5],
  ])('rejects %s', async (_name, status, score, coverage) => {
    const malformed = structuredClone(validCatalog)
    const results = malformed.index_results as Array<Record<string, unknown>>
    results[0].status = status
    results[0].score = score
    if (coverage !== undefined) results[0].coverage = coverage
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(malformed))),
    )

    await expect(getBuiltInModelCatalog()).rejects.toBeInstanceOf(ModelCatalogApiError)
  })
})

describe('built-in model catalog API evaluation records', () => {
  it.each(['missing', 'failed', 'withheld', 'not_applicable'])(
    'accepts %s evaluation records without fabricated metrics',
    async (status) => {
      const sparse = structuredClone(validCatalog)
      const evaluations = sparse.evaluations as Array<Record<string, unknown>>
      evaluations[0].status = status
      evaluations[0].metrics = {}
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => new Response(JSON.stringify(sparse), { status: 200 })),
      )

      await expect(getBuiltInModelCatalog()).resolves.toEqual(sparse)
    },
  )

  it('rejects an available evaluation without metrics', async () => {
    const malformed = structuredClone(validCatalog)
    const evaluations = malformed.evaluations as Array<Record<string, unknown>>
    evaluations[0].status = 'available'
    evaluations[0].metrics = {}
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
    ['without a calendar anchor', undefined],
    ['with an invalid calendar anchor', '2026-09-31'],
  ])('rejects an available evaluation %s', async (_name, observedAt) => {
    const malformed = structuredClone(validCatalog)
    const evaluations = malformed.evaluations as Array<Record<string, unknown>>
    delete evaluations[0].measured_at
    if (observedAt === undefined) delete evaluations[0].observed_at
    else evaluations[0].observed_at = observedAt
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(malformed), { status: 200 })),
    )

    await expect(getBuiltInModelCatalog()).rejects.toMatchObject({
      name: 'ModelCatalogApiError',
      status: 502,
    })
  })

  it('accepts a valid measured date as the available evaluation anchor', async () => {
    const measured = structuredClone(validCatalog)
    const evaluations = measured.evaluations as Array<Record<string, unknown>>
    delete evaluations[0].observed_at
    evaluations[0].measured_at = '2026-09-06'
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(measured), { status: 200 })),
    )

    await expect(getBuiltInModelCatalog()).resolves.toEqual(measured)
  })
})

describe('built-in model catalog API nested metadata', () => {
  it.each([
    [
      'protocol operations',
      (payload: Record<string, unknown>) => {
        const protocols = payload.protocols as Array<Record<string, unknown>>
        protocols[0].operations = []
      },
    ],
    [
      'protocol default base path',
      (payload: Record<string, unknown>) => {
        const protocols = payload.protocols as Array<Record<string, unknown>>
        delete protocols[0].default_base_path
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
      'reasoning activation parameter',
      (payload: Record<string, unknown>) => {
        const families = payload.reasoning_families as Array<Record<string, unknown>>
        families[0].activation_parameter = families[0].parameter
      },
    ],
    [
      'reasoning effort flags',
      (payload: Record<string, unknown>) => {
        const families = payload.reasoning_families as Array<Record<string, unknown>>
        const family = families.find((candidate) => candidate.effort_flags !== undefined)
        family!.effort_flags = { invented: 'low_effort' }
      },
    ],
    [
      'provider featured presentation flag',
      (payload: Record<string, unknown>) => {
        const providers = payload.providers as Array<Record<string, unknown>>
        const presentation = providers[0].presentation as Record<string, unknown>
        presentation.featured = 'yes'
      },
    ],
    [
      'model access relationship',
      (payload: Record<string, unknown>) => {
        const providers = payload.providers as Array<Record<string, unknown>>
        const provider = providers.find(
          (candidate) => Array.isArray(candidate.models) && candidate.models.length > 0,
        )
        const models = provider?.models as Array<Record<string, unknown>>
        models[0].relationship = 'brokered'
      },
    ],
    [
      'protocol-specific reasoning efforts',
      (payload: Record<string, unknown>) => {
        const providers = payload.providers as Array<Record<string, unknown>>
        const provider = providers.find((candidate) => candidate.id === 'openai')!
        const models = provider.models as Array<Record<string, unknown>>
        const astra = models.find((candidate) => candidate.catalog === 'openai/gpt-6-astra')!
        astra.reasoning_efforts_by_protocol = {
          'anthropic/messages@1': ['low'],
        }
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
})

describe('built-in model catalog API benchmark presentation metadata', () => {
  it.each([
    [
      'duplicate tags',
      (benchmarks: Array<Record<string, unknown>>) => {
        const benchmark = benchmarks.find((candidate) => Array.isArray(candidate.tags))!
        benchmark.tags = ['core', 'core']
      },
    ],
    [
      'empty tags',
      (benchmarks: Array<Record<string, unknown>>) => {
        const benchmark = benchmarks.find((candidate) => Array.isArray(candidate.tags))!
        benchmark.tags = []
      },
    ],
    [
      'invalid metric normalization',
      (benchmarks: Array<Record<string, unknown>>) => {
        const benchmark = benchmarks.find((candidate) => {
          const metrics = candidate.metrics as Array<Record<string, unknown>>
          return metrics.some((metric) => metric.normalization !== undefined)
        })!
        const metrics = benchmark.metrics as Array<Record<string, unknown>>
        const metric = metrics.find((candidate) => candidate.normalization !== undefined)!
        metric.normalization = { type: 'linear_clamp', min: 1, max: 0 }
      },
    ],
  ])('rejects %s', async (_name, mutate) => {
    const malformed = structuredClone(validCatalog)
    const benchmarks = malformed.benchmarks as Array<Record<string, unknown>>
    mutate(benchmarks)
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(malformed), { status: 200 })),
    )

    await expect(getBuiltInModelCatalog()).rejects.toMatchObject({
      name: 'ModelCatalogApiError',
      status: 502,
    })
  })
})

describe('built-in model catalog API transport failures', () => {
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
