import { afterEach, describe, expect, it, vi } from 'vitest'

import { setManagementNamespace } from './managementApiContract'
import { routingManagementApi } from './routingManagementApi'

const mediaType = 'application/vnd.vllm-semantic-router.management.v1+json'

const testRequest = (input: RequestInfo | URL, init?: RequestInit) =>
  new Request(new URL(String(input), 'http://dashboard.test'), init)

const model = (id: string) => ({
  id,
  name: `local/${id}`,
  status: 'active',
  revision: 1,
  modelRevision: 1,
  catalogRevision: `sha256:${'a'.repeat(64)}`,
  aliases: [],
  paramSize: '32B',
  contextWindowSize: 131072,
  description: 'A connected model.',
  capabilities: ['text'],
  loras: [],
  qualityScore: 0.9,
  modality: 'text',
  tags: ['general'],
  control: {
    retry: { count: 0, on: [] },
    timeout: { request: '30s', stream: '60s' },
  },
  pricing: {
    inputCostPerMillionTokens: null,
    outputCostPerMillionTokens: null,
    cacheReadCostPerMillionTokens: null,
    cacheWriteCostPerMillionTokens: null,
  },
  backends: [
    {
      providerId: 'private',
      providerModelId: id,
      credentialConfigured: false,
      weight: '1',
    },
  ],
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
})

const modelCard = (id: string) => ({
  id,
  name: `local/${id}`,
  card: {
    aliases: [],
    capabilities: ['text'],
    reasoning: { type: 'reasoning', efforts: ['high'] },
    loras: [],
    tags: ['balanced'],
  },
})

afterEach(() => {
  setManagementNamespace(null)
  vi.unstubAllGlobals()
})

describe('routingManagementApi', () => {
  it('exports the portable manifest through the generated text operation', async () => {
    const response = new Response('version: v0.3\n', {
      status: 200,
      headers: { 'Content-Type': 'application/yaml' },
    })
    const json = vi.spyOn(response, 'json')
    const request = vi.fn().mockResolvedValue(response)
    vi.stubGlobal('fetch', request)

    await expect(routingManagementApi.exportCurrentManifest()).resolves.toBe('version: v0.3\n')
    expect(json).not.toHaveBeenCalled()
    expect(
      testRequest(request.mock.calls[0][0], request.mock.calls[0][1]).headers.get('Accept'),
    ).toBe('application/yaml')
  })

  it('loads semantic Model Cards without a backend or runtime projection', async () => {
    const request = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(
          JSON.stringify({
            data: [modelCard('model-a')],
            page: { hasMore: false, pageSize: 100 },
          }),
          { status: 200, headers: { 'Content-Type': mediaType } },
        ),
    )
    vi.stubGlobal('fetch', request)

    await expect(routingManagementApi.listModelCards()).resolves.toEqual([modelCard('model-a')])
    expect(new URL(testRequest(request.mock.calls[0][0]).url).pathname).toBe(
      '/api/router/management/v1/routing/model-cards',
    )
  })

  it('accepts immutable Router distribution Recipes with complete provenance', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              data: [
                {
                  id: 'rcp_builtin',
                  name: 'Built-in',
                  status: 'draft',
                  revision: 1,
                  recipeRevision: 1,
                  origin: 'distribution',
                  immutable: true,
                  provenance: {
                    distributionId: 'mom-v1',
                    distributionVersion: '1.2.0',
                    assetDigest: `sha256:${'a'.repeat(64)}`,
                    sourceRecipeId: 'rcp_balance',
                    sourceRevision: 1,
                    recipeDigest: `sha256:${'b'.repeat(64)}`,
                    installedAt: '2026-08-23T00:00:00Z',
                  },
                  decisions: [
                    {
                      id: 'dec_simple',
                      name: 'Simple',
                      dispatchCardinality: 'single',
                    },
                  ],
                  document: { signals: {}, projections: {}, decisions: [] },
                  createdAt: '2026-08-23T00:00:00Z',
                  updatedAt: '2026-08-23T00:00:00Z',
                },
              ],
              page: { hasMore: false, pageSize: 100 },
            }),
            { status: 200, headers: { 'Content-Type': mediaType } },
          ),
      ),
    )

    await expect(routingManagementApi.listRecipes()).resolves.toMatchObject([
      {
        origin: 'distribution',
        immutable: true,
        provenance: { sourceRecipeId: 'rcp_balance' },
      },
    ])
  })

  it('follows Router cursors without falling back to a Dashboard store', async () => {
    setManagementNamespace('namespace-1')
    const requests: Request[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const request = testRequest(input, init)
        requests.push(request)
        const cursor = new URL(request.url).searchParams.get('cursor')
        return new Response(
          JSON.stringify({
            data: [model(cursor ? 'model-b' : 'model-a')],
            page: cursor
              ? { hasMore: false, pageSize: 100 }
              : { nextCursor: 'cursor-2', hasMore: true, pageSize: 100 },
          }),
          { status: 200, headers: { 'Content-Type': mediaType } },
        )
      }),
    )

    await expect(routingManagementApi.listModels()).resolves.toHaveLength(2)
    expect(requests).toHaveLength(2)
    expect(requests[0].headers.get('VLLM-SR-Namespace')).toBe('namespace-1')
    expect(new URL(requests[1].url).searchParams.get('cursor')).toBe('cursor-2')
  })

  it.each(['1e-3', '1000000.000000001'])(
    'rejects Model price responses outside the exact decimal contract: %s',
    async (inputCostPerMillionTokens) => {
      const valid = model('model-a')
      const invalid = {
        ...valid,
        pricing: { ...valid.pricing, inputCostPerMillionTokens },
      }
      vi.stubGlobal(
        'fetch',
        vi.fn(
          async () =>
            new Response(
              JSON.stringify({ data: [invalid], page: { hasMore: false, pageSize: 100 } }),
              { status: 200, headers: { 'Content-Type': mediaType } },
            ),
        ),
      )

      await expect(routingManagementApi.listModels()).rejects.toThrow('RoutingModelPage')
    },
  )

  it('rejects Model control responses outside the exact duration contract', async () => {
    const valid = model('model-a')
    const invalid = {
      ...valid,
      control: { ...valid.control, timeout: { request: '999ms', stream: '60s' } },
    }
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({ data: [invalid], page: { hasMore: false, pageSize: 100 } }),
            { status: 200, headers: { 'Content-Type': mediaType } },
          ),
      ),
    )

    await expect(routingManagementApi.listModels()).rejects.toThrow('Model control')
  })

  it('rejects response-only overload as retry evidence', async () => {
    const valid = model('model-a')
    const invalid = {
      ...valid,
      control: { ...valid.control, retry: { count: 1, on: ['overloaded'] } },
    }
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({ data: [invalid], page: { hasMore: false, pageSize: 100 } }),
            { status: 200, headers: { 'Content-Type': mediaType } },
          ),
      ),
    )

    await expect(routingManagementApi.listModels()).rejects.toThrow('RoutingModelPage')
  })

  it('supports bounded server-side search for routing selectors', async () => {
    const requests: Request[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const request = testRequest(input, init)
        requests.push(request)
        return new Response(
          JSON.stringify({
            data: [model('model-a')],
            page: { nextCursor: 'cursor-2', hasMore: true, pageSize: 20 },
          }),
          { status: 200, headers: { 'Content-Type': mediaType } },
        )
      }),
    )

    await expect(
      routingManagementApi.listModelsPage({
        search: 'qwen',
        cursor: 'cursor-1',
        pageSize: 20,
        status: 'active',
      }),
    ).resolves.toMatchObject({ data: [{ id: 'model-a' }], page: { hasMore: true } })

    expect(requests).toHaveLength(1)
    const query = new URL(requests[0].url).searchParams
    expect(query.get('search')).toBe('qwen')
    expect(query.get('cursor')).toBe('cursor-1')
    expect(query.get('pageSize')).toBe('20')
    expect(query.get('status')).toBe('active')
  })

  it('accepts the frozen assignment-set shape', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              data: {
                id: 'entrypoint-one',
                name: 'One',
                status: 'draft',
                revision: 1,
                entrypointRevision: 1,
                aliases: ['one'],
                recipeIds: ['recipe-one'],
                ruleCount: 1,
                assignedModelCount: 1,
                createdAt: '2026-08-23T00:00:00Z',
                updatedAt: '2026-08-23T00:00:00Z',
                rules: [
                  {
                    id: 'rule-one',
                    name: 'Default',
                    recipeId: 'recipe-one',
                    recipeRevision: 1,
                    assignments: {
                      decision_one: {
                        models: [
                          {
                            modelId: 'model-one',
                            modelRevision: 1,
                            priority: 0,
                            weight: '1',
                          },
                        ],
                      },
                    },
                  },
                ],
              },
            }),
            { status: 200, headers: { 'Content-Type': mediaType } },
          ),
      ),
    )

    await expect(
      routingManagementApi.getEntrypointTopology('entrypoint-one'),
    ).resolves.toMatchObject({
      rules: [{ assignments: { decision_one: { models: [{ modelId: 'model-one' }] } } }],
    })
  })

  it('rejects an Entrypoint summary that drops its Recipe references', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            data: [
              {
                id: 'entrypoint-one',
                name: 'One',
                status: 'active',
                revision: 1,
                entrypointRevision: 1,
                aliases: ['one'],
                recipeIds: [],
                ruleCount: 1,
                assignedModelCount: 1,
                createdAt: '2026-08-23T00:00:00Z',
                updatedAt: '2026-08-23T00:00:00Z',
              },
            ],
            page: { hasMore: false, pageSize: 100 },
          }),
          { status: 200, headers: { 'Content-Type': mediaType } },
        ),
      ),
    )

    await expect(routingManagementApi.listEntrypoints()).rejects.toThrow(
      'Entrypoint Recipe references',
    )
  })

  it('sends the frozen assignment set and validates its mutation receipt', async () => {
    const requests: Request[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        requests.push(testRequest(input, init))
        return new Response(
          JSON.stringify({
            resource: {
              kind: 'routing_entrypoint',
              id: 'entrypoint-one',
              revision: 1,
            },
            idempotency: { replayed: false },
          }),
          { status: 201, headers: { 'Content-Type': mediaType } },
        )
      }),
    )

    const receipt = await routingManagementApi.createEntrypoint({
      name: 'One',
      aliases: ['vllm-sr/one'],
      rules: [
        {
          name: 'Default',
          recipeId: 'recipe-one',
          assignments: {
            decision_one: {
              models: [
                { modelId: 'model-primary', priority: 0, weight: '1' },
                { modelId: 'model-backup', priority: 1, weight: '1' },
              ],
              fallback: { strategy: 'priority', on: ['timeout'] },
            },
          },
        },
      ],
    })

    expect('resource' in receipt ? receipt.resource.id : undefined).toBe('entrypoint-one')
    expect(requests[0].headers.get('Idempotency-Key')).toBeTruthy()
    expect(await requests[0].json()).toMatchObject({
      rules: [
        {
          assignments: {
            decision_one: {
              models: [{ priority: 0 }, { priority: 1 }],
              fallback: { strategy: 'priority', on: ['timeout'] },
            },
          },
        },
      ],
    })
  })

  it('patches Model policy without resubmitting hidden backend state', async () => {
    const requests: Request[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        requests.push(testRequest(input, init))
        return new Response(
          JSON.stringify({
            resource: { kind: 'routing_model', id: 'model-one', revision: 5 },
            idempotency: { replayed: false },
          }),
          { status: 200, headers: { 'Content-Type': mediaType } },
        )
      }),
    )

    await routingManagementApi.updateModel('model-one', 4, {
      control: {
        retry: { count: 3, on: ['unavailable'] },
        timeout: { request: '45s', stream: '5m' },
      },
    })

    expect(requests).toHaveLength(1)
    expect(requests[0].headers.get('If-Match')).toBe('"mdl:4"')
    expect(await requests[0].json()).toEqual({
      control: {
        retry: { count: 3, on: ['unavailable'] },
        timeout: { request: '45s', stream: '5m' },
      },
    })
  })

  it('validates typed probe and resolve responses', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const path = new URL(String(input), 'http://dashboard.test').pathname
        const payload = path.endsWith(':probe')
          ? {
              reachable: true,
              latencyMilliseconds: 12,
              checkedAt: '2026-08-23T00:00:00Z',
            }
          : { outcome: 'unclaimed' }
        return new Response(JSON.stringify(payload), {
          status: 200,
          headers: { 'Content-Type': mediaType },
        })
      }),
    )

    await expect(routingManagementApi.probeModel('model-one')).resolves.toMatchObject({
      reachable: true,
    })
    await expect(routingManagementApi.resolveEntrypoint('entrypoint-one', {})).resolves.toEqual({
      outcome: 'unclaimed',
    })
  })
})
