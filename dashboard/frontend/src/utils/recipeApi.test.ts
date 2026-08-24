import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  activateRecipePackage,
  buildRecipePackageQuery,
  buildRecipeProbeQuery,
  createRecipeProbeRunPlan,
  deactivateRecipePackage,
  getActiveRecipe,
  getRecipeProbe,
  listRecipePackages,
  listRecipeProbes,
  previewRecipePackageActivation,
  previewRecipePackageDeactivation,
  RecipeApiError,
  validateRecipeProbe,
} from './recipeApi'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('Recipe API client', () => {
  it('always requests a bounded server-side package page and trimmed search', () => {
    const query = new URLSearchParams(
      buildRecipePackageQuery({ page: 4, pageSize: 25, query: '  safety package  ' }),
    )

    expect(Object.fromEntries(query)).toEqual({
      page: '4',
      page_size: '25',
      q: 'safety package',
    })
  })

  it('forwards package list filters and abort signal', async () => {
    const controller = new AbortController()
    const fetchMock = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            items: [],
            page: 2,
            page_size: 25,
            total: 0,
            total_pages: 0,
            activation_state: 'none',
          }),
          { status: 200 },
        ),
    )
    vi.stubGlobal('fetch', fetchMock)

    await listRecipePackages({ page: 2, pageSize: 25, query: 'secure' }, controller.signal)

    expect(fetchMock).toHaveBeenCalledWith('/api/recipe/packages?page=2&page_size=25&q=secure', {
      signal: controller.signal,
    })
  })

  it('acknowledges package warnings only after the activation confirmation', async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ status: 'active', recipe_digest: 'sha256:recipe' }), {
          status: 200,
        }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await activateRecipePackage('sha256:recipe', true)

    expect(fetchMock).toHaveBeenCalledWith('/api/recipe/activate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recipe_digest: 'sha256:recipe',
        acknowledge_warnings: true,
      }),
      signal: undefined,
    })
  })

  it('previews topology and binds stack recreation to the exact plan digest', async () => {
    const plan = {
      recipe_digest: 'sha256:recipe',
      plan_digest: `sha256:${'f'.repeat(64)}`,
      mode: 'stack_recreation',
      requires_confirmation: true,
      listeners_before: [],
      listeners_after: [],
      storage: { before: [], after: ['redis'], add: ['redis'], remove: [], repair: [] },
      management_auth: { mode: 'disabled' },
      effects: ['recreate the managed stack'],
    }
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(plan), { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ status: 'active', recipe_digest: 'sha256:recipe' }), {
          status: 200,
        }),
      )
    vi.stubGlobal('fetch', fetchMock)

    const preview = await previewRecipePackageActivation('sha256:recipe', true)
    await activateRecipePackage('sha256:recipe', true, undefined, {
      expected_plan_digest: preview.plan_digest,
      confirm_stack_recreation: true,
    })

    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/recipe/activate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recipe_digest: 'sha256:recipe',
        acknowledge_warnings: true,
        expected_plan_digest: plan.plan_digest,
        confirm_stack_recreation: true,
      }),
      signal: undefined,
    })
  })

  it('restores the durable source runtime with a strict empty JSON request', async () => {
    const response = { status: 'inactive', previous_recipe_digest: 'sha256:recipe' }
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(response), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(deactivateRecipePackage()).resolves.toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith('/api/recipe/deactivate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
      signal: undefined,
    })
  })

  it('previews and confirms topology-changing source restoration', async () => {
    const plan = {
      recipe_digest: 'sha256:recipe',
      plan_digest: `sha256:${'e'.repeat(64)}`,
      mode: 'stack_recreation',
      requires_confirmation: true,
      listeners_before: [],
      listeners_after: [],
      storage: { before: ['redis'], after: [], add: [], remove: ['redis'], repair: [] },
      management_auth: { mode: 'disabled' },
      effects: ['restore source'],
    }
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(plan), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: 'inactive' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const preview = await previewRecipePackageDeactivation()
    await deactivateRecipePackage(undefined, {
      expected_plan_digest: preview.plan_digest,
      confirm_stack_recreation: true,
    })

    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/recipe/deactivate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        expected_plan_digest: plan.plan_digest,
        confirm_stack_recreation: true,
      }),
      signal: undefined,
    })
  })

  it('preserves machine-readable package errors for safe UI classification', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({ error: 'activation_rollback_failed', message: 'rollback failed' }),
            { status: 409 },
          ),
      ),
    )

    const request = activateRecipePackage('sha256:recipe', false)
    await expect(request).rejects.toBeInstanceOf(RecipeApiError)
    await expect(request).rejects.toMatchObject({
      code: 'activation_rollback_failed',
      status: 409,
    })
  })

  it('preserves invalid managed Recipe health returned with a 422', async () => {
    const descriptor = {
      managed: true,
      source_health: { status: 'invalid', files: {}, issues: ['probes.yaml is invalid'] },
      digests: {},
      counts: { unified_models: 0, recipes: 0, decisions: 0, probes: 0 },
    }
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(descriptor), { status: 422 })),
    )

    await expect(getActiveRecipe()).resolves.toEqual(descriptor)
  })

  it('builds bounded server-side probe pagination and filters', () => {
    const query = new URLSearchParams(
      buildRecipeProbeQuery({
        page: 3,
        pageSize: 25,
        query: '  safety review  ',
        decision: 'private/route',
        tag: 'language:zh',
        model: 'vllm-sr/mom private',
        requestShape: 'tools',
      }),
    )

    expect(Object.fromEntries(query)).toEqual({
      page: '3',
      page_size: '25',
      q: 'safety review',
      decision: 'private/route',
      tag: 'language:zh',
      model: 'vllm-sr/mom private',
      shape: 'tools',
    })
  })

  it('forwards list abort signals and never requests an unpaged collection', async () => {
    const controller = new AbortController()
    const fetchMock = vi.fn(
      async () => new Response(JSON.stringify({ items: [] }), { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await listRecipeProbes(
      { page: 2, pageSize: 50, recipe: 'privacy', decision: 'secure' },
      controller.signal,
    )

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/recipe/probes?page=2&page_size=50&recipe=privacy&decision=secure',
      { signal: controller.signal },
    )
  })

  it('encodes both stable probe path segments', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await getRecipeProbe('private/route', 'zh review')

    expect(fetchMock).toHaveBeenCalledWith('/api/recipe/probes/private%2Froute/zh%20review', {
      signal: undefined,
    })
  })

  it('preserves a strict failed validation returned with an upstream 502', async () => {
    const failedValidation = {
      probe_id: 'secure/failure',
      recipe_digest: 'sha256:recipe',
      passed: false,
      expected: { decision: 'secure' },
      actual: {
        decision: 'fallback',
        plugins: [],
        recommended_models: [],
        matched_signals: {},
        trace_decisions: [],
      },
      checks: {},
      failures: ['decision: expected secure, got fallback'],
      latency_ms: 11,
      error: 'router eval unavailable',
    }
    const fetchMock = vi.fn(
      async () => new Response(JSON.stringify(failedValidation), { status: 502 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(validateRecipeProbe('secure', 'failure', 'sha256:recipe')).resolves.toEqual(
      failedValidation,
    )
    expect(fetchMock).toHaveBeenCalledWith('/api/recipe/probes/secure/failure/validate', {
      method: 'POST',
      headers: { 'If-Match': '"sha256:recipe"' },
      signal: undefined,
    })
  })

  it('fetches a fresh run plan for each launch action', async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(
          JSON.stringify({ probe_id: 'secure/run', messages: [], request: {}, editable: true }),
          {
            status: 200,
          },
        ),
    )
    vi.stubGlobal('fetch', fetchMock)

    await createRecipeProbeRunPlan('secure', 'run', 'sha256:recipe')
    await createRecipeProbeRunPlan('secure', 'run', 'sha256:recipe')

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock).toHaveBeenLastCalledWith('/api/recipe/probes/secure/run/run-plan', {
      method: 'POST',
      headers: { 'If-Match': '"sha256:recipe"' },
      signal: undefined,
    })
  })
})
