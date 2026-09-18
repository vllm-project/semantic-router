import { afterEach, describe, expect, it, vi } from 'vitest'
import { benchApi } from './api'
import { DEFAULT_LIMITS, makeManifest } from './model'

afterEach(() => vi.unstubAllGlobals())

describe('shared sr-bench API', () => {
  it('submits the frozen manifest to the same service as CLI runs', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ id: 'run-1' })))
    vi.stubGlobal('fetch', fetcher)
    const manifest = makeManifest('Trial', 'live', 'smoke', undefined, [], DEFAULT_LIMITS)
    await benchApi.start(manifest, 'submit-1')
    expect(fetcher).toHaveBeenCalledWith(
      '/api/sr-bench/v1/runs',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ manifest, idempotency_key: 'submit-1' }),
      }),
    )
  })
  it('reports server failure without retrying a potentially accepted generation', async () => {
    const fetcher = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { message: 'Dispatch requires reconciliation' } }), {
        status: 409,
      }),
    )
    vi.stubGlobal('fetch', fetcher)
    await expect(benchApi.cancel('run-1')).rejects.toThrow('Dispatch requires reconciliation')
    expect(fetcher).toHaveBeenCalledTimes(1)
  })
})
