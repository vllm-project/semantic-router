import { afterEach, describe, expect, it, vi } from 'vitest'

import { discoverProviderModels } from './modelDiscoveryApi'

describe('model discovery API', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('sends connection details and returns a bounded model contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ models: [{ id: 'model-a', ownedBy: 'team-a' }] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      discoverProviderModels({
        baseUrl: 'http://provider.test/v1',
        modelsPath: '/models',
        apiKey: 'secret',
        extraHeaders: { 'anthropic-version': '2023-06-01' },
      }),
    ).resolves.toEqual([{ id: 'model-a', ownedBy: 'team-a' }])
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/models/discover',
      expect.objectContaining({ method: 'POST', body: expect.stringContaining('secret') }),
    )
    expect(fetchMock.mock.calls[0][1].body).toContain('anthropic-version')
    expect(fetchMock.mock.calls[0][1].body).toContain('modelsPath')
  })

  it('surfaces the product-safe server message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ message: 'This connection could not be reached.' }), {
          status: 502,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )
    await expect(discoverProviderModels({ baseUrl: 'http://provider.test' })).rejects.toEqual(
      expect.objectContaining({
        message: 'This connection could not be reached.',
        status: 502,
      }),
    )
  })
})
