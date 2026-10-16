import { afterEach, describe, expect, it, vi } from 'vitest'

import { runTestQueryPreview } from './useTestQuery'

afterEach(() => vi.unstubAllGlobals())

describe('Topology Preview failure state', () => {
  it.each([429, 503, 504])('keeps HTTP %s unavailable without simulated routing signals', async (status) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status })))

    const result = await runTestQueryPreview('long request', 'router/model')

    expect(result).toMatchObject({
      mode: 'dry-run',
      isAccurate: false,
      matchedSignals: [],
      matchedDecision: null,
      matchedModels: [],
      highlightedPath: ['client'],
    })
    expect(result.warning).toContain(String(status))
  })

  it('preserves an unavailable backend response instead of forcing accuracy', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({
      query: 'hello',
      mode: 'dry-run',
      isAccurate: false,
      matchedSignals: null,
      matchedDecision: null,
      matchedModels: null,
      highlightedPath: ['client'],
      warning: 'Router management credential is unavailable',
    })))

    const result = await runTestQueryPreview('hello')

    expect(result.isAccurate).toBe(false)
    expect(result.warning).toBe('Router management credential is unavailable')
    expect(result.matchedSignals).toEqual([])
    expect(result.matchedModels).toEqual([])
  })
})
