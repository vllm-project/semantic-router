import { afterEach, describe, expect, it, vi } from 'vitest'

import { renderCanonicalYaml } from './dslStoreYamlSupport'

describe('renderCanonicalYaml', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('uses the compiled YAML directly when there is no imported full config', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    await expect(renderCanonicalYaml('routing: {}', 'ROUTING {}', '')).resolves.toBe(
      'routing: {}',
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('renders a full config preview that preserves entrypoints and recipes', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        preview: [
          'version: v0.3',
          'entrypoints:',
          '  - model_names: [vllm-sr/balanced]',
          '    recipe: balanced',
          'recipes:',
          '  - name: balanced',
        ].join('\n'),
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await renderCanonicalYaml(
      'routing: {}',
      'ENTRYPOINT { model_names: ["vllm-sr/balanced"] recipe: "balanced" }',
      'version: v0.3',
    )

    expect(result).toContain('entrypoints:')
    expect(result).toContain('recipes:')
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/router/config/deploy/preview',
      expect.objectContaining({ method: 'POST' }),
    )
    const request = JSON.parse(fetchMock.mock.calls[0][1].body as string)
    expect(request).toMatchObject({
      yaml: 'routing: {}',
      baseYaml: 'version: v0.3',
      mode: 'replace',
    })
  })
})
