import { afterEach, describe, expect, it, vi } from 'vitest'

import { composeBuiltInRecipe, type BuiltInRecipeComposeRequest } from './configPageBuiltInRecipes'

const request: BuiltInRecipeComposeRequest = {
  bundle: 'mom-v1',
  recipe: 'vault',
  bindings: { private: [{ model: 'model-a', use_reasoning: false }] },
  model_names: ['my/vault', 'vault'],
}

afterEach(() => vi.unstubAllGlobals())

describe('built-in recipe composition', () => {
  it('posts the bindings and returns the composed config without saving it', async () => {
    const composed = {
      version: 'v0.3',
      entrypoints: [{ model_names: ['my/vault'], recipe: 'vault' }],
    }
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(Response.json(composed))
    vi.stubGlobal('fetch', fetcher)

    await expect(composeBuiltInRecipe(request)).resolves.toEqual(composed)
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(fetcher).toHaveBeenCalledWith('/api/recipes/builtin/compose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
  })

  it('reports the reason the packaged CLI rejected the request', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(
          Response.json(
            { error: 'recipe_rejected', message: "Config already contains recipe 'vault'" },
            { status: 400 },
          ),
        ),
    )

    await expect(composeBuiltInRecipe(request)).rejects.toThrow(
      "Config already contains recipe 'vault'",
    )
  })
})
