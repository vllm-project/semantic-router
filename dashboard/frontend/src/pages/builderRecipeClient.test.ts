import { beforeEach, describe, expect, it, vi } from 'vitest'

import { wasmBridge } from '@/lib/wasm'
import type { RoutingRecipe } from '@/utils/routingManagementApi'

import {
  compileBuilderRecipe,
  loadManagedRecipeSource,
  projectImportedRecipe,
} from './builderRecipeClient'

vi.mock('@/lib/wasm', () => ({
  wasmBridge: { compile: vi.fn(), decompile: vi.fn() },
}))

const recipe: RoutingRecipe = {
  id: 'rcp_one',
  name: 'One',
  description: 'First',
  status: 'draft',
  revision: 3,
  recipeRevision: 2,
  origin: 'custom',
  immutable: false,
  decisions: [],
  document: { signals: {}, decisions: [] },
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
}

beforeEach(() => vi.clearAllMocks())

describe('Builder Recipe projection', () => {
  it('loads one managed Recipe without provider, Model, or Entrypoint state', () => {
    vi.mocked(wasmBridge.decompile).mockReturnValue({ dsl: 'RECIPE One {}' })
    vi.mocked(wasmBridge.compile).mockReturnValue({
      yaml: '',
      diagnostics: [],
      recipeDocuments: [{ name: 'One', document: recipe.document }],
    })

    expect(loadManagedRecipeSource(recipe).document).toEqual(recipe.document)
    const bundle = JSON.parse(vi.mocked(wasmBridge.decompile).mock.calls[0][0])
    expect(bundle.recipes).toHaveLength(1)
    expect(bundle.recipes[0].routing).toEqual(recipe.document)
    expect(bundle.recipes[0]).not.toHaveProperty('document')
    expect(bundle.recipes[0]).not.toHaveProperty('id')
    expect(bundle.recipes[0]).not.toHaveProperty('revision')
    expect(bundle).not.toHaveProperty('version')
    expect(bundle).not.toHaveProperty('routing')
    expect(bundle).not.toHaveProperty('providers')
    expect(bundle).not.toHaveProperty('entrypoints')
  })

  it('uses recipeDocuments as the only save payload', () => {
    vi.mocked(wasmBridge.compile).mockReturnValue({
      yaml: 'providers: should-not-cross',
      diagnostics: [],
      recipeDocuments: [{ name: 'Generated', document: { strategy: 'priority' } }],
    })
    vi.mocked(wasmBridge.decompile).mockReturnValue({ dsl: 'RECIPE One {}' })

    expect(compileBuilderRecipe('RECIPE generated {}', recipe).document).toEqual({
      strategy: 'priority',
    })
  })

  it('imports an anonymous routing fragment into the selected Recipe', () => {
    vi.mocked(wasmBridge.decompile)
      .mockReturnValueOnce({ dsl: 'ROUTE direct {}' })
      .mockReturnValueOnce({ dsl: 'RECIPE One { ROUTE direct {} }' })
    vi.mocked(wasmBridge.compile).mockReturnValue({
      yaml: '',
      diagnostics: [],
      recipeDocuments: [{ document: { decisions: [{ name: 'direct' }] } }],
    })

    expect(projectImportedRecipe('routing: {}', recipe).document).toEqual({
      decisions: [{ name: 'direct' }],
    })
  })

  it('rejects imports that compile to more than one Recipe', () => {
    vi.mocked(wasmBridge.decompile).mockReturnValue({ dsl: 'RECIPE one {}\nRECIPE two {}' })
    vi.mocked(wasmBridge.compile).mockReturnValue({
      yaml: '',
      diagnostics: [],
      recipeDocuments: [
        { name: 'one', document: {} },
        { name: 'two', document: {} },
      ],
    })

    expect(() => projectImportedRecipe('recipes: []', recipe)).toThrow('exactly one Recipe')
  })
})
