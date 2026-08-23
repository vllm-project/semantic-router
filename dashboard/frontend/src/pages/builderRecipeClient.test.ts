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
      recipeDocuments: [{ id: 'rcp_one', name: 'One', document: recipe.document }],
    })

    expect(loadManagedRecipeSource(recipe).document).toEqual(recipe.document)
    const envelope = JSON.parse(vi.mocked(wasmBridge.decompile).mock.calls[0][0])
    expect(envelope.recipes).toHaveLength(1)
    expect(envelope.recipes[0].document).toEqual(recipe.document)
    expect(envelope.recipes[0]).not.toHaveProperty('routing')
    expect(envelope).not.toHaveProperty('routing')
    expect(envelope).not.toHaveProperty('providers')
    expect(envelope).not.toHaveProperty('entrypoints')
  })

  it('uses recipeDocuments as the only save payload', () => {
    vi.mocked(wasmBridge.compile).mockReturnValue({
      yaml: 'providers: should-not-cross',
      diagnostics: [],
      recipeDocuments: [{ id: 'generated', name: 'Generated', document: { strategy: 'priority' } }],
    })
    vi.mocked(wasmBridge.decompile).mockReturnValue({ dsl: 'RECIPE One {}' })

    expect(compileBuilderRecipe('RECIPE generated {}', recipe).document).toEqual({
      strategy: 'priority',
    })
  })

  it('rejects imports that compile to more than one Recipe', () => {
    vi.mocked(wasmBridge.decompile).mockReturnValue({ dsl: 'RECIPE one {}\nRECIPE two {}' })
    vi.mocked(wasmBridge.compile).mockReturnValue({
      yaml: '',
      diagnostics: [],
      recipeDocuments: [
        { id: 'one', name: 'one', document: {} },
        { id: 'two', name: 'two', document: {} },
      ],
    })

    expect(() => projectImportedRecipe('recipes: []', recipe)).toThrow('exactly one Recipe')
  })
})
