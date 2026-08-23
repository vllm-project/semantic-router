import { describe, expect, it } from 'vitest'

import type { RoutingRecipe } from '../utils/routingManagementApi'
import {
  recipeDocumentSummary,
  recipeWrite,
  EMPTY_RECIPE_DOCUMENT,
  suggestedRecipeCopyName,
} from './configPageRecipeDialogSupport'

const builtInRecipe: RoutingRecipe = {
  id: 'recipe-built-in',
  name: 'Balanced',
  description: 'Balanced routing',
  status: 'active',
  revision: 3,
  recipeRevision: 3,
  origin: 'distribution',
  immutable: true,
  provenance: {
    distributionId: 'core',
    distributionVersion: '1.0.0',
    assetDigest: `sha256:${'a'.repeat(64)}`,
    sourceRecipeId: 'balanced',
    sourceRevision: 3,
    recipeDigest: `sha256:${'b'.repeat(64)}`,
    installedAt: '2026-08-23T00:00:00Z',
  },
  decisions: [
    { id: 'simple', name: 'Simple', dispatchCardinality: 'single' },
    { id: 'complex', name: 'Complex', dispatchCardinality: 'single' },
  ],
  document: {
    signals: { keywords: [{ name: 'brief' }], embeddings: [{ name: 'complexity' }] },
    projections: { scores: [{ name: 'difficulty' }], mappings: [{ name: 'band' }] },
    decisions: [
      { id: 'simple', name: 'Simple', rules: {} },
      { id: 'complex', name: 'Complex', rules: {} },
    ],
  },
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
}

describe('Recipe dialog support', () => {
  it('creates a valid starter without requiring a pre-existing Signal', () => {
    const input = recipeWrite('My Recipe', '')

    expect(input).toEqual({ name: 'My Recipe', document: EMPTY_RECIPE_DOCUMENT })
    expect(input.document.signals).toEqual({})
    expect(input.document.decisions).toEqual([])
  })

  it('duplicates only writable Recipe fields into a new resource', () => {
    const input = recipeWrite(
      suggestedRecipeCopyName(builtInRecipe),
      builtInRecipe.description ?? '',
      builtInRecipe.document,
    )

    expect(input).toMatchObject({ name: 'Balanced copy', description: 'Balanced routing' })
    expect(input).not.toHaveProperty('id')
    expect(input).not.toHaveProperty('immutable')
    expect(input).not.toHaveProperty('provenance')
    expect(input.document).toEqual(builtInRecipe.document)
    expect(input.document).not.toBe(builtInRecipe.document)
  })

  it('counts resources, not Signal or Projection family names', () => {
    expect(recipeDocumentSummary(builtInRecipe)).toEqual({
      signals: 2,
      projections: 2,
      decisions: 2,
    })
  })
})
