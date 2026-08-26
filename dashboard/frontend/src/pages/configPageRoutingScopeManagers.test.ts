import { describe, expect, it } from 'vitest'

import type { RoutingRecipe } from '../utils/routingManagementApi'
import {
  managedRecipeConfig,
  managedRecipeDocument,
  withRecipeScope,
} from './configPageRoutingScopeSupport'

const recipe: RoutingRecipe = {
  id: '00000000-0000-4000-8000-000000000001',
  name: 'balanced',
  description: 'Balanced routing',
  status: 'active',
  revision: 7,
  recipeRevision: 7,
  origin: 'custom',
  immutable: false,
  decisions: [
    { id: '00000000-0000-4000-8000-000000000002', name: 'simple', dispatchCardinality: 'single' },
  ],
  document: {
    strategy: 'priority',
    signals: {
      keywords: [
        { name: 'simple-query', operator: 'OR', keywords: ['hello'], case_sensitive: false },
      ],
    },
    projections: { scores: [], mappings: [], partitions: [] },
    decisions: [
      {
        id: '00000000-0000-4000-8000-000000000002',
        name: 'simple',
        priority: 1,
        rules: { operator: 'AND', conditions: [{ type: 'keyword', name: 'simple-query' }] },
      },
    ],
  },
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
}

describe('managed Recipe editor boundary', () => {
  it('projects exactly one Router Recipe into the existing structured editors', () => {
    const config = managedRecipeConfig(recipe)

    expect(config.signals?.keywords?.[0]?.name).toBe('simple-query')
    expect(config.projections).toEqual(recipe.document.projections)
    expect(config.decisions?.[0]?.name).toBe('simple')
  })

  it('round-trips only the model-free Recipe document', () => {
    const document = managedRecipeDocument(managedRecipeConfig(recipe))

    expect(document).toEqual({
      strategy: 'priority',
      signals: recipe.document.signals,
      projections: recipe.document.projections,
      decisions: recipe.document.decisions,
    })
  })

  it('updates only the Recipe URL scope while preserving unrelated query state', () => {
    const selected = withRecipeScope(
      new URLSearchParams('view=compact&recipe=old-recipe'),
      'recipe/balanced',
    )

    expect(selected.toString()).toBe('view=compact&recipe=recipe%2Fbalanced')
    expect(withRecipeScope(selected, '').toString()).toBe('view=compact')
  })
})
