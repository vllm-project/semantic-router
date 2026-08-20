import { describe, expect, it } from 'vitest'

import { RECIPE_SIGNAL_FAMILIES } from './recipeSignalCatalog'

describe('recipe signal composer', () => {
  it('exposes every canonical router signal type through one library', () => {
    expect(RECIPE_SIGNAL_FAMILIES).toHaveLength(20)
    expect(RECIPE_SIGNAL_FAMILIES.map((item) => item.key)).toEqual([
      'keywords',
      'embeddings',
      'domains',
      'fact_check',
      'user_feedbacks',
      'reasks',
      'preferences',
      'language',
      'context',
      'structure',
      'complexity',
      'modality',
      'role_bindings',
      'jailbreak',
      'pii',
      'kb',
      'conversation',
      'events',
      'metadata',
      'classifiers',
    ])
  })
})
