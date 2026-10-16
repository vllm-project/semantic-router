import { describe, expect, it } from 'vitest'

import { getRouterModelAnchor, type RouterModelInfo } from '../utils/routerRuntime'
import { clampInventoryPage, filterAndSortRouterModels } from './routerModelInventorySupport'

const models: RouterModelInfo[] = [
  {
    name: 'embedding',
    type: 'encoder',
    loaded: true,
    registry: { purpose: 'Embedding', tags: ['matryoshka'] },
  },
  {
    name: 'intent',
    type: 'classifier',
    loaded: false,
    state: 'downloading',
    registry: { purpose: 'Domain classification', repo_id: 'org/mmbert-intent' },
  },
  { name: 'pii', type: 'classifier', loaded: false },
]

describe('router model inventory support', () => {
  it('keeps matching model names distinct across searchable recipe scopes', () => {
    const scoped = ['default', 'a-b', 'a_b', '安全'].map((recipe) => ({
      name: 'pii_classifier',
      type: 'pii_detection',
      loaded: true,
      recipe,
    }))
    expect(new Set(scoped.map(getRouterModelAnchor)).size).toBe(scoped.length)
    expect(filterAndSortRouterModels(scoped, '安全', 'all', 'state')).toEqual([scoped[3]])
    expect(getRouterModelAnchor({ name: 'pii_classifier' })).toBe('model-pii-classifier')
    expect(getRouterModelAnchor(scoped[0])).toBe('model-pii-classifier')
  })

  it('searches registry metadata and groups transient loading states', () => {
    expect(
      filterAndSortRouterModels(models, 'mmbert', 'all', 'state').map((model) => model.name),
    ).toEqual(['intent'])
    expect(
      filterAndSortRouterModels(models, '', 'loading', 'state').map((model) => model.name),
    ).toEqual(['intent'])
  })

  it('sorts by purpose and clamps pages after filters change', () => {
    expect(filterAndSortRouterModels(models, '', 'all', 'type').map((model) => model.name)).toEqual(
      ['pii', 'intent', 'embedding'],
    )
    expect(clampInventoryPage(5, 9, 8)).toBe(2)
    expect(clampInventoryPage(0, 0, 8)).toBe(1)
  })
})
