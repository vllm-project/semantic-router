import { afterEach, describe, expect, it, vi } from 'vitest'

import { countDecisions, countModels } from '../pages/dashboardPageStats'
import type { ManagedTopologyConfig } from '../pages/topology/types'
import { parseConfigToTopology } from '../pages/topology/utils/topologyParser'
import type { RoutingEntrypoint, RoutingModelCardView, RoutingRecipe } from './routingManagementApi'
import { routingManagementApi } from './routingManagementApi'
import {
  buildManagedRoutingSnapshot,
  fetchManagedRoutingSnapshot,
  fetchManagedRoutingSummary,
  listManagedRecipeScopes,
} from './managedRoutingSnapshot'

const model = (id: string, name: string): RoutingModelCardView => ({
  id,
  name,
  card: { aliases: [], capabilities: [], loras: [], tags: [] },
})

const recipe: RoutingRecipe = {
  id: 'recipe-balanced',
  name: 'balanced',
  description: 'Balanced work',
  status: 'active',
  revision: 1,
  recipeRevision: 1,
  origin: 'custom',
  immutable: false,
  decisions: [{ id: 'decision-complex', name: 'complex', dispatchCardinality: 'single' }],
  document: {
    strategy: 'priority',
    signals: { complexity: [{ name: 'hard', threshold: 0.7 }] },
    decisions: [
      {
        id: 'decision-complex',
        name: 'complex',
        priority: 100,
        rules: { operator: 'AND', conditions: [{ type: 'complexity', name: 'hard' }] },
      },
    ],
  },
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
}

const entrypoint = (
  id: string,
  alias: string,
  ruleID: string,
  modelID: string,
): RoutingEntrypoint => ({
  id,
  name: alias,
  status: 'active',
  revision: 1,
  entrypointRevision: 1,
  aliases: [alias],
  ruleCount: 1,
  assignedModelCount: 1,
  rules: [
    {
      id: ruleID,
      name: 'Default',
      recipeId: recipe.id,
      recipeRevision: recipe.revision,
      assignments: {
        'decision-complex': {
          models: [{ modelId: modelID, modelRevision: 1, priority: 0, weight: '1' }],
        },
      },
    },
  ],
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
})

describe('managed routing snapshot', () => {
  afterEach(() => vi.restoreAllMocks())

  it('counts managed resources once on the Dashboard', () => {
    const snapshot = buildManagedRoutingSnapshot(
      [model('model-fast', 'fast'), model('model-deep', 'deep')],
      [recipe],
      [
        entrypoint('entrypoint-public', 'vllm-sr/public', 'rule-public', 'model-fast'),
        entrypoint('entrypoint-premium', 'vllm-sr/premium', 'rule-premium', 'model-deep'),
      ],
    )

    expect(countModels(snapshot)).toBe(2)
    expect(countDecisions(snapshot)).toBe(1)
    expect(listManagedRecipeScopes(snapshot)[0].entrypointModelNames).toEqual([
      'vllm-sr/public',
      'vllm-sr/premium',
    ])
    expect(snapshot).not.toHaveProperty('dashboardConfig')
    expect(snapshot).not.toHaveProperty('topologyConfig')
    expect(snapshot).not.toHaveProperty('providers')
    expect(snapshot.recipes[0]).toHaveProperty('document')
    expect(snapshot.recipes[0]).not.toHaveProperty('routing')
  })

  it('builds one exact topology scope for each Entrypoint rule assignment', () => {
    const snapshot = buildManagedRoutingSnapshot(
      [model('model-fast', 'fast'), model('model-deep', 'deep')],
      [recipe],
      [
        entrypoint('entrypoint-public', 'vllm-sr/public', 'rule-public', 'model-fast'),
        entrypoint('entrypoint-premium', 'vllm-sr/premium', 'rule-premium', 'model-deep'),
      ],
    )
    const scopes = snapshot.routingScopes

    expect(scopes.map((scope) => scope.label)).toEqual(['vllm-sr/public', 'vllm-sr/premium'])
    expect(scopes.map((scope) => scope.entrypointModelNames)).toEqual([
      ['vllm-sr/public'],
      ['vllm-sr/premium'],
    ])

    const publicTopology = parseConfigToTopology({
      models: snapshot.models,
      document: scopes[0].document as ManagedTopologyConfig['document'],
    })
    const premiumTopology = parseConfigToTopology({
      models: snapshot.models,
      document: scopes[1].document as ManagedTopologyConfig['document'],
    })
    expect(publicTopology.decisions[0].modelRefs.map((reference) => reference.model)).toEqual([
      'fast',
    ])
    expect(premiumTopology.decisions[0].modelRefs.map((reference) => reference.model)).toEqual([
      'deep',
    ])
  })

  it('keeps Dashboard summaries free of per-Entrypoint topology requests', async () => {
    const summary: RoutingEntrypoint = {
      ...entrypoint('entrypoint-public', 'vllm-sr/public', 'rule-public', 'model-fast'),
      rules: undefined,
    }
    vi.spyOn(routingManagementApi, 'listModelCards').mockResolvedValue([
      model('model-fast', 'fast'),
    ])
    vi.spyOn(routingManagementApi, 'listRecipes').mockResolvedValue([recipe])
    vi.spyOn(routingManagementApi, 'listEntrypoints').mockResolvedValue([summary])
    const topology = vi.spyOn(routingManagementApi, 'getEntrypointTopology')

    const result = await fetchManagedRoutingSummary()

    expect(result.entrypoints).toEqual([summary])
    expect(topology).not.toHaveBeenCalled()
  })

  it('hydrates only the Entrypoint named by a topology deep link', async () => {
    const publicEntrypoint = entrypoint(
      'entrypoint-public',
      'vllm-sr/public',
      'rule-public',
      'model-fast',
    )
    const premiumEntrypoint = entrypoint(
      'entrypoint-premium',
      'vllm-sr/premium',
      'rule-premium',
      'model-deep',
    )
    const summaries = [publicEntrypoint, premiumEntrypoint].map((item) => ({
      ...item,
      rules: undefined,
    }))
    vi.spyOn(routingManagementApi, 'listModelCards').mockResolvedValue([
      model('model-fast', 'fast'),
      model('model-deep', 'deep'),
    ])
    vi.spyOn(routingManagementApi, 'listRecipes').mockResolvedValue([recipe])
    vi.spyOn(routingManagementApi, 'listEntrypoints').mockResolvedValue(summaries)
    const topology = vi
      .spyOn(routingManagementApi, 'getEntrypointTopology')
      .mockImplementation(async (id) =>
        id === premiumEntrypoint.id ? premiumEntrypoint : publicEntrypoint,
      )

    const result = await fetchManagedRoutingSnapshot('entrypoint:entrypoint-premium:rule-premium')

    expect(topology).toHaveBeenCalledTimes(1)
    expect(topology).toHaveBeenCalledWith('entrypoint-premium')
    expect(result.entrypoints).toEqual(summaries)
    const scopeIDs = result.routingScopes.map((scope) => scope.id)
    expect(scopeIDs).toContain('entrypoint:entrypoint-premium:rule-premium')
    expect(scopeIDs).toContain('entrypoint:entrypoint-public')
  })

  it('opens a Recipe topology without hydrating any Entrypoint', async () => {
    vi.spyOn(routingManagementApi, 'listModelCards').mockResolvedValue([
      model('model-fast', 'fast'),
    ])
    vi.spyOn(routingManagementApi, 'listRecipes').mockResolvedValue([recipe])
    vi.spyOn(routingManagementApi, 'listEntrypoints').mockResolvedValue([])
    const topology = vi.spyOn(routingManagementApi, 'getEntrypointTopology')

    const result = await fetchManagedRoutingSnapshot(`recipe:${recipe.id}`)

    expect(topology).not.toHaveBeenCalled()
    expect(result.routingScopes.map((scope) => scope.id)).toContain(`recipe:${recipe.id}`)
  })
})
