import { describe, expect, it } from 'vitest'

import type { RoutingEntrypoint, RoutingRecipe } from '../utils/routingManagementApi'
import type { ManagedRoutingSummary } from '../utils/managedRoutingSnapshot'
import { buildManagedRoutingSnapshot } from '../utils/managedRoutingSnapshot'
import {
  categorizeDecisions,
  countDecisions,
  countPlugins,
  countSignals,
  getAllDecisions,
} from './dashboardPageStats'

const recipe = (id: string, name: string, document: Record<string, unknown>): RoutingRecipe => ({
  id,
  name,
  status: 'active',
  revision: 1,
  recipeRevision: 1,
  origin: 'custom',
  immutable: false,
  decisions: [],
  document,
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
})

const entrypoint = (id: string, alias: string, recipeId: string): RoutingEntrypoint => ({
  id,
  name: alias,
  status: 'active',
  revision: 1,
  entrypointRevision: 1,
  aliases: [alias],
  recipeIds: [recipeId],
  ruleCount: 1,
  assignedModelCount: 0,
  rules: [
    {
      id: `rule-${id}`,
      name: 'Default',
      recipeId,
      recipeRevision: 1,
      assignments: {},
    },
  ],
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
})

const config: ManagedRoutingSummary = {
  models: [],
  recipes: [
    recipe('recipe-balanced', 'balanced', {
      signals: {
        keywords: [{ name: 'balanced-keyword' }],
        context: [{ name: 'balanced-context' }],
      },
      decisions: [
        {
          name: 'balanced-route',
          priority: 100,
          plugins: [{ type: 'response_cache' }],
        },
      ],
    }),
    recipe('recipe-privacy', 'privacy', {
      signals: { pii: [{ name: 'private-pii' }] },
      decisions: [{ name: 'private-route', priority: 1000 }],
    }),
  ],
  entrypoints: [
    entrypoint('entrypoint-balanced', 'vllm-sr/balanced', 'recipe-balanced'),
    entrypoint('entrypoint-private', 'vllm-sr/private', 'recipe-privacy'),
  ],
}

describe('recipe-aware dashboard stats', () => {
  it('counts signals, decisions, and plugins across native Recipe documents', () => {
    expect(countSignals(config)).toEqual({
      total: 3,
      byType: { keywords: 1, context: 1, pii: 1 },
    })
    expect(countDecisions(config)).toBe(2)
    expect(countPlugins(config)).toBe(1)
  })

  it('keeps Recipe and Entrypoint metadata on overview decisions', () => {
    expect(getAllDecisions(config)).toEqual([
      expect.objectContaining({
        name: 'balanced-route',
        routingScope: 'recipe:recipe-balanced',
        routingEntrypoints: ['vllm-sr/balanced'],
      }),
      expect.objectContaining({
        name: 'private-route',
        routingScope: 'recipe:recipe-privacy',
        routingEntrypoints: ['vllm-sr/private'],
      }),
    ])
    expect(categorizeDecisions(config).guardrails).toHaveLength(1)
  })

  it('uses hydrated Entrypoint assignments for the decision model overview', () => {
    const routedRecipe = recipe('recipe-routed', 'routed', {
      decisions: [{ id: 'decision-routed', name: 'routed', priority: 100 }],
    })
    routedRecipe.decisions = [
      { id: 'decision-routed', name: 'routed', dispatchCardinality: 'single' },
    ]
    const routedEntrypoint = entrypoint('entrypoint-routed', 'vllm-sr/routed', 'recipe-routed')
    routedEntrypoint.rules![0].assignments = {
      'decision-routed': {
        models: [{ modelId: 'model-fast', modelRevision: 1, priority: 0, weight: '1' }],
      },
    }
    const snapshot = buildManagedRoutingSnapshot(
      [
        {
          id: 'model-fast',
          name: 'local/fast',
          card: { aliases: [], capabilities: [], loras: [], tags: [] },
        },
      ],
      [routedRecipe],
      [routedEntrypoint],
    )

    expect(getAllDecisions(snapshot)).toEqual([
      expect.objectContaining({
        name: 'routed',
        routingScope: 'entrypoint:entrypoint-routed:rule-entrypoint-routed',
        routingEntrypoints: ['vllm-sr/routed'],
        modelRefs: [expect.objectContaining({ model: 'local/fast' })],
      }),
    ])
  })

  it('does not let an undeployed Recipe displace assigned decision models', () => {
    const draftRecipe = recipe('recipe-draft', 'draft', {
      decisions: [{ id: 'decision-draft', name: 'draft', priority: 100 }],
    })
    const deployedRecipe = recipe('recipe-deployed', 'deployed', {
      decisions: [{ id: 'decision-deployed', name: 'deployed', priority: 100 }],
    })
    deployedRecipe.decisions = [
      { id: 'decision-deployed', name: 'deployed', dispatchCardinality: 'single' },
    ]
    const deployedEntrypoint = entrypoint(
      'entrypoint-deployed',
      'vllm-sr/deployed',
      deployedRecipe.id,
    )
    deployedEntrypoint.rules![0].assignments = {
      'decision-deployed': {
        models: [{ modelId: 'model-fast', modelRevision: 1, priority: 0, weight: '1' }],
      },
    }
    const snapshot = buildManagedRoutingSnapshot(
      [
        {
          id: 'model-fast',
          name: 'local/fast',
          card: { aliases: [], capabilities: [], loras: [], tags: [] },
        },
      ],
      [draftRecipe, deployedRecipe],
      [deployedEntrypoint],
    )

    expect(getAllDecisions(snapshot)).toEqual([
      expect.objectContaining({
        name: 'deployed',
        modelRefs: [expect.objectContaining({ model: 'local/fast' })],
      }),
    ])
  })
})
