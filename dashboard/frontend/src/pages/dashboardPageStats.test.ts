import { describe, expect, it } from 'vitest'

import type { RoutingEntrypoint, RoutingRecipe } from '../utils/routingManagementApi'
import type { ManagedRoutingSummary } from '../utils/managedRoutingSnapshot'
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
})
