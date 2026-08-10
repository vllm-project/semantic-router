import { describe, expect, it } from 'vitest'

import {
  applyRoutingScopeProjection,
  collectScopedDecisions,
  countSignalsAcrossScopes,
  listRoutingScopes,
  projectConfigForRoutingScope,
  type RoutingScopedConfigLike,
} from './routingScopes'

const recipeConfig: RoutingScopedConfigLike = {
  routing: {
    strategy: 'priority',
    decisions: [],
  },
  entrypoints: [
    { model_names: ['vllm-sr/balanced'], recipe: 'balanced' },
    { model_names: ['vllm-sr/private'], recipe: 'privacy' },
  ],
  recipes: [
    {
      name: 'balanced',
      routing: {
        signals: {
          keywords: [{ name: 'balanced-keyword' }],
          context: [{ name: 'balanced-context' }],
        },
        projections: {
          scores: [{ name: 'balanced-score' }],
          mappings: [{ name: 'balanced-map' }],
        },
        decisions: [{ name: 'balanced-route', plugins: [{ type: 'semantic-cache' }] }],
      },
    },
    {
      name: 'privacy',
      routing: {
        signals: {
          pii: [{ name: 'private-pii' }],
        },
        decisions: [{ name: 'private-route' }],
      },
    },
  ],
}

describe('routingScopes', () => {
  it('lists entrypoint-owned recipes without inventing an empty default scope', () => {
    const scopes = listRoutingScopes(recipeConfig)

    expect(scopes.map((scope) => scope.id)).toEqual(['balanced', 'privacy'])
    expect(scopes[0].entrypointModelNames).toEqual(['vllm-sr/balanced'])
  })

  it('projects one recipe into the existing manager and topology view shape', () => {
    const projected = projectConfigForRoutingScope(recipeConfig, 'privacy')

    expect(projected.signals).toEqual({ pii: [{ name: 'private-pii' }] })
    expect(projected.decisions).toEqual([{ name: 'private-route' }])
    expect(projected.routing?.decisions).toEqual([{ name: 'private-route' }])
  })

  it('applies manager edits back to only the selected recipe', () => {
    const projected = projectConfigForRoutingScope(recipeConfig, 'privacy')
    projected.signals = {
      pii: [{ name: 'updated-private-pii' }],
    }
    projected.decisions = [{ name: 'updated-private-route' }]

    const updated = applyRoutingScopeProjection(recipeConfig, projected, 'privacy')

    expect(updated.recipes?.[0]).toEqual(recipeConfig.recipes?.[0])
    expect(updated.recipes?.[1].routing.signals).toEqual({
      pii: [{ name: 'updated-private-pii' }],
    })
    expect(updated.recipes?.[1].routing.decisions).toEqual([
      { name: 'updated-private-route' },
    ])
    expect(updated.routing?.decisions).toEqual([])
  })

  it('aggregates signals and decisions across isolated recipes', () => {
    expect(countSignalsAcrossScopes(recipeConfig)).toEqual({
      total: 3,
      byType: { keywords: 1, context: 1, pii: 1 },
    })
    expect(
      collectScopedDecisions<{ name: string }>(recipeConfig).map(
        ({ scope, value }) => `${scope.id}:${value.name}`,
      ),
    ).toEqual(['balanced:balanced-route', 'privacy:private-route'])
  })

  it('keeps a single-profile config available as the default scope', () => {
    const config = {
      routing: {
        signals: { keywords: [{ name: 'hello' }] },
        decisions: [{ name: 'hello-route' }],
      },
    }

    const scopes = listRoutingScopes(config)

    expect(scopes).toHaveLength(1)
    expect(scopes[0]).toMatchObject({ id: 'default', isDefault: true })
  })

  it('updates an explicit default recipe without creating top-level routing state', () => {
    const config: RoutingScopedConfigLike = {
      routing: { decisions: [] },
      recipes: [
        {
          name: 'default',
          routing: { decisions: [{ name: 'explicit-default' }] },
        },
      ],
    }
    const projected = projectConfigForRoutingScope(config, 'default')
    projected.decisions = [{ name: 'updated-default' }]

    const updated = applyRoutingScopeProjection(config, projected, 'default')

    expect(updated.recipes?.[0].routing.decisions).toEqual([{ name: 'updated-default' }])
    expect(updated.routing?.decisions).toEqual([])
  })
})
