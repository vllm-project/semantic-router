import { describe, expect, it } from 'vitest'

import { assertKeyScopedRoutingCatalog, keyScopedCatalogSnapshot } from './keyScopedRoutingCatalog'

const digest = 'a'.repeat(64)

const catalog = {
  keyId: '10000000-0000-4000-8000-000000000001',
  policyRevision: 3,
  policyDigest: digest,
  routingRevision: 7,
  routingDigest: 'b'.repeat(64),
  models: [
    {
      id: 'model_fast',
      revision: 2,
      name: 'local/fast',
      aliases: [],
      capabilities: ['text'],
      loras: [],
      tags: ['fast'],
      pricing: {
        inputCostPerMillionTokens: '0.10',
        outputCostPerMillionTokens: '0.20',
        cacheReadCostPerMillionTokens: null,
        cacheWriteCostPerMillionTokens: null,
      },
    },
  ],
  recipes: [
    {
      id: 'recipe_blend',
      revision: 4,
      name: 'Blend',
      decisions: [{ id: 'decision_simple', name: 'Simple', dispatchCardinality: 'single' }],
    },
  ],
  entrypoints: [
    {
      id: 'entrypoint_blend',
      revision: 5,
      name: 'blend',
      aliases: ['vllm-sr/blend'],
      rules: [
        {
          id: 'rule_default',
          name: 'Default',
          recipeId: 'recipe_blend',
          recipeRevision: 4,
          assignments: {
            decision_simple: {
              models: [
                {
                  modelId: 'model_fast',
                  modelRevision: 2,
                  priority: 0,
                  weight: '1',
                },
              ],
            },
          },
        },
      ],
    },
  ],
}

describe('key-scoped routing catalog', () => {
  it('builds a hydrated read-only topology without provider or Recipe source data', () => {
    const parsed = assertKeyScopedRoutingCatalog(catalog)
    const snapshot = keyScopedCatalogSnapshot(parsed)

    expect(snapshot.models).toEqual([
      expect.objectContaining({ id: 'model_fast', name: 'local/fast' }),
    ])
    expect(snapshot.models[0]).not.toHaveProperty('backends')
    expect(snapshot.recipes[0].document).toEqual({
      decisions: [
        {
          id: 'decision_simple',
          name: 'Simple',
          dispatch_cardinality: 'single',
        },
      ],
    })
    expect(snapshot.entrypoints[0]).toMatchObject({
      assignedModelCount: 1,
      ruleCount: 1,
    })
    expect(snapshot.routingScopes[0]).toMatchObject({
      id: 'entrypoint:entrypoint_blend:rule_default',
      entrypointModelNames: ['vllm-sr/blend'],
      hydrated: true,
    })
  })

  it('rejects server fields outside the credential-free contract', () => {
    expect(() =>
      assertKeyScopedRoutingCatalog({
        ...catalog,
        models: [{ ...catalog.models[0], backends: [{ credentialId: 'secret' }] }],
      }),
    ).toThrow('RoutingCatalog')
    expect(() =>
      assertKeyScopedRoutingCatalog({
        ...catalog,
        recipes: [{ ...catalog.recipes[0], document: { signals: {} } }],
      }),
    ).toThrow('RoutingCatalog')
  })

  it('rejects assignments to a Model outside the visible projection', () => {
    const inconsistent = structuredClone(catalog)
    inconsistent.entrypoints[0].rules[0].assignments.decision_simple.models[0].modelId =
      'model_hidden'
    expect(() => assertKeyScopedRoutingCatalog(inconsistent)).toThrow(
      'inconsistent key-scoped routing catalog',
    )
  })
})
