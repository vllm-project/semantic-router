import { describe, expect, it } from 'vitest'
import { DECISION_RUNTIME_CATALOG, DECISION_RUNTIME_CAPABILITIES } from './decisionRuntimeCatalog'
import {
  compatibleRuntimeConsumers,
  configuredDecisionRuntimes,
  decisionRuntimeConsumers,
  withDecisionRuntimeDeployment,
} from './decisionRuntimeDeployment'
import type { RouterConfig } from './dashboardPageTypes'
import type { ModelRuntimeInventory } from './decisionModelManagement'

const entry = DECISION_RUNTIME_CATALOG[0]
function snapshot(): RouterConfig {
  return {
    version: 'v0.3',
    listeners: [{ name: 'public', port: 8000 }],
    providers: {
      models: [{ name: 'answer', api_key: 'test-provider-secret', endpoints: [{ name: 'gpu' }] }],
    },
    global: {
      model_catalog: {
        system: { decision_model: 'Vela-2.0-4B', pii: 'private-specialist' },
        deployments: {
          other: { provider: 'model_runtime', artifact: 'team/other', profile: 'batching' },
        },
        admission: { custom: { max_inflight: 7 } },
      },
      services: { secret: 'test-opaque-value' },
    },
    routing: {
      signals: {
        decision: [
          { name: 'task', question: { type: 'noul', instructions: 'Is this coding?' } },
          { name: 'entities', deployment: 'other', question: { type: 'span' } },
        ],
      },
      decisions: [],
    },
    recipes: [
      {
        name: 'research',
        routing: {
          signals: {
            decision: [
              {
                name: 'task',
                deployment: 'other',
                question: { type: 'score', instructions: 'Difficulty?', levels: ['easy', 'hard'] },
                predicate: { gte: 1 },
              },
            ],
          },
          decisions: [
            {
              name: 'select',
              algorithm: {
                type: 'decision',
                decision: {
                  deployment: 'other',
                  instructions: 'Choose',
                  candidates: { answer: 'General' },
                },
              },
              modelRefs: [{ model: 'answer' }],
            },
          ],
          replay: { custom_future_field: 'preserve' },
        },
      },
    ],
    entrypoints: [{ model_names: ['research'], recipe: 'research' }],
  } as unknown as RouterConfig
}

describe('Decision runtime catalog projection', () => {
  it('contains the projected release families and their supported question types', () => {
    expect(DECISION_RUNTIME_CATALOG.filter((value) => value.family === 'decision2')).toHaveLength(6)
    expect(DECISION_RUNTIME_CATALOG.filter((value) => value.family === 'decision1')).toHaveLength(7)
    expect(DECISION_RUNTIME_CAPABILITIES).toEqual(['choice', 'noul', 'score'])
  })

  it('shows decision models and bindings without misclassifying other runtime consumers', () => {
    const current = snapshot()
    current.global = {
      model_catalog: {
        deployments: {
          embedding: { provider: 'model_runtime', artifact: 'team/embedding' },
          reranker: { provider: 'model_runtime', artifact: 'team/reranker' },
          unboundDecision: { provider: 'model_runtime', artifact: entry.id },
          other: { provider: 'model_runtime', artifact: 'team/custom-question-model' },
          observedQuestion: { provider: 'model_runtime', artifact: 'team/observed-question' },
          external: { provider: 'external', artifact: entry.id },
        },
      },
    }
    const inventory = {
      deployments: [
        { name: 'embedding', ready: true, surfaces: ['embedding'] },
        { name: 'reranker', ready: true, surfaces: ['rerank'] },
        { name: 'observedQuestion', ready: true, surfaces: ['question'] },
      ],
    } as ModelRuntimeInventory
    expect(configuredDecisionRuntimes(current, inventory).map(([name]) => name)).toEqual([
      'unboundDecision',
      'other',
      'observedQuestion',
    ])
    expect(configuredDecisionRuntimes(current, null).map(([name]) => name)).toEqual([
      'unboundDecision',
      'other',
    ])
  })
})

describe('Decision runtime canonical mutations', () => {
  it('binds only the selected recipe question and preserves credentials, defaults, policies and unknown leaves', () => {
    const current = snapshot()
    const before = structuredClone(current)
    const consumer = compatibleRuntimeConsumers(current).find(
      (value) => value.recipe === 'research' && value.kind === 'question',
    )!
    const next = withDecisionRuntimeDeployment(current, {
      entry,
      name: 'research-decider',
      device: 'auto',
      consumerId: consumer.id,
    })
    expect(current).toEqual(before)
    const expected = structuredClone(before)
    const global = expected.global as { model_catalog: { deployments: Record<string, unknown> } }
    global.model_catalog.deployments['research-decider'] = {
      provider: 'model_runtime',
      artifact: entry.id,
      revision: entry.revision,
      device: 'auto',
    }
    expected.recipes![0].routing.signals!.decision[0].deployment = 'research-decider'
    expect(next).toEqual(expected)
  })

  it('keeps top-level and recipe consumers with duplicate names distinct and excludes span/set', () => {
    const current = snapshot()
    const consumers = compatibleRuntimeConsumers(current)
    expect(consumers.map((value) => [value.recipe, value.name, value.kind])).toEqual([
      [null, 'task', 'question'],
      ['research', 'task', 'question'],
      ['research', 'select', 'selector'],
    ])
    const next = withDecisionRuntimeDeployment(current, {
      entry,
      name: 'decider',
      device: 'cpu',
      consumerId: consumers[0].id,
    })
    expect(next.routing?.signals?.decision[0].deployment).toBe('decider')
    expect(next.recipes?.[0].routing.signals?.decision[0].deployment).toBe('other')
  })

  it('writes a decision selector without changing its candidate set or routing rules', () => {
    const current = snapshot()
    const consumer = compatibleRuntimeConsumers(current).find((value) => value.kind === 'selector')!
    const next = withDecisionRuntimeDeployment(current, {
      entry,
      name: 'selector-runtime',
      device: 'rocm:0',
      consumerId: consumer.id,
    })
    expect(next.recipes?.[0].routing.decisions?.[0]).toEqual({
      ...current.recipes![0].routing.decisions![0],
      algorithm: {
        type: 'decision',
        decision: {
          deployment: 'selector-runtime',
          instructions: 'Choose',
          candidates: { answer: 'General' },
        },
      },
    })
  })

  it('saves an unbound declaration without inventing a routing consumer', () => {
    const current = snapshot()
    const next = withDecisionRuntimeDeployment(current, {
      entry,
      name: 'saved-only',
      device: 'auto',
      consumerId: '',
    })
    expect(next.routing).toEqual(current.routing)
    expect(next.recipes).toEqual(current.recipes)
    expect(decisionRuntimeConsumers(next).some((value) => value.deployment === 'saved-only')).toBe(
      false,
    )
  })

  it('preserves existing runtime tuning and pinned revisions when adding a consumer', () => {
    const current = snapshot()
    current.global = {
      ...current.global,
      model_catalog: {
        deployments: {
          managed: {
            provider: 'model_runtime',
            artifact: entry.id,
            revision: 'a'.repeat(40),
            device: 'rocm:1',
            profile: 'batching',
            input: { max_tokens: 2048, overflow: 'window' },
          },
        },
      },
    }
    const next = withDecisionRuntimeDeployment(current, {
      entry,
      name: 'managed',
      existingName: 'managed',
      device: 'rocm:0',
      consumerId: compatibleRuntimeConsumers(current)[0].id,
    })
    expect(
      (next.global?.model_catalog as { deployments: Record<string, unknown> }).deployments.managed,
    ).toEqual({
      provider: 'model_runtime',
      artifact: entry.id,
      revision: 'a'.repeat(40),
      device: 'rocm:0',
      profile: 'batching',
      input: { max_tokens: 2048, overflow: 'window' },
    })
  })

  it('rejects stale incompatible consumers, collisions, reserved names and removed deployments', () => {
    const current = snapshot()
    const consumerId = compatibleRuntimeConsumers(current)[0].id
    current.routing!.signals!.decision[0].question = { type: 'set' }
    const request = { entry, name: 'decider', device: 'auto', consumerId }
    expect(() => withDecisionRuntimeDeployment(current, request)).toThrow('no longer compatible')
    expect(() => withDecisionRuntimeDeployment(snapshot(), { ...request, name: 'other' })).toThrow(
      'already in use',
    )
    expect(() =>
      withDecisionRuntimeDeployment(snapshot(), { ...request, name: '@implicit' }),
    ).toThrow('reserved @')
    expect(() =>
      withDecisionRuntimeDeployment(snapshot(), { ...request, existingName: 'decider' }),
    ).toThrow('was removed')
  })

  it('does not introduce a revision pin when managing an existing unpinned runtime', () => {
    const current = snapshot()
    current.global = {
      model_catalog: {
        deployments: { managed: { provider: 'model_runtime', artifact: entry.id, device: 'cpu' } },
      },
    }
    const next = withDecisionRuntimeDeployment(current, {
      entry,
      name: 'managed',
      existingName: 'managed',
      device: 'auto',
      consumerId: '',
    })
    expect(
      (next.global?.model_catalog as { deployments: Record<string, unknown> }).deployments.managed,
    ).toEqual({ provider: 'model_runtime', artifact: entry.id, device: 'auto' })
  })
})
