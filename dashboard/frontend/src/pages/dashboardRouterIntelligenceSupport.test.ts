import { describe, expect, it } from 'vitest'
import type { RouterModelInfo } from '../utils/routerRuntime'
import type { RouterConfig } from './dashboardPageTypes'
import {
  buildIntelligenceRoutingScopes,
  decisionModelHealth,
  getDecisionRuntimeSummary,
} from './dashboardRouterIntelligenceSupport'
import type { ModelRuntimeDeployment } from './decisionModelManagement'

const config: RouterConfig = {
  global: {
    model_catalog: {
      system: { decision_model: { deployment: 'judge' } },
      deployments: { judge: { artifact: 'vllm-sr/Vela-2.0-4B' } },
    },
  },
  entrypoints: [{ recipe: 'balanced', model_names: ['vllm-sr/balanced'] }],
  recipes: [
    {
      name: 'balanced',
      routing: {
        signals: {
          systemone: [{ name: 'task', question: { type: 'choice' } }],
          pii: [{ name: 'personal_data' }],
          context: [{ name: 'long_input' }],
        },
        projections: { scores: [{ name: 'effort_score' }], mappings: [{ name: 'effort' }] },
      },
    },
    {
      name: 'private',
      routing: { signals: { systemone: [{ name: 'task', question: { type: 'set' } }] } },
    },
  ],
}

const binding = (overrides: Partial<RouterModelInfo> = {}): RouterModelInfo => ({
  name: 'pii_classifier',
  type: 'pii_detection',
  loaded: true,
  state: 'ready',
  metadata: { deployment: 'judge', resource_id: 'shared', provider: 'model_runtime' },
  ...overrides,
})

describe('decision model runtime evidence', () => {
  it('counts one shared runtime and preserves all reported consumers', () => {
    expect(
      getDecisionRuntimeSummary(config, { models: [binding(), binding({ name: 'jailbreak' })] }),
    ).toEqual({ model: 'Vela-2.0-4B', state: 'ready', resources: 1, bindings: 2 })
  })

  it('does not infer readiness from config, inventory summaries, or artifact names', () => {
    expect(getDecisionRuntimeSummary(config).state).toBe('unreported')
    expect(
      getDecisionRuntimeSummary(config, {
        models: null,
        summary: { ready: true, loaded_models: 1 },
      }).state,
    ).toBe('unreported')
    expect(
      getDecisionRuntimeSummary(config, {
        models: [
          binding({ model_path: 'models/Vela-2.0-4B', metadata: { deployment: 'another-model' } }),
        ],
      }).state,
    ).toBe('unreported')
  })

  it('requires an exact decision-model deployment identity', () => {
    expect(
      getDecisionRuntimeSummary(config, {
        models: [binding({ metadata: { deployment: 'judge-other' } })],
      }).state,
    ).toBe('unreported')
    expect(
      getDecisionRuntimeSummary(config, {
        models: [binding({ metadata: { deployment: 'judge', provider: 'model_runtime' } })],
      }).state,
    ).toBe('ready')
  })

  it('keeps failed or pending consumers visible beside ready ones', () => {
    expect(
      getDecisionRuntimeSummary(config, {
        models: [binding(), binding({ name: 'guard', loaded: false, state: 'initializing' })],
      }),
    ).toEqual({ model: 'Vela-2.0-4B', state: 'attention', resources: 1, bindings: 2 })
    expect(
      getDecisionRuntimeSummary(config, {
        models: [binding({ loaded: false })],
      }).state,
    ).toBe('attention')
  })
})

describe('decision model deployment state', () => {
  const deployment = (overrides: Partial<ModelRuntimeDeployment> = {}): ModelRuntimeDeployment => ({
    name: 'judge',
    managed: true,
    served_name: 'judge',
    ready: true,
    state: 'ready',
    device: 'rocm:0',
    desired_replicas: 2,
    ready_replicas: 2,
    ...overrides,
  })

  it('reads the deployment the model runtime reports, as Models › Decision models does', () => {
    expect(decisionModelHealth(config, { deployments: [deployment()] })).toEqual({
      model: 'Vela-2.0-4B',
      deployment: 'judge',
      tone: 'success',
      label: 'Ready',
      detail: 'rocm:0 · 2 of 2 replicas ready',
    })
    expect(
      decisionModelHealth(config, {
        deployments: [deployment({ state: 'degraded', ready_replicas: 1 })],
      }),
    ).toMatchObject({
      tone: 'warning',
      label: 'Degraded capacity',
      detail: 'rocm:0 · 1 of 2 replicas ready',
    })
    expect(
      decisionModelHealth(config, {
        deployments: [deployment({ ready: false, state: 'loading_weights', reason: undefined })],
      }),
    ).toMatchObject({
      tone: 'warning',
      label: 'Needs attention',
      detail: 'The model runtime reports it loading weights',
    })
  })

  it('falls back to the Router bindings and says when nothing reports the deployment', () => {
    const other = { deployments: [deployment({ name: 'another' })] }
    expect(decisionModelHealth(config, other, { models: [binding()] })).toMatchObject({
      tone: 'success',
      label: 'Ready',
      detail: 'The Router reports it ready',
    })
    expect(
      decisionModelHealth(config, null, { models: [binding({ loaded: false })] }),
    ).toMatchObject({ tone: 'warning', label: 'Needs attention' })
    expect(decisionModelHealth(config, other)).toEqual({
      model: 'Vela-2.0-4B',
      deployment: 'judge',
      tone: 'neutral',
      label: 'Not reported',
      detail: 'The model runtime has not reported the judge deployment yet',
    })
  })
})

describe('configured routing overview', () => {
  it('keeps questions, signals and projections scoped to their recipe', () => {
    expect(buildIntelligenceRoutingScopes(config)).toEqual([
      {
        id: 'default',
        label: 'Default routing',
        entrypoints: ['vllm-sr/auto'],
        questions: [],
        signals: [],
        projections: [],
      },
      {
        id: 'balanced',
        label: 'balanced',
        entrypoints: ['vllm-sr/balanced'],
        questions: [{ name: 'task', kind: 'choice' }],
        signals: [
          { type: 'pii', names: ['personal_data'] },
          { type: 'context', names: ['long_input'] },
        ],
        projections: [
          { name: 'effort_score', kind: 'scores' },
          { name: 'effort', kind: 'mappings' },
        ],
      },
      {
        id: 'private',
        label: 'private',
        entrypoints: [],
        questions: [{ name: 'task', kind: 'set' }],
        signals: [],
        projections: [],
      },
    ])
  })

  it('reads canonical top-level routing and excludes empty or malformed groups', () => {
    expect(
      buildIntelligenceRoutingScopes({
        routing: {
          signals: {
            systemone: [{ name: 'difficulty', question: { type: 'score' } }],
            pii: [],
            context: [{}],
          },
        },
      })[0],
    ).toMatchObject({
      id: 'default',
      questions: [{ name: 'difficulty', kind: 'score' }],
      signals: [],
      projections: [],
    })
  })
})
