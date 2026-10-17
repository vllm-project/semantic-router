import { describe, expect, it } from 'vitest'
import type { RouterModelInfo } from '../utils/routerRuntime'
import type { RouterConfig } from './dashboardPageTypes'
import {
  buildIntelligenceRoutingScopes,
  getDecisionRuntimeSummary,
} from './dashboardRouterIntelligenceSupport'

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
          decision: [{ name: 'task', question: { type: 'choice' } }],
          pii: [{ name: 'personal_data' }],
          context: [{ name: 'long_input' }],
        },
        projections: { scores: [{ name: 'effort_score' }], mappings: [{ name: 'effort' }] },
      },
    },
    {
      name: 'private',
      routing: { signals: { decision: [{ name: 'task', question: { type: 'set' } }] } },
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

describe('configured routing overview', () => {
  it('keeps questions, signals and projections scoped to their recipe', () => {
    expect(buildIntelligenceRoutingScopes(config)).toEqual([
      { id: 'default', label: 'Default routing', entrypoints: ['vllm-sr/auto'], questions: [], signals: [], projections: [] },
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
            decision: [{ name: 'difficulty', question: { type: 'score' } }],
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
