import { describe, expect, it } from 'vitest'
import {
  DECISION_MODEL_OPTIONS,
  configuredDecisionDeployment,
  configuredDecisionModel,
  withDecisionModel,
} from './decisionModelSupport'

describe('default decision deployment', () => {
  it('resolves the resource reference without interpreting the deployment name', () => {
    expect(configuredDecisionModel({})).toBe('Vela-2.0-0.3B')
    const config = {
      global: {
        model_catalog: {
          system: { decision_model: { deployment: 'judge' } },
          deployments: { judge: { artifact: 'vllm-sr/Vela-2.0-9B' } },
        },
      },
    }
    expect(configuredDecisionDeployment(config)).toBe('judge')
    expect(configuredDecisionModel(config)).toBe('Vela-2.0-9B')
    expect(
      configuredDecisionModel({
        global: {
          model_catalog: {
            system: { decision_model: { deployment: 'custom-judge' } },
          },
        },
      }),
    ).toBe('custom-judge')
  })

  it('offers all generic families with truthful native question types', () => {
    expect(new Set(DECISION_MODEL_OPTIONS.map((model) => model.family))).toEqual(
      new Set(['Vela 2.0', 'Decision 1.0', 'Decision 2.0', 'Decision 3.0']),
    )
    for (const model of DECISION_MODEL_OPTIONS) {
      expect(model.label).not.toMatch(/Qwen|Llama/i)
      expect(model.questionTypes).toContain('choice')
      if (model.family.startsWith('Decision')) expect(model.questionTypes).not.toContain('span')
    }
  })

  it('preserves existing resources and task overrides when changing the default', () => {
    const config = {
      version: 'v0.3',
      global: {
        model_catalog: {
          system: { decision_model: { deployment: 'judge' }, hazard: 'models/custom-hazard' },
          deployments: { judge: { artifact: 'vllm-sr/Vela-2.0-4B', device: 'cuda:1' } },
          modules: { prompt_guard: { model_binding: { deployment: 'judge' } } },
        },
      },
    }
    const chosen = withDecisionModel(config, 'Vela-2.0-0.8B')
    expect(configuredDecisionModel(chosen)).toBe('Vela-2.0-0.8B')
    expect(chosen.global.model_catalog.deployments.judge).toEqual(
      config.global.model_catalog.deployments.judge,
    )
    expect(chosen.global.model_catalog.modules).toEqual(config.global.model_catalog.modules)
    expect(chosen.global.model_catalog.system.hazard).toBe('models/custom-hazard')
    expect(config.global.model_catalog.system.decision_model).toEqual({ deployment: 'judge' })
  })

  it('reuses an existing resource and avoids overwriting occupied names', () => {
    const config = {
      global: {
        model_catalog: {
          deployments: {
            'vela-2-0-0-8b': { artifact: 'other/model' },
            existing: { artifact: 'vllm-sr/Vela-2.0-9B', device: 'cpu' },
          },
        },
      },
    }
    expect(configuredDecisionDeployment(withDecisionModel(config, 'Vela-2.0-9B'))).toBe('existing')
    expect(configuredDecisionDeployment(withDecisionModel(config, 'Vela-2.0-0.8B'))).toBe(
      'vela-2-0-0-8b-2',
    )
    expect(() => withDecisionModel(config, 'unsupported')).toThrow('catalog')
  })
})
