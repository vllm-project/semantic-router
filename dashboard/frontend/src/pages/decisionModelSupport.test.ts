import { describe, expect, it } from 'vitest'
import {
  DECISION_CATALOG_MODELS,
  DECISION_MODEL_OPTIONS,
  compareDecisionFamilies,
  configuredDecisionDeployment,
  configuredDecisionModel,
  decisionDeviceClass,
  decisionFamilyLabel,
  decisionModelLabel,
  formatParameters,
  withDecisionModel,
} from './decisionModelSupport'

describe('default decision deployment', () => {
  it('resolves the resource reference without interpreting the deployment name', () => {
    expect(configuredDecisionModel({})).toBe('vela3-fast')
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
    const families = new Set(DECISION_MODEL_OPTIONS.map((model) => model.family))
    for (const family of ['Vela 3.0', 'Vela 2.0', 'Decision 1.0', 'Decision 2.0', 'Decision 3.0'])
      expect(families).toContain(family)
    for (const model of DECISION_MODEL_OPTIONS) {
      expect(model.label).not.toMatch(/Qwen|Llama/i)
      expect(model.questionTypes).toContain('choice')
      expect(model.revision).toMatch(/^[0-9a-f]{40}$/)
      if (model.family.startsWith('Decision')) expect(model.questionTypes).not.toContain('span')
    }
  })

  it('lists every Vela and Decision release the registry projects, grouped by family', () => {
    const names = DECISION_CATALOG_MODELS.map((model) => model.name)
    for (const name of [
      'vela3-fast',
      'vela3-fast-vision',
      'vela3-lite',
      'vela3-mini',
      'vela3-flash',
      'vela3-turbo',
      'Vela-2.0-0.3B',
      'Vela-2.0-9B',
      'd3-ultra',
      'd3-turbo',
      'Decision-2.0-Vega-27B',
      'Decision-1.0-Route-0.6B',
      'vela3.1-fast',
      'vela3.1-pico-150m',
      'vela3.1-pico-380m',
    ])
      expect(names).toContain(name)
    const order = [...new Set(DECISION_CATALOG_MODELS.map((model) => model.family))]
    expect(order.slice(0, 6)).toEqual([
      'Vela 3.1',
      'Vela 3.0',
      'Vela 2.0',
      'Decision 3.0',
      'Decision 2.0',
      'Decision 1.0',
    ])
    const vision = DECISION_CATALOG_MODELS.find((model) => model.name === 'vela3-fast-vision')
    expect(vision).toMatchObject({ servable: false, questionTypes: [] })
    expect(DECISION_MODEL_OPTIONS.some((model) => model.name === 'vela3-fast-vision')).toBe(false)
    expect(DECISION_CATALOG_MODELS.find((model) => model.name === 'vela3-mini')).toMatchObject({
      family: 'Vela 3.0',
      familyId: 'vela3_d3',
      questionTypes: ['choice', 'noul', 'score', 'span'],
      modalities: ['text', 'image', 'video'],
      device: 'gpu',
    })
  })

  it('serves the Vela 3.1 models the registry pins', () => {
    for (const name of ['vela3.1-fast', 'vela3.1-pico-150m', 'vela3.1-pico-380m']) {
      expect(DECISION_CATALOG_MODELS.find((model) => model.name === name)).toMatchObject({
        family: 'Vela 3.1',
        servable: true,
      })
      expect(DECISION_MODEL_OPTIONS).toContainEqual(expect.objectContaining({ name }))
    }
    expect(
      DECISION_CATALOG_MODELS.find((model) => model.name === 'vela3.1-pico-150m'),
    ).toMatchObject({
      parameters: 153_207_299,
      minMemoryGiB: 2,
    })
    expect(DECISION_CATALOG_MODELS.find((model) => model.name === 'vela3.1-fast')).toMatchObject({
      parameters: 1_927_527_449,
      minMemoryGiB: 15,
    })
  })

  it('derives labels, families, sizes and device hints from release facts', () => {
    expect(decisionModelLabel('vela3-fast')).toBe('Vela 3.0 fast')
    expect(decisionModelLabel('vela3.1-pico-150m')).toBe('Vela 3.1 pico 150M')
    expect(decisionModelLabel('Vela-2.0-0.3B')).toBe('Vela 2.0 0.3B')
    expect(decisionModelLabel('Decision-2.0-Kai-0.6B')).toBe('Decision 2.0 Kai 0.6B')
    expect(decisionModelLabel('d3-ultra')).toBe('Decision 3.0 d3-ultra')
    expect(
      decisionFamilyLabel({ name: 'vela3.1-pico-380m', family: 'vela3_pico', familyLabel: '' }),
    ).toBe('Vela 3.1')
    expect(
      decisionFamilyLabel({ name: 'd3-lite', family: 'decision3', familyLabel: 'Decision 3.0' }),
    ).toBe('Decision 3.0')
    expect(compareDecisionFamilies('Vela 3.1', 'Vela 3.0')).toBeLessThan(0)
    expect(compareDecisionFamilies('Decision 3.0', 'Vela 2.0')).toBeGreaterThan(0)
    expect(compareDecisionFamilies('Custom', 'Decision 1.0')).toBeGreaterThan(0)
    expect(formatParameters(1_518_582_809)).toBe('1.5B')
    expect(formatParameters(26_086_635_760)).toBe('26B')
    expect(formatParameters(153_207_299)).toBe('153M')
    expect(decisionDeviceClass({ backbone: 'ultrabert', minMemoryGiB: 12 })).toBe('cpu')
    expect(decisionDeviceClass({ backbone: 'qwen3_5_text', minMemoryGiB: 8 })).toBe('flexible')
    expect(decisionDeviceClass({ backbone: 'qwen3_5_text', minMemoryGiB: 16 })).toBe('gpu')
  })

  it('preserves existing resources and task overrides when changing the default', () => {
    const config = {
      version: 'v0.5',
      global: {
        model_catalog: {
          system: { decision_model: { deployment: 'judge' } },
          deployments: { judge: { artifact: 'vllm-sr/Vela-2.0-4B', device: 'cuda:1' } },
          bindings: { jailbreak: 'judge' },
        },
      },
    }
    const chosen = withDecisionModel(config, 'Vela-2.0-0.8B')
    expect(configuredDecisionModel(chosen)).toBe('Vela-2.0-0.8B')
    expect(chosen.global.model_catalog.deployments.judge).toEqual(
      config.global.model_catalog.deployments.judge,
    )
    expect(chosen.global.model_catalog.bindings).toEqual({ jailbreak: 'judge' })
    expect(chosen.global.model_catalog.system).toEqual({
      decision_model: { deployment: 'vela-2-0-0-8b' },
    })
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
