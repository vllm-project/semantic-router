import { describe, expect, it } from 'vitest'

import type { NormalizedModel } from './configPageSupport'
import {
  filterModelInventory,
  getModelDeleteBlocker,
  getModelReferenceCounts,
  getReasoningFamilyFilterOptions,
  validateModelStructuredFields,
  validateNewModelName,
} from './configPageModelInventory'

const models: NormalizedModel[] = Array.from({ length: 500 }, (_, index) => ({
  name: `local/model-${String(index + 1).padStart(3, '0')}`,
  provider_model_id: `physical-${index + 1}`,
  reasoning_family: index % 3 === 0 ? 'reasoning' : index % 3 === 1 ? 'standard' : undefined,
  tags: index % 10 === 0 ? ['premium'] : ['general'],
  endpoints: index % 4 === 0
    ? []
    : [{ name: `vllm-${index % 8}`, endpoint: 'vllm:8000', protocol: 'http', weight: 1 }],
}))

describe('model inventory filtering', () => {
  it('searches aliases, physical IDs, metadata, and endpoint labels across 500 models', () => {
    const result = filterModelInventory(models, {
      search: 'physical-420',
      reasoningFamily: 'all',
      endpointState: 'all',
      role: 'all',
      defaultModel: models[0].name,
    })

    expect(result.map((model) => model.name)).toEqual(['local/model-420'])
  })

  it('composes family, endpoint, and default-role filters', () => {
    const result = filterModelInventory(models, {
      search: '',
      reasoningFamily: 'reasoning',
      endpointState: 'missing',
      role: 'standard',
      defaultModel: models[0].name,
    })

    expect(result.length).toBeGreaterThan(0)
    expect(result.every((model) => model.reasoning_family === 'reasoning')).toBe(true)
    expect(result.every((model) => (model.endpoints?.length ?? 0) === 0)).toBe(true)
    expect(result.some((model) => model.name === models[0].name)).toBe(false)
  })

  it('builds stable filter options without duplicates', () => {
    expect(getReasoningFamilyFilterOptions(models)).toEqual(['reasoning', 'standard'])
  })

  it('protects default and decision-referenced models from destructive actions', () => {
    const referenceCounts = getModelReferenceCounts({
      routing: {
        decisions: [{
          name: 'premium-route',
          description: '',
          priority: 100,
          rules: { operator: 'AND', conditions: [] },
          modelRefs: [{ model: models[1].name, use_reasoning: false }],
        }],
      },
    })

    expect(getModelDeleteBlocker(models[0].name, models[0].name, referenceCounts)).toMatch(/different default/i)
    expect(getModelDeleteBlocker(models[1].name, models[0].name, referenceCounts)).toMatch(/1 routing decision/i)
    expect(getModelDeleteBlocker(models[2].name, models[0].name, referenceCounts)).toBeNull()
  })

  it('counts nested algorithm control-plane models once per routing decision', () => {
    const coordinator = models[10].name
    const synthesisModel = models[11].name
    const referenceCounts = getModelReferenceCounts({
      routing: {
        decisions: [{
          name: 'workflow-route',
          description: '',
          priority: 100,
          rules: { operator: 'AND', conditions: [] },
          modelRefs: [{ model: models[1].name, use_reasoning: false }],
          algorithm: {
            type: 'workflows',
            workflows: {
              planner: { model: coordinator },
              final: { model: synthesisModel },
              roles: [{ models: [coordinator, models[12].name] }],
            },
            fusion: {
              model: coordinator,
              analysis_models: [models[13].name],
            },
            remom: { synthesis_model: synthesisModel },
          },
        }],
      },
    })

    expect(referenceCounts.get(coordinator)).toBe(1)
    expect(referenceCounts.get(synthesisModel)).toBe(1)
    expect(referenceCounts.get(models[12].name)).toBe(1)
    expect(referenceCounts.get(models[13].name)).toBe(1)
    expect(getModelDeleteBlocker(coordinator, models[0].name, referenceCounts)).toMatch(/routing decision/i)
  })

  it('rejects duplicate names and malformed structured fields before saving', () => {
    expect(() => validateNewModelName('  local/model-001  ', models)).toThrow(/already exists/i)
    expect(validateNewModelName('  local/new-model  ', models)).toBe('local/new-model')
    expect(() => validateModelStructuredFields({ backend_refs: '{broken json' })).toThrow(/json array/i)
    expect(() => validateModelStructuredFields({ backend_refs: [{}] })).toThrow(/endpoint or base url/i)
    expect(() => validateModelStructuredFields({ backend_refs: [{ endpoint: 'localhost:8000', protocol: 'grpc' }] })).toThrow(/http or https/i)
    expect(() => validateModelStructuredFields({ backend_refs: [{ endpoint: 'localhost:8000', extra_headers: { valid: 1 } }] })).toThrow(/text key\/value pairs/i)
    expect(() => validateModelStructuredFields({ tags: 'premium,fast' })).toThrow(/list of text values/i)
    expect(() => validateModelStructuredFields({ capabilities: 'tools,vision' })).toThrow(/list of text values/i)
    expect(() => validateModelStructuredFields({ loras: [{ description: 'missing name' }] })).toThrow(/requires a name/i)
    expect(() => validateModelStructuredFields({ external_model_ids: { openai: '' } })).toThrow(/non-empty provider\/model id pairs/i)
    expect(() => validateModelStructuredFields({ pricing: { prompt_per_1m: -1 } })).toThrow(/zero or greater/i)
    expect(() => validateModelStructuredFields({ pricing: [] })).toThrow(/json object/i)
    expect(() => validateModelStructuredFields({
      backend_refs: [{ endpoint: 'localhost:8000', protocol: 'http', weight: 1, extra_headers: { 'X-Tenant': 'demo' } }],
      capabilities: ['tools', 'vision'],
      loras: [{ name: 'code-expert', description: 'Code specialization' }],
      external_model_ids: { openai: 'gpt-4.1' },
      tags: ['premium', 'fast'],
      pricing: { currency: 'USD', prompt_per_1m: 0.5 },
    })).not.toThrow()
  })
})
