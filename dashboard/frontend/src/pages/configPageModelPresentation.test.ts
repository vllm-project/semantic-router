import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { providersServingCatalogModel } from './configPageConnectModelsDialogController'
import { collectDecisionTargetModels } from './configPageEntrypointsRecipesSupport'
import {
  filterModelInventoryFacets,
  modelBackendKind,
  type ModelInventoryFacets,
} from './configPageModelInventory'
import {
  modelDecisionReferences,
  modelDeployments,
  modelKind,
  modelPriceSummary,
  modelServedBy,
  providerDisplayName,
  providerLogoKey,
} from './configPageModelPresentation'
import type { ConfigData, NormalizedModel } from './configPageSupport'

const catalog = generatedCatalog as unknown as BuiltInModelCatalog

const config: ConfigData = {
  providers: {
    models: [
      { name: 'local', backend_refs: [{ provider: 'vllm', endpoint: 'vllm:8000' }] },
      { name: 'native', deployment: 'decision-kai' } as never,
    ],
  },
  routing: {
    decisions: [
      {
        name: 'route',
        description: '',
        priority: 1,
        rules: { operator: 'AND', conditions: [] },
        modelRefs: [{ model: 'local', use_reasoning: false }],
      },
    ],
  },
  recipes: [
    {
      name: 'privacy',
      routing: {
        decisions: [
          {
            name: 'private',
            description: '',
            priority: 2,
            rules: { operator: 'AND', conditions: [] },
            modelRefs: [],
            algorithm: { type: 'confidence', confidence: { escalation_model: 'local' } },
          },
        ],
      },
    },
  ],
  global: {
    model_catalog: {
      deployments: {
        'decision-kai': { provider: 'model_runtime', artifact: 'vllm-sr/kai' },
        remote: { provider: 'openai_chat', model: 'local' },
      },
    },
  },
}

const models: NormalizedModel[] = [
  {
    name: 'local',
    endpoints: [{ name: 'primary', endpoint: 'vllm:8000', protocol: 'http', weight: 1 }],
    backend_refs: [{ provider: 'vllm', endpoint: 'vllm:8000' }],
    reasoning_family: 'qwen3',
  },
  { name: 'native', endpoints: [] },
  { name: 'card-only', endpoints: [], tags: ['draft'] },
]

describe('backend model presentation', () => {
  it('finds the shared deployment a model names and the deployments that name it', () => {
    expect(modelDeployments(config, 'native')).toEqual({
      referenced: {
        name: 'decision-kai',
        deployment: { provider: 'model_runtime', artifact: 'vllm-sr/kai' },
      },
      naming: [],
    })
    expect(modelDeployments(config, 'local')).toEqual({
      referenced: null,
      naming: [{ name: 'remote', deployment: { provider: 'openai_chat', model: 'local' } }],
    })
  })

  it('lists the decisions of every recipe that can send a model traffic', () => {
    const references = modelDecisionReferences(config, 'local', collectDecisionTargetModels)
    expect(references.map((reference) => [reference.scopeLabel, reference.decision.name])).toEqual([
      ['Default routing', 'route'],
      ['privacy', 'private'],
    ])
  })

  it('formats prices and resolves catalog provider names and logos', () => {
    expect(modelPriceSummary({ currency: 'USD', prompt_per_1m: 3.5 })).toEqual({
      input: '$3.50',
      output: null,
    })
    expect(providerDisplayName('openai-compatible', catalog)).toBe('OpenAI Compatible')
    expect(providerLogoKey('anthropic-compatible', catalog)).toBe('anthropic')
    expect(providerLogoKey('unknown-provider', catalog)).toBe('unknown-provider')
  })

  it('offers only the providers that serve a proposed catalog model', () => {
    expect(providersServingCatalogModel(catalog, 'anthropic/claude-opus-5.5')).toEqual([
      'anthropic',
    ])
    expect(providersServingCatalogModel(catalog, undefined)).toEqual([])
  })
})

describe('inventory facets', () => {
  const deployments = new Map([['native', 'decision-kai']])
  const facets: ModelInventoryFacets = {
    search: '',
    providers: [],
    families: [],
    backends: [],
    roles: [],
    defaultModel: 'local',
    deployments,
  }

  it('tells endpoints, shared deployments and models without a backend apart', () => {
    expect(models.map((model) => modelBackendKind(model, deployments))).toEqual([
      'endpoints',
      'deployment',
      'none',
    ])
  })

  it('matches any value within a facet and every facet together', () => {
    const names = (overrides: Partial<ModelInventoryFacets>) =>
      filterModelInventoryFacets(models, { ...facets, ...overrides }).map((model) => model.name)
    expect(names({ backends: ['deployment', 'none'] })).toEqual(['native', 'card-only'])
    expect(names({ providers: ['vllm'] })).toEqual(['local'])
    expect(names({ families: ['__unassigned__'] })).toEqual(['native', 'card-only'])
    expect(names({ roles: ['default'] })).toEqual(['local'])
    expect(names({ roles: ['standard'], search: 'draft' })).toEqual(['card-only'])
    expect(names({ search: 'decision-kai' })).toEqual(['native'])
  })
})

describe('model kind and serving', () => {
  const model = (overrides: Partial<NormalizedModel>): NormalizedModel => ({
    name: 'm',
    endpoints: [],
    ...overrides,
  })

  it('names what a model is for: output first, then reasoning, then input', () => {
    expect(modelKind(model({ api_format: 'systemone' }), catalog)).toBe('decision')
    expect(modelKind(model({ api_format: 'images' }), catalog)).toBe('image')
    expect(modelKind(model({ capabilities: ['embedding'] }), catalog)).toBe('embedding')
    expect(
      modelKind(
        model({
          reasoning_family: 'qwen3',
          modalities: { input: ['text', 'image'], output: ['text'] },
        }),
        catalog,
      ),
    ).toBe('reasoning')
    expect(
      modelKind(model({ modalities: { input: ['text', 'image'], output: ['text'] } }), catalog),
    ).toBe('vision')
    expect(modelKind(model({}), catalog)).toBe('chat')
  })

  it('falls back to the catalog card for capabilities and virtual models', () => {
    const virtual = catalog.models.find((entry) => entry.kind === 'virtual')!
    expect(modelKind(model({ catalog: virtual.id }), catalog)).toBe('mixture')
    const reasoning = catalog.models.find(
      (entry) => entry.kind === 'physical' && entry.capabilities.includes('reasoning'),
    )!
    expect(modelKind(model({ catalog: reasoning.id }), catalog)).toBe('reasoning')
  })

  it('says who serves a model on one line', () => {
    const endpoint = { name: 'a', endpoint: 'vllm:8000', protocol: 'http' as const, weight: 1 }
    expect(
      modelServedBy(
        model({
          endpoints: [endpoint, { ...endpoint, name: 'b' }],
          backend_refs: [{ provider: 'vllm' }],
        }),
        undefined,
        catalog,
      ),
    ).toBe(`${providerDisplayName('vllm', catalog)} ×2`)
    expect(modelServedBy(model({}), 'decision-kai', catalog)).toBe('Model runtime · decision-kai')
    expect(modelServedBy(model({}), undefined, catalog)).toBeNull()
  })
})
