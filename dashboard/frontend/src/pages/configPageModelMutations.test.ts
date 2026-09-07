import { describe, expect, it } from 'vitest'

import generatedCatalog from '../generated/modelCatalog.json'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { buildAddedModelConfig, buildConnectedModelsConfig } from './configPageModelMutations'
import { newModelFormData } from './configPageModelsSectionSupport'
import type { ConfigData } from './configPageSupport'
import { modelProviderPresetsFromCatalog } from './modelProviderCatalog'

const serialized = (value: ConfigData): ConfigData => JSON.parse(JSON.stringify(value))

describe('model pricing mutations', () => {
  const cases: Array<[string, boolean, ConfigData]> = [
    ['canonical', true, { providers: { models: [] } }],
    ['legacy', false, { model_config: {} }],
  ]

  it.each(cases)(
    'does not serialize untouched pricing for %s models',
    (_label, canonical, config) => {
      const result = serialized(
        buildAddedModelConfig(config, 'private/model', newModelFormData(), canonical),
      )
      const model = canonical ? result.providers?.models[0] : result.model_config?.['private/model']

      expect(model).toBeDefined()
      expect(model).not.toHaveProperty('pricing')
    },
  )

  it('serializes explicitly entered zero rates for a canonical model', () => {
    const pricing = {
      currency: 'USD',
      prompt_per_1m: 0,
      cached_input_per_1m: 0,
      completion_per_1m: 0,
    }
    const result = serialized(
      buildAddedModelConfig(
        { providers: { models: [] } },
        'private/model',
        { ...newModelFormData(), pricing },
        true,
      ),
    )

    expect(result.providers?.models[0].pricing).toEqual(pricing)
  })

  it.each([
    ['canonical', true, { providers: { models: [] } } as ConfigData],
    ['legacy', false, { model_config: {} } as ConfigData],
  ])(
    'persists an explicit free prompt rate with a safe currency for %s models',
    (_label, canonical, config) => {
      const result = serialized(
        buildAddedModelConfig(
          config,
          'private/model',
          { ...newModelFormData(), pricing: { prompt_per_1m: 0 } },
          canonical,
        ),
      )
      const model = canonical ? result.providers?.models[0] : result.model_config?.['private/model']

      expect(model?.pricing).toEqual({ prompt_per_1m: 0, currency: 'USD' })
    },
  )
})

describe('catalog-backed model connection mutations', () => {
  it('writes an operator deployment name in the CLI-compatible catalog shape', () => {
    const catalog = generatedCatalog as unknown as BuiltInModelCatalog
    const provider = modelProviderPresetsFromCatalog(catalog.providers).find(
      (candidate) => candidate.id === 'microsoft-foundry',
    )
    expect(provider).toBeDefined()

    const result = serialized(
      buildConnectedModelsConfig({ version: 'v0.3' }, [], {
        provider: provider!,
        baseUrl: 'https://foundry.example.test',
        apiKey: '',
        modelIds: ['mai-thinking-1'],
        modelNames: { 'mai-thinking-1': 'mai' },
        catalogModels: { 'mai-thinking-1': 'microsoft/mai-thinking-1' },
        providerModelIds: { 'mai-thinking-1': ' operator-mai-production ' },
        metadata: {},
      }),
    )

    expect(result.providers?.models).toEqual([
      {
        name: 'mai',
        catalog: 'microsoft/mai-thinking-1',
        provider_model_id: 'operator-mai-production',
        backend_refs: [
          {
            name: 'microsoft-foundry-primary',
            provider: 'microsoft-foundry',
            base_url: 'https://foundry.example.test',
          },
        ],
      },
    ])
  })
})
