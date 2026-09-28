import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  buildAddedModelConfig,
  buildConnectedModelsConfig,
  buildEditedModelConfig,
} from './configPageModelMutations'
import { getNormalizedModels } from './configPageModelNormalization'
import { modelFormDataForSave } from './configPageModelFormSupport'
import { validateModelStructuredFields } from './configPageModelInventory'
import { editModelFormData, newModelFormData } from './configPageModelsSectionSupport'
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

describe('effective model edit round trips', () => {
  const catalog = generatedCatalog as unknown as BuiltInModelCatalog
  const builtIn = catalog.models.find((model) => model.id === 'zai/glm-5.3-flash')!
  const initial: ConfigData = {
    providers: {
      models: [
        {
          name: 'local',
          catalog: builtIn.id,
          backend_refs: [{ provider: 'vllm', endpoint: 'localhost:8000' }],
        },
      ],
    },
    routing: {
      modelCards: [
        { name: builtIn.id, description: 'Deployment description', max_output_tokens: 8192 },
      ],
    },
  }
  const normalized = (config = initial) => getNormalizedModels(config, true, catalog)[0]

  it('shows inherited metadata and API format without persisting them on no-op save', () => {
    const model = normalized()
    const form = editModelFormData(model)
    expect(form.param_size).toBe(builtIn.parameter_size)
    expect(form.context_window_size).toBe(builtIn.limits!.context_window_size)
    expect(form.capabilities).toEqual(builtIn.capabilities)
    expect(form.tags).toEqual(builtIn.tags ?? [])
    expect(model.reasoning_family).toBe('glm-5.3')
    expect(form.reasoning_family).toBe('')
    expect(model.api_format).toBe('openai')
    expect(form.api_format).toBe('')
    validateModelStructuredFields(form)
    expect(serialized(buildEditedModelConfig(initial, model, form, true))).toEqual(initial)
  })

  it('writes only changed metadata, preserving fields that the form does not edit', () => {
    const model = normalized()
    const result = serialized(
      buildEditedModelConfig(
        initial,
        model,
        {
          ...editModelFormData(model),
          param_size: 'custom-size',
        },
        true,
      ),
    )
    expect(result.routing?.modelCards).toEqual([
      { ...initial.routing!.modelCards![0], param_size: 'custom-size' },
    ])
    expect(result.providers?.models).toEqual(initial.providers?.models)
  })

  it('clears an override to inherit while preserving other authored metadata', () => {
    const model = normalized()
    const result = serialized(
      buildEditedModelConfig(
        initial,
        model,
        { ...editModelFormData(model), description: '' },
        true,
      ),
    )
    expect(result.routing?.modelCards).toEqual([{ name: builtIn.id, max_output_tokens: 8192 }])
    expect(normalized(result).description).toBe(builtIn.description)
  })

  it('keeps an explicit format until the user chooses inherit', () => {
    const config = structuredClone(initial)
    config.providers!.models[0].api_format = 'responses'
    const model = normalized(config)
    const form = editModelFormData(model)
    expect(form.api_format).toBe('responses')
    expect(
      serialized(buildEditedModelConfig(config, model, form, true)).providers?.models[0].api_format,
    ).toBe('responses')
    expect(
      serialized(buildEditedModelConfig(config, model, { ...form, api_format: '' }, true)).providers
        ?.models[0],
    ).not.toHaveProperty('api_format')
  })

  it('does not carry inherited fields to a newly selected catalog card', () => {
    const model = normalized()
    const nextID = catalog.models.find(
      (entry) => entry.kind === 'physical' && entry.id !== builtIn.id,
    )!.id
    const result = serialized(
      buildEditedModelConfig(
        initial,
        model,
        { ...editModelFormData(model), catalog: nextID },
        true,
      ),
    )
    expect(result.routing?.modelCards).toEqual([])
    expect(result.providers?.models[0].catalog).toBe(nextID)
  })

  it('drops hidden custom reasoning when selecting a built-in model', () => {
    const form = modelFormDataForSave({
      ...newModelFormData(),
      catalog: builtIn.id,
      reasoning_family: 'deepseek',
      reasoning_type: 'reasoning_effort',
      reasoning_parameter: 'effort',
    })
    expect(() => validateModelStructuredFields(form)).not.toThrow()
    const result = serialized(
      buildAddedModelConfig({ providers: { models: [] } }, 'local', form, true),
    )
    expect(result.providers?.models[0]).not.toHaveProperty('reasoning')
  })
})
