import { describe, expect, it } from 'vitest'

import {
  buildConnectedProviderModel,
  resolveConnectedModelName,
} from './configPageConnectModelSupport'
import generatedCatalog from '../generated/modelCatalog.json'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  mergeModelInventory,
  modelInventoryForProvider,
  modelsForProvider,
} from './configPageConnectModelsDialogController'

describe('connected model naming', () => {
  it('keeps the upstream model id when the public namespace is free', () => {
    expect(resolveConnectedModelName('', 'vllm', 'local/qwen', new Set())).toBe('local/qwen')
  })

  it('scopes an upstream model id that conflicts with a virtual model', () => {
    const reserved = new Set(['vllm-sr/mom-v1-blend'])

    expect(resolveConnectedModelName('', 'vllm', 'vllm-sr/mom-v1-blend', reserved)).toBe(
      'vllm/vllm-sr/mom-v1-blend',
    )
  })

  it('creates a stable unique name when the provider-scoped name is also occupied', () => {
    const reserved = new Set(['blend', 'vllm/blend', 'vllm/blend-2'])

    expect(resolveConnectedModelName('', 'vllm', 'blend', reserved)).toBe('vllm/blend-3')
  })
})

describe('catalog-backed model matching', () => {
  it('does not materialize removed provider models from provider discovery', () => {
    const models = modelsForProvider(
      generatedCatalog as unknown as BuiltInModelCatalog,
      'anthropic',
    )
    expect(models.get('claude-sonnet-4-20250514')).toBeUndefined()
    expect(models.get('claude-sonnet-5')).toBe('anthropic/claude-sonnet-5')
  })

  it('seeds the provider picker from the built-in provider inventory', () => {
    const models = modelInventoryForProvider(
      generatedCatalog as unknown as BuiltInModelCatalog,
      'openai',
    )

    expect(models).toContain('gpt-5.6-sol')
    expect(models).toContain('gpt-5.4')
  })

  it('keeps built-in choices when provider discovery adds live models', () => {
    expect(mergeModelInventory(['gpt-5.6-sol', 'gpt-5.4'], ['gpt-5.4', 'custom-preview'])).toEqual([
      'gpt-5.6-sol',
      'gpt-5.4',
      'custom-preview',
    ])
  })
})

describe('quick connect provider model payloads', () => {
  const common = {
    name: 'frontier',
    providerModelID: 'gpt-5.6-sol',
    providerID: 'openai',
    providerAPIFormat: 'openai',
    baseURL: 'https://api.openai.com/v1',
    apiKey: '',
  }

  it('leaves catalog binding identity and protocol to the materializer', () => {
    const model = buildConnectedProviderModel({
      ...common,
      catalog: 'openai/gpt-5.6-sol',
      reasoningFamily: 'gpt',
    })

    expect(model).toMatchObject({
      name: 'frontier',
      catalog: 'openai/gpt-5.6-sol',
      backend_refs: [{ provider: 'openai' }],
    })
    expect(model.provider_model_id).toBeUndefined()
    expect(model.api_format).toBeUndefined()
    expect(model.reasoning).toBeUndefined()
  })

  it('keeps provider defaults for a custom model without catalog metadata', () => {
    expect(
      buildConnectedProviderModel({
        ...common,
        name: 'custom',
        providerModelID: 'custom-preview',
        reasoningFamily: 'gpt',
      }),
    ).toMatchObject({
      name: 'custom',
      reasoning: { family: 'gpt' },
      provider_model_id: 'custom-preview',
      api_format: 'openai',
    })
  })
})
