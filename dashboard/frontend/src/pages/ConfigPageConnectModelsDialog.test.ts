import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { ModelOption } from './ConfigPageConnectModelsDialogView'
import {
  buildConnectedProviderModel,
  resolveConnectedModelName,
} from './configPageConnectModelSupport'
import {
  type ConnectModelsDialogController,
  mergeModelInventory,
  missingRequiredProviderModelID,
  modelInventoryForProvider,
  modelsForProvider,
  providerModelIDRequirements,
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
  const catalog = generatedCatalog as unknown as BuiltInModelCatalog

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

  it('projects operator-defined deployment-name requirements from provider bindings', () => {
    const requirements = providerModelIDRequirements(catalog, 'microsoft-foundry')

    expect(requirements.get('mai-thinking-1')).toBe('deployment_name')
    expect(providerModelIDRequirements(catalog, 'openai').has('gpt-5.6-sol')).toBe(false)
  })

  it('requires a non-blank provider model ID only for selected restricted bindings', () => {
    const selected = new Set(['mai-thinking-1', 'custom-model'])
    const requirements = new Map([['mai-thinking-1', 'deployment_name']])

    expect(missingRequiredProviderModelID(selected, requirements, new Map())).toBe('mai-thinking-1')
    expect(
      missingRequiredProviderModelID(
        selected,
        requirements,
        new Map([['mai-thinking-1', ' operator-mai-production ']]),
      ),
    ).toBeUndefined()
  })

  it('renders the deployment name as a required field for a selected restricted binding', () => {
    const controller = {
      advanced: { namePrefix: '' },
      resolvedModelNames: new Map([['mai-thinking-1', 'mai-thinking-1']]),
      catalogModels: new Map([['mai-thinking-1', 'microsoft/mai-thinking-1']]),
      providerModelIdRequirements: new Map([['mai-thinking-1', 'deployment_name']]),
      modelDisplayNames: new Map([['microsoft/mai-thinking-1', 'MAI-Thinking-1']]),
      selected: new Set(['mai-thinking-1']),
      providerModelIds: new Map<string, string>(),
      setSelected: () => undefined,
      setProviderModelId: () => undefined,
    } as unknown as ConnectModelsDialogController

    const markup = renderToStaticMarkup(
      createElement(ModelOption, { controller, model: 'mai-thinking-1' }),
    )

    expect(markup).toContain('Deployment name')
    expect(markup).toContain('required=""')
    expect(markup).not.toContain('disabled=""')
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

  it('preserves an operator deployment name for a restricted catalog binding', () => {
    expect(
      buildConnectedProviderModel({
        ...common,
        catalog: 'microsoft/mai-thinking-1',
        catalogProviderModelID: ' operator-mai-production ',
        providerID: 'microsoft-foundry',
        baseURL: 'https://foundry.example.test',
      }),
    ).toMatchObject({
      name: 'frontier',
      catalog: 'microsoft/mai-thinking-1',
      provider_model_id: 'operator-mai-production',
      backend_refs: [
        {
          provider: 'microsoft-foundry',
          base_url: 'https://foundry.example.test',
        },
      ],
    })
  })
})
