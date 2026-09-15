import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { getNormalizedModels } from './configPageModelNormalization'
import type { ConfigData } from './configPageSupport'

const catalog = generatedCatalog as unknown as BuiltInModelCatalog

describe('config page model normalization', () => {
  it('joins a request alias to its catalog card and canonical-name override', () => {
    const builtIn = catalog.models[0]
    const config: ConfigData = {
      providers: {
        models: [
          {
            name: 'frontier',
            catalog: builtIn.id,
            backend_refs: [{ name: 'primary', provider: 'vllm' }],
          },
        ],
      },
      routing: {
        modelCards: [{ name: builtIn.id, description: 'Approved production override' }],
      },
    }

    expect(getNormalizedModels(config, true, catalog)).toEqual([
      expect.objectContaining({
        name: 'frontier',
        catalog: builtIn.id,
        description: 'Approved production override',
        capabilities: builtIn.capabilities,
        card_override: expect.objectContaining({ name: builtIn.id }),
      }),
    ])
  })

  it('keeps custom model cards optional and supports inline reasoning', () => {
    const config: ConfigData = {
      providers: {
        models: [
          {
            name: 'private-reasoner',
            reasoning: {
              type: 'chat_template_kwargs',
              parameter: 'think_mode',
              modes: ['enabled', 'disabled'],
              default_mode: 'enabled',
            },
          },
        ],
      },
    }

    expect(getNormalizedModels(config, true, catalog)).toEqual([
      expect.objectContaining({
        name: 'private-reasoner',
        reasoning: {
          type: 'chat_template_kwargs',
          parameter: 'think_mode',
          modes: ['enabled', 'disabled'],
          default_mode: 'enabled',
        },
        endpoints: [],
      }),
    ])
  })

  it('narrows a model reasoning contract to the selected provider binding', () => {
    const builtIn = catalog.models.find((model) => model.id === 'minimax/minimax-m3')
    expect(builtIn).toBeDefined()
    const config: ConfigData = {
      providers: {
        models: [
          {
            name: 'minimax-production',
            catalog: builtIn!.id,
            backend_refs: [{ name: 'official', provider: 'minimax' }],
          },
        ],
      },
    }

    expect(getNormalizedModels(config, true, catalog)).toEqual([
      expect.objectContaining({
        reasoning_modes: ['disabled', 'adaptive', 'enabled'],
      }),
    ])
  })

  it('narrows reasoning efforts to the selected provider protocol', () => {
    const builtIn = catalog.models.find((model) => model.id === 'openai/gpt-6-astra')
    expect(builtIn).toBeDefined()
    const config: ConfigData = {
      providers: {
        models: [
          {
            name: 'astra-chat',
            catalog: builtIn!.id,
            backend_refs: [{ name: 'chat', provider: 'openai' }],
          },
          {
            name: 'astra-responses',
            catalog: builtIn!.id,
            api_format: 'responses',
            backend_refs: [{ name: 'responses', provider: 'openai' }],
          },
        ],
      },
    }

    const [chat, responses] = getNormalizedModels(config, true, catalog)
    expect(chat.reasoning_efforts).toEqual(['low', 'medium', 'high', 'xhigh'])
    expect(responses.reasoning_efforts).toEqual(['low', 'medium', 'high', 'xhigh', 'max'])
  })
})
