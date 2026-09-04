import { describe, expect, it } from 'vitest'

import generatedCatalog from '../generated/modelCatalog.json'
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
        },
        endpoints: [],
      }),
    ])
  })
})
