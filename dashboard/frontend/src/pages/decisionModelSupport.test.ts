import { describe, expect, it } from 'vitest'

import { DECISION_MODELS, configuredDecisionModel, withDecisionModel } from './decisionModelSupport'

describe('decision model', () => {
  it('reads the configured decision model as the Router does', () => {
    expect(configuredDecisionModel({})).toBe('Vela-2.0-0.3B')
    expect(
      configuredDecisionModel({
        global: { model_catalog: { system: { decision_model: ' vela-2.0-9b ' } } },
      }),
    ).toBe('Vela-2.0-9B')
    expect(
      configuredDecisionModel({
        global: { model_catalog: { system: { decision_model: 'Decision-2.0-Kai' } } },
      }),
    ).toBe('Vela-2.0-0.3B')
    expect(DECISION_MODELS).toEqual([
      'Vela-2.0-0.3B',
      'Vela-2.0-0.8B',
      'Vela-2.0-4B',
      'Vela-2.0-9B',
      'Vela-1.0',
    ])
  })

  it('writes the decision model without touching the rest of the config', () => {
    const config = {
      version: 'v0.3',
      global: {
        model_catalog: {
          system: { hazard: 'models/Vela-1.0-Encoder-307M-Hazard' },
          modules: { prompt_guard: {} },
        },
      },
    }
    const chosen = withDecisionModel(config, 'Vela-2.0-0.8B')
    expect(chosen.global.model_catalog).toEqual({
      system: { hazard: 'models/Vela-1.0-Encoder-307M-Hazard', decision_model: 'Vela-2.0-0.8B' },
      modules: { prompt_guard: {} },
    })
    expect(config.global.model_catalog.system).toEqual({
      hazard: 'models/Vela-1.0-Encoder-307M-Hazard',
    })
  })
})
