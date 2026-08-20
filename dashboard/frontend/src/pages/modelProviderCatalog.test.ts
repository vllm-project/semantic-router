import { describe, expect, it } from 'vitest'

import {
  FEATURED_MODEL_PROVIDERS,
  MODEL_PROVIDERS,
  filterModelProviders,
  getModelProvider,
} from './modelProviderCatalog'

describe('modelProviderCatalog', () => {
  it('keeps the requested runtime providers first', () => {
    expect(FEATURED_MODEL_PROVIDERS.map((provider) => provider.id)).toEqual([
      'vllm',
      'sglang',
      'amd-atom',
      'openai-compatible',
    ])
  })

  it('has a unique identity and visual mark for every provider', () => {
    expect(new Set(MODEL_PROVIDERS.map((provider) => provider.id)).size).toBe(
      MODEL_PROVIDERS.length,
    )
    expect(MODEL_PROVIDERS.length).toBeGreaterThan(60)
    for (const provider of MODEL_PROVIDERS) {
      expect(provider.name).toBeTruthy()
      expect(provider.shortName).toBeTruthy()
      expect(provider.accent).toMatch(/^#/)
      expect(
        provider.runtimeProvider === 'openai' || provider.runtimeProvider === 'anthropic',
      ).toBe(true)
    }
  })

  it('ships native Anthropic defaults and searchable external providers', () => {
    expect(getModelProvider('anthropic')).toMatchObject({
      apiFormat: 'anthropic',
      runtimeProvider: 'anthropic',
      authHeader: 'x-api-key',
      authPrefix: '',
      chatPath: '/v1/messages',
    })
    expect(filterModelProviders('openrouter').map((provider) => provider.id)).toContain(
      'openrouter',
    )
    expect(MODEL_PROVIDERS.some((provider) => provider.id === 'openai-like')).toBe(false)
  })
})
