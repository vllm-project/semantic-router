import { describe, expect, it } from 'vitest'

import {
  FEATURED_MODEL_PROVIDERS,
  MODEL_PROVIDERS,
  filterModelProviders,
  getModelProvider,
} from './modelProviderCatalog'
import { getModelProviderLogoSource } from './modelProviderLogoSupport'

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
    expect(MODEL_PROVIDERS.length).toBeGreaterThan(30)
    for (const provider of MODEL_PROVIDERS) {
      expect(provider.name).toBeTruthy()
      expect(provider.shortName).toBeTruthy()
      expect(provider.accent).toMatch(/^#/)
      expect(
        provider.runtimeProvider === 'openai' || provider.runtimeProvider === 'anthropic',
      ).toBe(true)
      expect(getModelProviderLogoSource(provider.id)).toBeTruthy()
    }
  })

  it('only offers one-key model APIs with complete discovery endpoints', () => {
    const modelAPIs = MODEL_PROVIDERS.filter((provider) => provider.category === 'Model APIs')
    expect(modelAPIs.length).toBeGreaterThan(20)
    for (const provider of modelAPIs) {
      expect(provider.baseUrl).toMatch(/^https:\/\//)
      expect(provider.modelsPath).toMatch(/^\//)
      expect(provider.apiKeyOptional).not.toBe(true)
    }
    expect(modelAPIs.map((provider) => provider.id)).toContain('sakana')
  })

  it('ships native Anthropic defaults and searchable external providers', () => {
    expect(getModelProvider('anthropic')).toMatchObject({
      apiFormat: 'anthropic',
      runtimeProvider: 'anthropic',
      authHeader: 'x-api-key',
      authPrefix: '',
      chatPath: '/v1/messages',
      modelsPath: '/v1/models',
    })
    expect(filterModelProviders('openrouter').map((provider) => provider.id)).toContain(
      'openrouter',
    )
    expect(MODEL_PROVIDERS.some((provider) => provider.id === 'openai-like')).toBe(false)
  })
})
