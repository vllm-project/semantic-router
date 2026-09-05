import { describe, expect, it } from 'vitest'

import { findModelProviderPreset, modelProviderCatalog } from './modelProviderCatalog'

describe('model provider catalog', () => {
  it('keeps private endpoints explicit and hosted APIs ready to connect', () => {
    for (const provider of modelProviderCatalog) {
      expect(provider.monogram).not.toBe('')
      if (provider.icon) expect(provider.icon).toMatch(/^(data:image\/svg\+xml|https:\/\/|\/)/)
      if (provider.baseUrl) expect(provider.baseUrl).toMatch(/^https:\/\//)
      if (provider.category !== 'Model APIs') expect(provider.baseUrl).toBe('')
    }
    expect(modelProviderCatalog.map((provider) => provider.id)).toEqual(
      expect.arrayContaining(['anthropic-compatible', 'nvidia-riva', 'triton']),
    )
  })

  it('only emits upstream wire formats supported by the router', () => {
    const formats = new Set(modelProviderCatalog.map((provider) => provider.apiFormat))
    expect([...formats].sort()).toEqual(['anthropic', 'openai'])
  })

  it('derives model discovery from the provider operation contract', () => {
    expect(modelProviderCatalog.find((provider) => provider.id === 'openai')?.supportsModelDiscovery)
      .toBe(true)
    expect(modelProviderCatalog.find((provider) => provider.id === 'azure-openai')?.supportsModelDiscovery)
      .toBe(false)
  })

  it('places the local serving options first', () => {
    expect(modelProviderCatalog.slice(0, 4).map((provider) => provider.name)).toEqual([
      'vLLM',
      'SGLang',
      'AMD ATOM',
      'OpenAI Compatible',
    ])
  })

  it('resolves provider marks by stable backend identity before wire format', () => {
    expect(
      findModelProviderPreset({
        backendName: 'openrouter-primary',
        baseUrl: 'https://openrouter.ai/api/v1',
        apiFormat: 'openai',
      })?.id,
    ).toBe('openrouter')
    expect(
      findModelProviderPreset({
        backendName: 'production',
        baseUrl: 'https://api.deepseek.com/v1',
        apiFormat: 'openai',
      })?.id,
    ).toBe('deepseek')
    expect(findModelProviderPreset({ apiFormat: 'openai' })?.id).toBe('openai-compatible')
    expect(findModelProviderPreset({ apiFormat: 'anthropic' })?.id).toBe('anthropic')
  })
})
