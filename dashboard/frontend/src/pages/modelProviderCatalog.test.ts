import { describe, expect, it } from 'vitest'

import { findModelProviderPreset, modelProviderCatalog } from './modelProviderCatalog'

describe('model provider catalog', () => {
  it('keeps private endpoints explicit and hosted APIs ready to connect', () => {
    for (const provider of modelProviderCatalog) {
      expect(provider.icon).toMatch(/^(data:image\/svg\+xml|https:\/\/|\/)/)
      expect(provider.monogram).not.toBe('')
      if (provider.category !== 'Model APIs') {
        expect(provider.baseUrl).toBe('')
      } else {
        expect(provider.baseUrl).toMatch(/^https:\/\//)
      }
    }
    expect(modelProviderCatalog.map((provider) => provider.id)).toEqual(
      expect.arrayContaining(['anthropic-compatible', 'nvidia-riva', 'triton']),
    )
  })

  it('only emits upstream wire formats supported by the router', () => {
    const formats = new Set(modelProviderCatalog.map((provider) => provider.apiFormat))
    expect([...formats].sort()).toEqual(['anthropic', 'openai'])
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
