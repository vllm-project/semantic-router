import { describe, expect, it } from 'vitest'

import generatedCatalog from '../generated/modelCatalog.json'
import type { CatalogProvider } from '../types/modelCatalog'
import {
  filterModelProviderPresets,
  findModelProviderPreset,
  hiddenModelProviderPresetCount,
  modelProviderCatalog,
  type ModelProviderPreset,
} from './modelProviderCatalog'

describe('model provider catalog contracts', () => {
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
    expect([...formats].sort()).toEqual(['anthropic', 'openai', 'responses'])
  })

  it('derives model discovery from the provider operation contract', () => {
    expect(
      modelProviderCatalog.find((provider) => provider.id === 'openai')?.supportsModelDiscovery,
    ).toBe(true)
    expect(
      modelProviderCatalog.find((provider) => provider.id === 'azure-openai')
        ?.supportsModelDiscovery,
    ).toBe(false)
  })

  it('projects local serving options into the start-here category', () => {
    expect(
      modelProviderCatalog
        .filter((provider) => provider.category === 'Start here')
        .map((provider) => provider.name),
    ).toEqual(expect.arrayContaining(['vLLM', 'SGLang', 'AMD ATOM', 'OpenAI Compatible']))
  })

  it('uses catalog presentation metadata for provider marks', () => {
    const baidu = modelProviderCatalog.find((provider) => provider.id === 'baidu-ai-studio')
    const openAI = modelProviderCatalog.find((provider) => provider.id === 'openai')

    expect(baidu?.icon).not.toBe('')
    expect(baidu?.monochrome).toBe(false)
    expect(openAI?.monochrome).toBe(true)
  })
})

describe('model provider catalog presentation', () => {
  it('projects a high-signal featured subset without shrinking the provider registry', () => {
    const featured = modelProviderCatalog.filter((provider) => provider.featured)

    expect(featured.map((provider) => provider.id)).toEqual(
      expect.arrayContaining([
        'openai',
        'anthropic',
        'bedrock',
        'openrouter',
        'vllm',
        'sglang',
        'ollama',
      ]),
    )
    expect(featured.length).toBeGreaterThan(0)
    expect(featured.length).toBeLessThan(modelProviderCatalog.length)
    expect(modelProviderCatalog.some((provider) => !provider.featured)).toBe(true)
  })

  it('keeps custom-only provider contracts available without claiming model mappings', () => {
    const providers = generatedCatalog.providers as CatalogProvider[]
    const customOnly = providers.find((provider) => !provider.models?.length)

    expect(customOnly).toBeDefined()
    expect(modelProviderCatalog.map((provider) => provider.id)).toContain(customOnly!.id)
    expect(modelProviderCatalog).toHaveLength(providers.length)
  })
})

describe('model provider catalog lookup and filtering', () => {
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

  it('keeps the default picker concise while search still covers the full registry', () => {
    const base = modelProviderCatalog[0]
    const providers: ModelProviderPreset[] = [
      { ...base, id: 'mainstream', name: 'Mainstream Cloud', featured: true },
      { ...base, id: 'specialized-edge', name: 'Specialized Edge', featured: false },
    ]

    expect(filterModelProviderPresets(providers, '', false).map((provider) => provider.id)).toEqual(
      ['mainstream'],
    )
    expect(filterModelProviderPresets(providers, '', true)).toHaveLength(2)
    expect(
      filterModelProviderPresets(providers, 'specialized', false).map((provider) => provider.id),
    ).toEqual(['specialized-edge'])
    expect(hiddenModelProviderPresetCount(providers)).toBe(1)
  })

  it('falls back to the full registry when an older catalog has no featured metadata', () => {
    const providers = modelProviderCatalog.slice(0, 2).map((provider) => ({
      ...provider,
      featured: false,
    }))

    expect(filterModelProviderPresets(providers, '', false)).toHaveLength(2)
    expect(hiddenModelProviderPresetCount(providers)).toBe(0)
  })
})
