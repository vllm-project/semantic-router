import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import ModelProviderLogo from './ModelProviderLogo'
import type { ModelProviderPreset } from './modelProviderCatalog'

const provider = (patch: Partial<ModelProviderPreset>): ModelProviderPreset => ({
  id: 'example',
  name: 'Example Provider',
  description: 'Example provider contract',
  category: 'Model APIs',
  baseUrl: '',
  apiFormat: 'openai',
  authStrategy: 'bearer',
  icon: '',
  monogram: 'EX',
  supportTier: 'compatible',
  protocols: ['openai/chat-completions@1'],
  supportsModelDiscovery: false,
  featured: false,
  monochrome: false,
  ...patch,
})

describe('model provider logo', () => {
  it('uses a provider monogram when its catalog presentation has no logo', () => {
    const markup = renderToStaticMarkup(<ModelProviderLogo provider={provider({})} />)

    expect(markup).toContain('>EX</span>')
    expect(markup).not.toContain('/vllm.png')
    expect(markup).not.toContain('<img')
  })

  it('keeps the vLLM image fallback for an anonymous runtime model', () => {
    const markup = renderToStaticMarkup(<ModelProviderLogo />)

    expect(markup).toContain('src="/vllm.png"')
  })
})
